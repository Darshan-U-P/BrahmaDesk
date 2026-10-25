"""Lifecycle supervisor for Brahma Engine backends.

The supervisor owns orchestration and state transitions. Backend adapters own
their implementation-specific process and inference details. The supervisor
never discovers or terminates unrelated external processes.
"""
from __future__ import annotations

import asyncio
from pathlib import Path
from typing import AsyncIterator

from .contracts import (
    EngineError,
    EngineErrorCode,
    EngineHealth,
    EngineSettings,
    EngineState,
    GenerationChunk,
    GenerationRequest,
    InferenceBackend,
    ModelArtifact,
    ResourcePlan,
)


class EngineSupervisor:
    """Serialize lifecycle operations and expose consistent engine state."""

    def __init__(self, backend: InferenceBackend) -> None:
        self._backend = backend
        self._operation_lock = asyncio.Lock()
        self._state = EngineState.STOPPED
        self._model: ModelArtifact | None = None
        self._settings: EngineSettings | None = None
        self._plan: ResourcePlan | None = None
        self._last_error: EngineError | None = None

    @property
    def state(self) -> EngineState:
        return self._state

    @property
    def selected_model(self) -> ModelArtifact | None:
        return self._model

    @property
    def resource_plan(self) -> ResourcePlan | None:
        return self._plan

    @property
    def last_error(self) -> EngineError | None:
        return self._last_error

    async def prepare(
        self, model_path: Path, settings: EngineSettings
    ) -> tuple[ModelArtifact, ResourcePlan]:
        """Validate a model and produce a plan without starting inference."""
        try:
            model = await self._backend.inspect_model(model_path)
            plan = await self._backend.plan(model, settings)
            self._model = model
            self._settings = settings
            self._plan = plan
            self._last_error = None
            return model, plan
        except EngineError as exc:
            self._last_error = exc
            raise
        except Exception as exc:
            error = EngineError(
                EngineErrorCode.CONFIGURATION_INVALID,
                "The model could not be inspected or planned.",
                "prepare",
                {"reason": str(exc)},
            )
            self._last_error = error
            raise error from exc

    async def start(
        self, model_path: Path, settings: EngineSettings
    ) -> EngineHealth:
        async with self._operation_lock:
            if self._state in {EngineState.STARTING, EngineState.STOPPING, EngineState.GENERATING}:
                raise EngineError(
                    EngineErrorCode.ENGINE_NOT_READY,
                    f"Cannot start while the engine is {self._state.value}.",
                    "start",
                )
            self._state = EngineState.STARTING
            try:
                model, plan = await self.prepare(model_path, settings)
                health = await self._backend.start(model, settings)
                if health.state != EngineState.READY or not health.generation_ready:
                    self._state = EngineState.FAILED
                    error = EngineError(
                        EngineErrorCode.RUNTIME_START_FAILED,
                        health.message or "The backend started but did not report generation readiness.",
                        "start",
                        {"backend": self._backend.name, "health_state": health.state.value},
                    )
                    self._last_error = error
                    raise error
                self._model, self._settings, self._plan = model, settings, plan
                self._state = EngineState.READY
                self._last_error = None
                return health
            except EngineError:
                # A backend may have spawned its owned process before discovering
                # that startup did not complete. Always ask it to clean up; adapters
                # must stop only resources they own. Preserve the original startup
                # error if cleanup also fails.
                try:
                    await self._backend.stop()
                except Exception:
                    pass
                self._state = EngineState.FAILED
                raise
            except Exception as exc:
                try:
                    await self._backend.stop()
                except Exception:
                    pass
                self._state = EngineState.FAILED
                error = EngineError(
                    EngineErrorCode.RUNTIME_START_FAILED,
                    "The inference backend failed to start.",
                    "start",
                    {"backend": self._backend.name, "reason": str(exc)},
                )
                self._last_error = error
                raise error from exc

    async def health(self) -> EngineHealth:
        backend_health = await self._backend.health()
        if self._state == EngineState.FAILED:
            return EngineHealth(
                state=EngineState.FAILED,
                backend=self._backend.name,
                model_id=self._model.model_id if self._model else None,
                process_owned=backend_health.process_owned,
                generation_ready=False,
                message=self._last_error.message if self._last_error else backend_health.message,
                diagnostics=backend_health.diagnostics,
            )
        if self._state == EngineState.GENERATING:
            return EngineHealth(
                state=EngineState.GENERATING,
                backend=self._backend.name,
                model_id=self._model.model_id if self._model else None,
                process_owned=backend_health.process_owned,
                generation_ready=backend_health.generation_ready,
                message=backend_health.message,
                diagnostics=backend_health.diagnostics,
            )
        return backend_health

    async def generate(
        self, request: GenerationRequest
    ) -> AsyncIterator[GenerationChunk]:
        async with self._operation_lock:
            if self._state != EngineState.READY or self._model is None:
                raise EngineError(
                    EngineErrorCode.ENGINE_NOT_READY,
                    "Load and start a model before requesting generation.",
                    "generate",
                )
            self._state = EngineState.GENERATING
            try:
                async for chunk in self._backend.generate(request):
                    yield chunk
                self._state = EngineState.READY
            except EngineError as exc:
                self._last_error = exc
                self._state = EngineState.FAILED
                raise
            except Exception as exc:
                error = EngineError(
                    EngineErrorCode.GENERATION_FAILED,
                    "Text generation failed in the inference backend.",
                    "generate",
                    {"backend": self._backend.name, "reason": str(exc)},
                )
                self._last_error = error
                self._state = EngineState.FAILED
                raise error from exc
            finally:
                if self._state == EngineState.GENERATING:
                    self._state = EngineState.READY

    async def stop(self) -> None:
        async with self._operation_lock:
            if self._state == EngineState.STOPPED:
                return
            self._state = EngineState.STOPPING
            try:
                await self._backend.stop()
            except Exception as exc:
                error = EngineError(
                    EngineErrorCode.RUNTIME_START_FAILED,
                    "The inference backend could not be stopped cleanly.",
                    "stop",
                    {"backend": self._backend.name, "reason": str(exc)},
                )
                self._last_error = error
                self._state = EngineState.FAILED
                raise error from exc
            self._state = EngineState.STOPPED
            self._model = None
            self._settings = None
            self._plan = None
            self._last_error = None
