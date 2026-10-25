"""Owned-process llama.cpp adapter for the Brahma Engine contract.

This adapter is intentionally opt-in: it owns only the Popen process it creates
and binds it to a newly allocated loopback port. It never attaches to or kills
an unrelated llama-server process.
"""
from __future__ import annotations

import asyncio
import hashlib
import socket
import subprocess
import time
from pathlib import Path
from typing import AsyncIterator

import httpx

from .contracts import (
    EngineError,
    EngineErrorCode,
    EngineHealth,
    EngineSettings,
    EngineState,
    GenerationChunk,
    GenerationRequest,
    ModelArtifact,
    ResourcePlan,
)


class LlamaCppServerBackend:
    """InferenceBackend implementation backed by a private llama-server process."""

    def __init__(
        self,
        executable: Path,
        *,
        host: str = "127.0.0.1",
        startup_timeout: float = 90.0,
        request_timeout: float = 120.0,
    ) -> None:
        self._executable = Path(executable).expanduser().resolve()
        self._host = host
        self._startup_timeout = startup_timeout
        self._request_timeout = request_timeout
        self._process: subprocess.Popen | None = None
        self._base_url: str | None = None
        self._model: ModelArtifact | None = None
        self._settings: EngineSettings | None = None
        self._last_message: str | None = None

    @property
    def name(self) -> str:
        return "llama.cpp-server-owned"

    async def inspect_model(self, path: Path) -> ModelArtifact:
        candidate = Path(path).expanduser().resolve()
        if not candidate.is_file():
            raise EngineError(
                EngineErrorCode.MODEL_NOT_FOUND,
                "The selected GGUF model file does not exist.",
                "inspect_model",
                {"path": str(candidate)},
            )
        if candidate.suffix.casefold() != ".gguf":
            raise EngineError(
                EngineErrorCode.MODEL_INVALID_GGUF,
                "The selected model must use the .gguf file extension.",
                "inspect_model",
            )
        try:
            with candidate.open("rb") as stream:
                magic = stream.read(4)
        except OSError as exc:
            raise EngineError(
                EngineErrorCode.MODEL_NOT_FOUND,
                "The model file could not be read.",
                "inspect_model",
                {"reason": str(exc)},
            ) from exc
        if magic != b"GGUF":
            raise EngineError(
                EngineErrorCode.MODEL_INVALID_GGUF,
                "The model file does not contain the GGUF magic header.",
                "inspect_model",
            )
        digest = await asyncio.to_thread(self._sha256, candidate)
        return ModelArtifact(
            model_id=candidate.stem,
            path=candidate,
            sha256=digest,
            verified=True,
            metadata={"size_bytes": candidate.stat().st_size},
        )

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    async def plan(
        self, model: ModelArtifact, settings: EngineSettings
    ) -> ResourcePlan:
        if not self._executable.is_file():
            raise EngineError(
                EngineErrorCode.RUNTIME_NOT_INSTALLED,
                "Brahma Desk could not find the configured llama-server executable.",
                "plan",
                {"executable": str(self._executable)},
            )
        if settings.mode == "cpu":
            selected_mode = "cpu"
        elif settings.mode in {"gpu", "hybrid"}:
            selected_mode = settings.mode
        else:
            selected_mode = "auto"
        # GGUF metadata is not parsed here. This is a transparent weight-size
        # estimate, not a hardware compatibility guarantee.
        size_gb = model.path.stat().st_size / (1024 ** 3)
        return ResourcePlan(
            selected_mode=selected_mode,
            context_size=settings.context_size,
            estimated_ram_gb=round(size_gb + 2.0, 2),
            estimated_vram_gb=round(size_gb + 1.0, 2) if selected_mode != "cpu" else 0.0,
            warnings=("Memory estimates are conservative heuristics, not a hardware guarantee.",),
        )

    @staticmethod
    def _available_port(host: str) -> int:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.bind((host, 0))
            return int(probe.getsockname()[1])

    async def start(
        self, model: ModelArtifact, settings: EngineSettings
    ) -> EngineHealth:
        if self._process is not None and self._process.poll() is None:
            if self._model and self._model.path == model.path and self._settings == settings:
                return await self.health()
            await self.stop()
        if not self._executable.is_file():
            raise EngineError(
                EngineErrorCode.RUNTIME_NOT_INSTALLED,
                "The configured llama-server executable does not exist.",
                "start",
                {"executable": str(self._executable)},
            )

        port = self._available_port(self._host)
        gpu_layers = settings.gpu_layers
        if gpu_layers is None:
            gpu_layers = 0 if settings.mode == "cpu" else (20 if settings.mode == "hybrid" else 999)
        threads = settings.threads or max(1, min((__import__("os").cpu_count() or 4) - 1, 16))
        command = [
            str(self._executable), "-m", str(model.path),
            "--alias", model.model_id, "--host", self._host, "--port", str(port),
            "-t", str(threads), "-c", str(settings.context_size),
            "-ngl", str(gpu_layers), "--no-webui",
        ]
        try:
            self._process = subprocess.Popen(
                command,
                cwd=str(self._executable.parent),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except OSError as exc:
            self._process = None
            raise EngineError(
                EngineErrorCode.RUNTIME_START_FAILED,
                "Could not start the owned llama.cpp process.",
                "start",
                {"reason": str(exc)},
            ) from exc

        self._base_url = f"http://{self._host}:{port}"
        self._model = model
        self._settings = settings
        deadline = time.monotonic() + self._startup_timeout
        while time.monotonic() < deadline:
            if self._process.poll() is not None:
                return_code = self._process.returncode
                self._last_message = f"llama-server exited during startup (code {return_code})."
                await self.stop()
                raise EngineError(
                    EngineErrorCode.RUNTIME_START_FAILED,
                    self._last_message,
                    "start",
                    {"return_code": return_code},
                )
            if await self._endpoint_ready():
                return EngineHealth(
                    state=EngineState.READY,
                    backend=self.name,
                    model_id=model.model_id,
                    process_owned=True,
                    generation_ready=True,
                    diagnostics={"base_url": self._base_url, "pid": self._process.pid},
                )
            await asyncio.sleep(0.5)

        await self.stop()
        raise EngineError(
            EngineErrorCode.RUNTIME_START_FAILED,
            "Timed out waiting for the owned llama.cpp process to become ready.",
            "start",
            {"timeout_seconds": self._startup_timeout},
        )

    async def _endpoint_ready(self) -> bool:
        if not self._base_url:
            return False
        try:
            async with httpx.AsyncClient(timeout=1.5) as client:
                response = await client.get(f"{self._base_url}/v1/models")
                return response.status_code == 200
        except httpx.HTTPError:
            return False

    async def health(self) -> EngineHealth:
        alive = self._process is not None and self._process.poll() is None
        ready = alive and await self._endpoint_ready()
        return EngineHealth(
            state=EngineState.READY if ready else (EngineState.STOPPED if not alive else EngineState.STARTING),
            backend=self.name,
            model_id=self._model.model_id if self._model else None,
            process_owned=alive,
            generation_ready=bool(ready),
            message=self._last_message,
            diagnostics={
                "base_url": self._base_url,
                "pid": self._process.pid if alive and self._process else None,
            },
        )

    async def generate(
        self, request: GenerationRequest
    ) -> AsyncIterator[GenerationChunk]:
        if not await self._endpoint_ready() or not self._base_url or not self._model:
            raise EngineError(
                EngineErrorCode.ENGINE_NOT_READY,
                "The owned llama.cpp process is not ready for generation.",
                "generate",
            )
        try:
            async with httpx.AsyncClient(timeout=self._request_timeout) as client:
                response = await client.post(
                    f"{self._base_url}/v1/chat/completions",
                    json={
                        "model": self._model.model_id,
                        "messages": [{"role": "user", "content": request.prompt}],
                        "max_tokens": request.max_tokens,
                        "temperature": request.temperature,
                        "stream": False,
                        **request.options,
                    },
                )
                response.raise_for_status()
                payload = response.json()
                choices = payload.get("choices") or []
                answer = choices[0].get("message", {}).get("content", "") if choices else ""
                if not isinstance(answer, str) or not answer.strip():
                    raise ValueError("The model returned no text")
        except (httpx.HTTPError, ValueError, KeyError, TypeError) as exc:
            raise EngineError(
                EngineErrorCode.GENERATION_FAILED,
                "The owned llama.cpp process failed to generate text.",
                "generate",
                {"reason": str(exc)},
            ) from exc
        yield GenerationChunk(text=answer, finished=True, finish_reason="stop")

    async def stop(self) -> None:
        process = self._process
        if process is not None:
            try:
                if process.poll() is None:
                    process.terminate()
                    try:
                        await asyncio.to_thread(process.wait, timeout=10)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        await asyncio.to_thread(process.wait, timeout=5)
            finally:
                self._process = None
                self._base_url = None
                self._model = None
                self._settings = None
