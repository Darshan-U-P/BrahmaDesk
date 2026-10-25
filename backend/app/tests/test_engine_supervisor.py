from __future__ import annotations

from pathlib import Path

import pytest

from app.engine import (
    EngineError,
    EngineErrorCode,
    EngineHealth,
    EngineSettings,
    EngineState,
    EngineSupervisor,
    GenerationChunk,
    GenerationRequest,
    ModelArtifact,
    ResourcePlan,
)


class FakeBackend:
    name = "fake"

    def __init__(self, *, fail_start: bool = False, fail_generate: bool = False) -> None:
        self.fail_start = fail_start
        self.fail_generate = fail_generate
        self.started = False
        self.stopped = False
        self.generate_calls = 0

    async def inspect_model(self, path: Path) -> ModelArtifact:
        if not path.exists():
            raise EngineError(
                EngineErrorCode.MODEL_NOT_FOUND,
                "Model file does not exist.",
                "inspect_model",
            )
        return ModelArtifact(model_id="test-model", path=path, verified=True)

    async def plan(self, model: ModelArtifact, settings: EngineSettings) -> ResourcePlan:
        return ResourcePlan(selected_mode=settings.mode, context_size=settings.context_size)

    async def start(self, model: ModelArtifact, settings: EngineSettings) -> EngineHealth:
        if self.fail_start:
            raise RuntimeError("simulated startup failure")
        self.started = True
        return EngineHealth(
            state=EngineState.READY,
            backend=self.name,
            model_id=model.model_id,
            process_owned=True,
            generation_ready=True,
        )

    async def health(self) -> EngineHealth:
        return EngineHealth(
            state=EngineState.READY if self.started and not self.stopped else EngineState.STOPPED,
            backend=self.name,
            process_owned=self.started and not self.stopped,
            generation_ready=self.started and not self.stopped,
        )

    async def generate(self, request: GenerationRequest):
        self.generate_calls += 1
        if self.fail_generate:
            raise RuntimeError("simulated generation failure")
        yield GenerationChunk(text="hello ")
        yield GenerationChunk(text="world", finished=True, finish_reason="stop")

    async def stop(self) -> None:
        self.stopped = True
        self.started = False


@pytest.mark.asyncio
async def test_supervisor_start_generate_and_stop(tmp_path: Path) -> None:
    model_path = tmp_path / "model.gguf"
    model_path.write_bytes(b"GGUF" + b"\0" * 16)
    backend = FakeBackend()
    supervisor = EngineSupervisor(backend)

    health = await supervisor.start(model_path, EngineSettings(context_size=2048, mode="cpu"))
    assert health.generation_ready
    assert supervisor.state == EngineState.READY
    assert supervisor.selected_model is not None
    assert supervisor.resource_plan is not None

    chunks = [chunk async for chunk in supervisor.generate(GenerationRequest(prompt="Say hello"))]
    assert "".join(chunk.text for chunk in chunks) == "hello world"
    assert supervisor.state == EngineState.READY

    await supervisor.stop()
    assert supervisor.state == EngineState.STOPPED
    assert backend.stopped


@pytest.mark.asyncio
async def test_supervisor_rejects_generation_before_start() -> None:
    supervisor = EngineSupervisor(FakeBackend())
    with pytest.raises(EngineError) as exc:
        _ = [chunk async for chunk in supervisor.generate(GenerationRequest(prompt="Hello"))]
    assert exc.value.code == EngineErrorCode.ENGINE_NOT_READY


@pytest.mark.asyncio
async def test_supervisor_reports_start_failure(tmp_path: Path) -> None:
    model_path = tmp_path / "model.gguf"
    model_path.write_bytes(b"GGUF" + b"\0" * 16)
    supervisor = EngineSupervisor(FakeBackend(fail_start=True))

    with pytest.raises(EngineError) as exc:
        await supervisor.start(model_path, EngineSettings())

    assert exc.value.code == EngineErrorCode.RUNTIME_START_FAILED
    assert supervisor.state == EngineState.FAILED


@pytest.mark.asyncio
async def test_supervisor_wraps_generation_failure(tmp_path: Path) -> None:
    model_path = tmp_path / "model.gguf"
    model_path.write_bytes(b"GGUF" + b"\0" * 16)
    supervisor = EngineSupervisor(FakeBackend(fail_generate=True))
    await supervisor.start(model_path, EngineSettings())

    with pytest.raises(EngineError) as exc:
        _ = [chunk async for chunk in supervisor.generate(GenerationRequest(prompt="Hello"))]

    assert exc.value.code == EngineErrorCode.GENERATION_FAILED
    assert supervisor.state == EngineState.FAILED


@pytest.mark.asyncio
async def test_supervisor_rejects_missing_model(tmp_path: Path) -> None:
    supervisor = EngineSupervisor(FakeBackend())

    with pytest.raises(EngineError) as exc:
        await supervisor.start(tmp_path / "missing.gguf", EngineSettings())

    assert exc.value.code == EngineErrorCode.MODEL_NOT_FOUND
    assert supervisor.state == EngineState.FAILED
