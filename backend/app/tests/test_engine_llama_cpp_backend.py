from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from app.engine.contracts import EngineError, EngineErrorCode, EngineSettings
from app.engine.llama_cpp_backend import LlamaCppServerBackend


def test_inspect_model_rejects_missing_file(tmp_path: Path) -> None:
    backend = LlamaCppServerBackend(tmp_path / "llama-server.exe")

    with pytest.raises(EngineError) as error:
        asyncio.run(backend.inspect_model(tmp_path / "missing.gguf"))

    assert error.value.code == EngineErrorCode.MODEL_NOT_FOUND


def test_inspect_model_rejects_invalid_gguf_header(tmp_path: Path) -> None:
    model = tmp_path / "invalid.gguf"
    model.write_bytes(b"NOPE" + b"x" * 32)
    backend = LlamaCppServerBackend(tmp_path / "llama-server.exe")

    with pytest.raises(EngineError) as error:
        asyncio.run(backend.inspect_model(model))

    assert error.value.code == EngineErrorCode.MODEL_INVALID_GGUF


def test_inspect_model_returns_sha256_for_gguf(tmp_path: Path) -> None:
    model = tmp_path / "valid.gguf"
    model.write_bytes(b"GGUF" + b"x" * 32)
    backend = LlamaCppServerBackend(tmp_path / "llama-server.exe")

    artifact = asyncio.run(backend.inspect_model(model))

    assert artifact.path == model.resolve()
    assert artifact.verified is True
    assert len(artifact.sha256 or "") == 64
    assert artifact.metadata["size_bytes"] == model.stat().st_size


def test_plan_requires_existing_runtime(tmp_path: Path) -> None:
    model = tmp_path / "valid.gguf"
    model.write_bytes(b"GGUF" + b"x" * 32)
    backend = LlamaCppServerBackend(tmp_path / "missing-server.exe")
    artifact = asyncio.run(backend.inspect_model(model))

    with pytest.raises(EngineError) as error:
        asyncio.run(backend.plan(artifact, EngineSettings(context_size=2048, mode="cpu")))

    assert error.value.code == EngineErrorCode.RUNTIME_NOT_INSTALLED


def test_cpu_plan_reports_zero_estimated_vram(tmp_path: Path) -> None:
    model = tmp_path / "valid.gguf"
    model.write_bytes(b"GGUF" + b"x" * 1024)
    executable = tmp_path / "llama-server.exe"
    executable.write_bytes(b"placeholder")
    backend = LlamaCppServerBackend(executable)
    artifact = asyncio.run(backend.inspect_model(model))

    plan = asyncio.run(backend.plan(artifact, EngineSettings(context_size=2048, mode="cpu")))

    assert plan.selected_mode == "cpu"
    assert plan.context_size == 2048
    assert plan.estimated_vram_gb == 0.0


def test_stop_does_not_touch_process_when_none_is_owned(tmp_path: Path) -> None:
    backend = LlamaCppServerBackend(tmp_path / "llama-server.exe")

    asyncio.run(backend.stop())

    assert backend._process is None
