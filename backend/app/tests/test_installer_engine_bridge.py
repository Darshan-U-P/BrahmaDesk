from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from fastapi import HTTPException

from app import installer


def test_installer_readiness_runs_real_generation_smoke_test(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = tmp_path / "model.gguf"
    model.write_bytes(b"GGUF" + b"x" * 128)
    registered = {
        "model_id": "local-test",
        "path": str(model),
        "size_bytes": model.stat().st_size,
        "verified": True,
    }
    monkeypatch.setattr(installer, "_registry_rows", lambda: {"local-test": registered})

    async def configuration() -> dict[str, object]:
        return {"config": {"model_id": "local-test"}}

    async def smoke_test(model_id: str) -> dict[str, str]:
        assert model_id == "local-test"
        return {
            "generation_test": "passed",
            "runtime": "llama.cpp",
            "sample_output": "READY",
        }

    monkeypatch.setattr(installer, "get_inference_configuration", configuration)
    monkeypatch.setattr(installer, "_run_runtime_smoke_test", smoke_test)

    result = asyncio.run(installer.test_model_readiness("local-test"))

    assert result["status"] == "ready"
    assert result["runtime_inference_test"] == "passed"
    assert result["sample_output"] == "READY"


def test_installer_readiness_does_not_hide_runtime_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = tmp_path / "model.gguf"
    model.write_bytes(b"GGUF" + b"x" * 128)
    registered = {
        "model_id": "local-test",
        "path": str(model),
        "size_bytes": model.stat().st_size,
        "verified": True,
    }
    monkeypatch.setattr(installer, "_registry_rows", lambda: {"local-test": registered})

    async def configuration() -> dict[str, object]:
        return {"config": {"model_id": "local-test"}}

    async def smoke_test(model_id: str) -> dict[str, str]:
        raise HTTPException(status_code=502, detail="Generation smoke test failed")

    monkeypatch.setattr(installer, "get_inference_configuration", configuration)
    monkeypatch.setattr(installer, "_run_runtime_smoke_test", smoke_test)

    with pytest.raises(HTTPException) as error:
        asyncio.run(installer.test_model_readiness("local-test"))

    assert error.value.status_code == 502
    assert error.value.detail == "Generation smoke test failed"


def test_engine_v1_installer_smoke_test_uses_configured_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = tmp_path / "model.gguf"
    model.write_bytes(b"GGUF" + b"x" * 128)
    executable = tmp_path / "llama-server.exe"
    executable.write_bytes(b"test executable")
    registered = {
        "model_id": "local-test",
        "path": str(model),
        "size_bytes": model.stat().st_size,
        "verified": True,
    }
    monkeypatch.setattr(installer, "_registry_rows", lambda: {"local-test": registered})
    monkeypatch.setattr(installer, "REGISTRY_DB", tmp_path / "registry.sqlite3")
    monkeypatch.setattr(installer, "DATA_DIR", tmp_path)
    monkeypatch.setenv("BRAHMA_LLAMA_SERVER_EXE", str(executable))
    monkeypatch.setenv("BRAHMA_ENGINE_STARTUP_TIMEOUT", "42")

    async def configuration() -> dict[str, object]:
        return {
            "config": {
                "model_id": "local-test",
                "mode": "cpu",
                "context_size": 2048,
            }
        }

    calls: dict[str, object] = {}

    async def probe(exe: Path, model_path: Path, **kwargs: object) -> dict[str, object]:
        calls.update({"executable": exe, "model_path": model_path, **kwargs})
        return {
            "status": "passed",
            "backend": "llama.cpp-server-owned",
            "generation_test": "passed",
            "sample_output": "BRAHMA_ENGINE_OK",
            "resource_plan": {"mode": "cpu", "context_size": 2048},
        }

    monkeypatch.setattr(installer, "get_inference_configuration", configuration)
    monkeypatch.setattr("app.engine.runtime_probe.probe_runtime", probe)

    result = asyncio.run(installer._run_engine_v1_smoke_test("local-test"))

    assert result["engine_v1"] is True
    assert result["generation_test"] == "passed"
    assert result["sample_output"] == "BRAHMA_ENGINE_OK"
    assert calls["executable"] == executable
    assert calls["model_path"] == model
    assert calls["mode"] == "cpu"
    assert calls["context_size"] == 2048
    assert calls["startup_timeout"] == 42.0


def test_engine_v1_installer_smoke_test_rejects_missing_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = tmp_path / "model.gguf"
    model.write_bytes(b"GGUF" + b"x" * 128)
    monkeypatch.setattr(
        installer,
        "_registry_rows",
        lambda: {
            "local-test": {
                "model_id": "local-test",
                "path": str(model),
                "size_bytes": model.stat().st_size,
                "verified": True,
            }
        },
    )
    monkeypatch.setattr(installer, "get_inference_configuration", lambda: None)
    monkeypatch.delenv("BRAHMA_LLAMA_SERVER_EXE", raising=False)
    monkeypatch.setattr(installer, "REGISTRY_DB", tmp_path / "registry.sqlite3")
    monkeypatch.setattr(installer, "DATA_DIR", tmp_path)

    async def configuration() -> dict[str, object]:
        return {"config": {"model_id": "local-test", "mode": "cpu", "context_size": 2048}}

    monkeypatch.setattr(installer, "get_inference_configuration", configuration)

    with pytest.raises(HTTPException) as error:
        asyncio.run(installer._run_engine_v1_smoke_test("local-test"))

    assert error.value.status_code == 409
    assert "configured llama-server.exe" in error.value.detail

