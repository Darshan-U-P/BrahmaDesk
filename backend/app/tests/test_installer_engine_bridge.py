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
