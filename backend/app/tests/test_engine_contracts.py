from __future__ import annotations

from pathlib import Path

import pytest

from app.engine import (
    EngineError,
    EngineErrorCode,
    EngineSettings,
    EngineState,
    GenerationRequest,
    ModelArtifact,
)


def test_engine_error_has_stable_safe_response() -> None:
    error = EngineError(
        code=EngineErrorCode.RUNTIME_START_FAILED,
        message="The selected runtime could not start.",
        stage="start",
        details={"hint": "Check runtime compatibility"},
    )

    assert error.code.value == "RUNTIME_START_FAILED"
    assert error.as_response("corr-123") == {
        "error": {
            "code": "RUNTIME_START_FAILED",
            "message": "The selected runtime could not start.",
            "stage": "start",
            "correlation_id": "corr-123",
            "details": {"hint": "Check runtime compatibility"},
        }
    }


def test_engine_settings_reject_invalid_context_and_mode() -> None:
    with pytest.raises(ValueError, match="context_size"):
        EngineSettings(context_size=128)
    with pytest.raises(ValueError, match="mode"):
        EngineSettings(mode="mystery")


def test_generation_request_rejects_empty_prompt_and_invalid_limits() -> None:
    with pytest.raises(ValueError, match="prompt"):
        GenerationRequest(prompt="   ")
    with pytest.raises(ValueError, match="max_tokens"):
        GenerationRequest(prompt="Hello", max_tokens=0)


def test_model_artifact_preserves_model_identity_and_path() -> None:
    path = Path("models/example.gguf")
    artifact = ModelArtifact(
        model_id="local-example",
        path=path,
        sha256="abc123",
        verified=True,
    )

    assert artifact.model_id == "local-example"
    assert artifact.path == path
    assert artifact.verified is True


def test_engine_states_are_explicit() -> None:
    assert {state.value for state in EngineState} == {
        "stopped", "starting", "ready", "generating", "stopping", "failed"
    }
