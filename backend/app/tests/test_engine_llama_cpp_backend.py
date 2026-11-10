from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from app.engine.contracts import (
    EngineError,
    EngineErrorCode,
    EngineSettings,
    GenerationRequest,
)
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



def test_generate_uses_full_messages_and_streams_chunks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    backend = LlamaCppServerBackend(tmp_path / "llama-server.exe")
    backend._base_url = "http://127.0.0.1:12345"
    backend._model = type("Model", (), {"model_id": "test-model"})()

    async def endpoint_ready() -> bool:
        return True

    class FakeResponse:
        status_code = 200

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def aiter_lines(self):
            yield 'data: {"choices":[{"delta":{"content":"Hello"},"finish_reason":null}]}'
            yield 'data: {"choices":[{"delta":{"content":" world"},"finish_reason":null}]}'
            yield 'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}'
            yield "data: [DONE]"

        def raise_for_status(self):
            return None

    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        def stream(self, method, url, *, json, headers):
            assert method == "POST"
            assert url.endswith("/v1/chat/completions")
            assert json["stream"] is True
            assert json["messages"] == [
                {"role": "system", "content": "Be concise."},
                {"role": "user", "content": "Hello"},
            ]
            return FakeResponse()

    monkeypatch.setattr(backend, "_endpoint_ready", endpoint_ready)
    monkeypatch.setattr("app.engine.llama_cpp_backend.httpx.AsyncClient", FakeClient)

    async def collect():
        chunks = []
        async for chunk in backend.generate(
            GenerationRequest(
                prompt="Hello",
                messages=[
                    {"role": "system", "content": "Be concise."},
                    {"role": "user", "content": "Hello"},
                ],
                stream=True,
            )
        ):
            chunks.append(chunk)
        return chunks

    chunks = asyncio.run(collect())
    assert "".join(chunk.text for chunk in chunks) == "Hello world"
    assert any(chunk.finished and chunk.finish_reason == "stop" for chunk in chunks)



def test_parse_tool_calls_validates_and_normalizes_arguments(tmp_path: Path) -> None:
    backend = LlamaCppServerBackend(tmp_path / "llama-server.exe")

    calls = backend._parse_tool_calls([
        {
            "id": "call-1",
            "function": {
                "name": "filesystem.list_files",
                "arguments": '{"path":"."}',
            },
        },
        {
            "id": "call-2",
            "function": {
                "name": "system.get_platform_info",
                "arguments": {},
            },
        },
    ])

    assert len(calls) == 2
    assert calls[0].id == "call-1"
    assert calls[0].name == "filesystem.list_files"
    assert calls[0].arguments == {"path": "."}
    assert calls[1].arguments == {}


def test_parse_tool_calls_rejects_malformed_or_non_object_arguments(tmp_path: Path) -> None:
    backend = LlamaCppServerBackend(tmp_path / "llama-server.exe")

    with pytest.raises(ValueError, match="malformed JSON"):
        backend._parse_tool_calls([
            {
                "id": "call-1",
                "function": {"name": "test.tool", "arguments": "{bad"},
            }
        ])

    with pytest.raises(ValueError, match="JSON object"):
        backend._parse_tool_calls([
            {
                "id": "call-1",
                "function": {"name": "test.tool", "arguments": '["not", "an", "object"]'},
            }
        ])


def test_generate_non_streaming_returns_structured_tool_calls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    backend = LlamaCppServerBackend(tmp_path / "llama-server.exe")
    backend._base_url = "http://127.0.0.1:12345"
    backend._model = type("Model", (), {"model_id": "test-model"})()

    async def endpoint_ready() -> bool:
        return True

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "choices": [{
                    "finish_reason": "tool_calls",
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [{
                            "id": "call-abc",
                            "type": "function",
                            "function": {
                                "name": "system.get_platform_info",
                                "arguments": "{}",
                            },
                        }],
                    },
                }]
            }

    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def post(self, url, *, json):
            assert url.endswith("/v1/chat/completions")
            assert json["stream"] is False
            assert json["tool_choice"] == "auto"
            assert len(json["tools"]) == 1
            return FakeResponse()

    monkeypatch.setattr(backend, "_endpoint_ready", endpoint_ready)
    monkeypatch.setattr("app.engine.llama_cpp_backend.httpx.AsyncClient", FakeClient)

    async def collect():
        return [
            chunk async for chunk in backend.generate(
                GenerationRequest(
                    prompt="Inspect this machine",
                    messages=[{"role": "user", "content": "Inspect this machine"}],
                    stream=False,
                    tools=[{
                        "type": "function",
                        "function": {
                            "name": "system.get_platform_info",
                            "description": "Inspect platform",
                            "parameters": {"type": "object", "properties": {}},
                        },
                    }],
                )
            )
        ]

    chunks = asyncio.run(collect())
    assert len(chunks) == 1
    assert chunks[0].finished is True
    assert chunks[0].finish_reason == "tool_calls"
    assert chunks[0].tool_calls[0].id == "call-abc"
    assert chunks[0].tool_calls[0].arguments == {}
