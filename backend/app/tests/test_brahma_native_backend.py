from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from app.engine.brahma_native_backend import BrahmaNativeBackend
from app.engine.contracts import EngineSettings, GenerationRequest


def make_gguf(path: Path) -> Path:
    path.write_bytes(b"GGUF" + b"\x00" * 32)
    return path


@pytest.mark.asyncio
async def test_inspect_model_rejects_missing_file(tmp_path: Path) -> None:
    backend = BrahmaNativeBackend(tmp_path / "brahma-engine.exe")

    with pytest.raises(Exception, match="does not exist"):
        await backend.inspect_model(tmp_path / "missing.gguf")


@pytest.mark.asyncio
async def test_inspect_model_rejects_invalid_gguf_header(tmp_path: Path) -> None:
    model = tmp_path / "bad.gguf"
    model.write_bytes(b"NOPE" + b"\x00" * 32)
    backend = BrahmaNativeBackend(tmp_path / "brahma-engine.exe")

    with pytest.raises(Exception, match="GGUF magic header"):
        await backend.inspect_model(model)


@pytest.mark.asyncio
async def test_start_requires_native_executable(tmp_path: Path) -> None:
    model = make_gguf(tmp_path / "model.gguf")
    backend = BrahmaNativeBackend(tmp_path / "missing-engine.exe")
    artifact = await backend.inspect_model(model)

    with pytest.raises(Exception, match="could not find brahma-engine.exe"):
        await backend.plan(artifact, EngineSettings())


class FakeStream:
    def __init__(self, chunks: list[bytes]) -> None:
        self.chunks = list(chunks)

    async def read(self, _size: int = -1) -> bytes:
        await asyncio.sleep(0)
        return self.chunks.pop(0) if self.chunks else b""


class FakeProcess:
    def __init__(self) -> None:
        self.stdout = FakeStream([
            b"Brahma Engine 0.1.0 | llama.cpp backend\nModel: model.gguf\n",
            b"\nHello ",
            b"world!",
        ])
        self.stderr = FakeStream([])
        self.returncode: int | None = None
        self.pid = 12345

    async def wait(self) -> int:
        self.returncode = 0
        return 0

    def kill(self) -> None:
        self.returncode = -9


@pytest.mark.asyncio
async def test_generate_streams_only_response_not_cli_banner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    executable = tmp_path / "brahma-engine.exe"
    executable.write_bytes(b"placeholder")
    model_path = make_gguf(tmp_path / "model.gguf")
    backend = BrahmaNativeBackend(executable)
    artifact = await backend.inspect_model(model_path)
    await backend.start(
        artifact,
        EngineSettings(context_size=2048, mode="cpu", threads=2, gpu_layers=0),
    )

    fake_process = FakeProcess()
    captured: dict[str, object] = {}

    async def fake_create_subprocess_exec(*args: object, **kwargs: object) -> FakeProcess:
        captured["args"] = args
        captured["kwargs"] = kwargs
        return fake_process

    monkeypatch.setattr(
        asyncio,
        "create_subprocess_exec",
        fake_create_subprocess_exec,
    )

    chunks = [
        chunk
        async for chunk in backend.generate(
            GenerationRequest(
                prompt="Say hello",
                max_tokens=16,
                temperature=0.0,
                messages=[{"role": "user", "content": "Say hello"}],
                stream=True,
            )
        )
    ]

    assert "".join(chunk.text for chunk in chunks) == "Hello world!"
    assert chunks[-1].finished is True
    args = captured["args"]
    assert isinstance(args, tuple)
    assert str(executable.resolve()) == args[0]
    assert "--model" in args
    assert "--prompt" in args
    assert "--gpu-layers" in args
    assert args[args.index("--gpu-layers") + 1] == "0"
