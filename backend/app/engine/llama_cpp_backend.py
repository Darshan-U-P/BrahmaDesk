"""Owned-process llama.cpp adapter for the Brahma Engine contract.

This adapter is intentionally opt-in: it owns only the Popen process it creates
and binds it to a newly allocated loopback port. It never attaches to or kills
an unrelated llama-server process.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
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
    GenerationToolCall,
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

    @staticmethod
    def _parse_tool_calls(raw_calls: object) -> tuple[GenerationToolCall, ...]:
        """Validate provider tool-call envelopes before exposing them to the agent layer."""
        if raw_calls is None:
            return ()
        if not isinstance(raw_calls, list) or len(raw_calls) > 32:
            raise ValueError("The model returned an invalid or oversized tool-call collection")
        parsed: list[GenerationToolCall] = []
        for raw_call in raw_calls:
            if not isinstance(raw_call, dict):
                raise ValueError("The model returned an invalid tool call")
            function = raw_call.get("function")
            if not isinstance(function, dict):
                raise ValueError("A tool call is missing its function payload")
            name = function.get("name")
            call_id = raw_call.get("id")
            raw_arguments = function.get("arguments", "{}")
            if not isinstance(name, str) or not name.strip():
                raise ValueError("A tool call is missing its function name")
            if not isinstance(call_id, str) or not call_id.strip():
                raise ValueError("A tool call is missing its call ID")
            if isinstance(raw_arguments, str):
                try:
                    arguments = json.loads(raw_arguments)
                except json.JSONDecodeError as exc:
                    raise ValueError("A tool call contains malformed JSON arguments") from exc
            else:
                arguments = raw_arguments
            if not isinstance(arguments, dict):
                raise ValueError("Tool-call arguments must be a JSON object")
            parsed.append(GenerationToolCall(
                id=call_id,
                name=name,
                arguments=arguments,
            ))
        return tuple(parsed)

    async def generate(
        self, request: GenerationRequest
    ) -> AsyncIterator[GenerationChunk]:
        if not await self._endpoint_ready() or not self._base_url or not self._model:
            raise EngineError(
                EngineErrorCode.ENGINE_NOT_READY,
                "The owned llama.cpp process is not ready for generation.",
                "generate",
            )

        payload = {
            **request.options,
            "model": self._model.model_id,
            "messages": request.messages or [
                {"role": "user", "content": request.prompt}
            ],
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "stream": request.stream,
        }
        if request.tools:
            payload["tools"] = request.tools
            payload["tool_choice"] = "auto"

        try:
            async with httpx.AsyncClient(timeout=self._request_timeout) as client:
                if not request.stream:
                    response = await client.post(
                        f"{self._base_url}/v1/chat/completions",
                        json=payload,
                    )
                    response.raise_for_status()
                    data = response.json()
                    choices = data.get("choices") or []
                    message = choices[0].get("message", {}) if choices else {}
                    answer = message.get("content") or ""
                    if not isinstance(answer, str):
                        raise ValueError("The model returned an invalid text response")
                    tool_calls = self._parse_tool_calls(message.get("tool_calls"))
                    if not answer.strip() and not tool_calls:
                        raise ValueError("The model returned neither text nor tool calls")

                    yield GenerationChunk(
                        text=answer,
                        finished=True,
                        finish_reason=(
                            choices[0].get("finish_reason") or "stop"
                            if choices else "stop"
                        ),
                        tool_calls=tool_calls,
                    )
                    return

                answer_seen = False
                tool_call_fragments: dict[int, dict[str, str]] = {}
                stream_complete = False
                finish_reason: str | None = None
                async with client.stream(
                    "POST",
                    f"{self._base_url}/v1/chat/completions",
                    json=payload,
                    headers={"Accept": "text/event-stream"},
                ) as response:
                    response.raise_for_status()

                    async for line in response.aiter_lines():
                        if not line.startswith("data:"):
                            continue
                        data = line[5:].strip()
                        if not data:
                            continue
                        if data == "[DONE]":
                            stream_complete = True
                            break

                        event = json.loads(data)
                        choices = event.get("choices") or []
                        if not choices:
                            continue

                        choice = choices[0]
                        delta = choice.get("delta") or {}
                        content = delta.get("content", "")
                        if content:
                            if not isinstance(content, str):
                                raise ValueError(
                                    "The model returned an invalid text chunk"
                                )
                            answer_seen = True
                            yield GenerationChunk(text=content)

                        for raw_call in delta.get("tool_calls") or []:
                            if not isinstance(raw_call, dict):
                                raise ValueError("The model returned an invalid tool-call delta")
                            index = raw_call.get("index")
                            if not isinstance(index, int) or index < 0 or index > 31:
                                raise ValueError("The model returned an invalid tool-call index")
                            fragment = tool_call_fragments.setdefault(
                                index, {"id": "", "name": "", "arguments": ""}
                            )
                            if raw_call.get("id"):
                                fragment["id"] = str(raw_call["id"])
                            function = raw_call.get("function") or {}
                            if not isinstance(function, dict):
                                raise ValueError("The model returned an invalid function delta")
                            if function.get("name"):
                                fragment["name"] += str(function["name"])
                            if function.get("arguments"):
                                arguments_part = function["arguments"]
                                if not isinstance(arguments_part, str):
                                    raise ValueError("Tool-call argument deltas must be strings")
                                fragment["arguments"] += arguments_part

                        if choice.get("finish_reason") is not None:
                            finish_reason = str(choice["finish_reason"])

                if not stream_complete:
                    raise ValueError("The model response stream ended before its completion marker")
                assembled_calls = [
                    {
                        "id": item["id"],
                        "function": {
                            "name": item["name"],
                            "arguments": item["arguments"] or "{}",
                        },
                    }
                    for _, item in sorted(tool_call_fragments.items())
                ]
                tool_calls = self._parse_tool_calls(assembled_calls)
                if not answer_seen and not tool_calls:
                    raise ValueError("The streaming response contained neither text nor tool calls")
                yield GenerationChunk(
                    text="",
                    finished=True,
                    finish_reason=finish_reason or "stop",
                    tool_calls=tool_calls,
                )

        except httpx.HTTPError as exc:
            raise EngineError(
                EngineErrorCode.GENERATION_FAILED,
                "The owned llama.cpp process failed to generate text.",
                "generate",
                {"reason": str(exc)},
            ) from exc
        except (ValueError, KeyError, TypeError) as exc:
            raise EngineError(
                EngineErrorCode.GENERATION_FAILED,
                "The owned llama.cpp process returned an invalid response.",
                "generate",
                {"reason": str(exc)},
            ) from exc

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
