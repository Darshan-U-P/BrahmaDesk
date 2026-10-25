"""Native Brahma Engine CLI adapter.

The current native executable is a one-shot CLI, not a persistent server. This
adapter validates the runtime/model at startup and launches one owned process
per generation, forwarding its stdout as streamed text chunks.
"""
from __future__ import annotations

import asyncio
import codecs
import hashlib
import os
import subprocess
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
    ModelArtifact,
    ResourcePlan,
)


class BrahmaNativeBackend:
    """InferenceBackend backed by engine/build-llama/.../brahma-engine.exe."""

    def __init__(
        self,
        executable: Path,
        *,
        generation_timeout: float = 600.0,
        batch_size: int = 512,
    ) -> None:
        self._executable = Path(executable).expanduser().resolve()
        self._generation_timeout = generation_timeout
        self._batch_size = batch_size
        self._model: ModelArtifact | None = None
        self._settings: EngineSettings | None = None
        self._last_message: str | None = None

    @property
    def name(self) -> str:
        return "brahma-native-cli"

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
                "The selected model file could not be read.",
                "inspect_model",
                {"reason": str(exc)},
            ) from exc
        if magic != b"GGUF":
            raise EngineError(
                EngineErrorCode.MODEL_INVALID_GGUF,
                "The selected model file does not contain the GGUF magic header.",
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
                "Brahma Desk could not find brahma-engine.exe.",
                "plan",
                {"executable": str(self._executable)},
            )
        size_gb = model.path.stat().st_size / (1024 ** 3)
        gpu_layers = settings.gpu_layers or 0
        selected_mode = "cpu" if gpu_layers == 0 else "hybrid"
        warnings = [
            "The native CLI loads and unloads the model for each generation.",
            "Memory estimates are heuristics, not a hardware compatibility guarantee.",
        ]
        return ResourcePlan(
            selected_mode=selected_mode,
            context_size=settings.context_size,
            estimated_ram_gb=round(size_gb + 2.0, 2),
            estimated_vram_gb=round(size_gb + 1.0, 2) if gpu_layers else 0.0,
            warnings=tuple(warnings),
        )

    async def start(
        self, model: ModelArtifact, settings: EngineSettings
    ) -> EngineHealth:
        if not self._executable.is_file():
            raise EngineError(
                EngineErrorCode.RUNTIME_NOT_INSTALLED,
                "The configured brahma-engine.exe does not exist.",
                "start",
                {"executable": str(self._executable)},
            )
        if not model.path.is_file():
            raise EngineError(
                EngineErrorCode.MODEL_NOT_FOUND,
                "The selected GGUF model file no longer exists.",
                "start",
                {"path": str(model.path)},
            )
        self._model = model
        self._settings = settings
        self._last_message = None
        # This CLI has no persistent model server. Readiness means the executable
        # and validated model are available to launch for the next request.
        return await self.health()

    async def health(self) -> EngineHealth:
        ready = (
            self._model is not None
            and self._settings is not None
            and self._executable.is_file()
            and self._model.path.is_file()
        )
        gpu_layers = (
            self._settings.gpu_layers
            if self._settings and self._settings.gpu_layers is not None
            else 0
        )
        return EngineHealth(
            state=EngineState.READY if ready else EngineState.STOPPED,
            backend=self.name,
            model_id=self._model.model_id if self._model else None,
            process_owned=False,
            generation_ready=ready,
            message=self._last_message,
            diagnostics={
                "executable": str(self._executable),
                "model_path": str(self._model.path) if self._model else None,
                "execution_mode": "one-process-per-generation",
                "gpu_layers": gpu_layers,
            },
        )

    def _format_prompt(self, request: GenerationRequest) -> str:
        messages = request.messages or [{"role": "user", "content": request.prompt}]
        # The current CLI tokenizes a raw prompt rather than applying the GGUF
        # chat template itself. Use Qwen's native template for the default Qwen
        # models; retain an explicit transcript for other architectures.
        if self._model and "qwen" in self._model.model_id.casefold():
            parts: list[str] = []
            for message in messages:
                role = str(message.get("role", "user")).strip().lower()
                if role not in {"system", "user", "assistant"}:
                    role = "user"
                content = str(message.get("content", "")).strip()
                if content:
                    parts.append(f"<|im_start|>{role}\\n{content}<|im_end|>")
            parts.append("<|im_start|>assistant\\n")
            return "\\n".join(parts)

        parts = []
        for message in messages:
            role = str(message.get("role", "user")).strip().upper()
            content = str(message.get("content", "")).strip()
            if content:
                parts.append(f"{role}:\\n{content}")
        parts.append("ASSISTANT:")
        return "\\n\\n".join(parts)

    async def generate(
        self, request: GenerationRequest
    ) -> AsyncIterator[GenerationChunk]:
        if self._model is None or self._settings is None:
            raise EngineError(
                EngineErrorCode.ENGINE_NOT_READY,
                "Brahma Engine has not been configured with a model.",
                "generate",
            )
        if request.max_tokens > 4096:
            raise EngineError(
                EngineErrorCode.CONFIGURATION_INVALID,
                "The current native CLI supports at most 4096 generated tokens per request.",
                "generate",
                {"requested_max_tokens": request.max_tokens, "maximum": 4096},
            )

        options = request.options or {}
        settings = self._settings
        gpu_layers = settings.gpu_layers if settings.gpu_layers is not None else 0
        threads = settings.threads or max(1, (os.cpu_count() or 4) - 1)
        prompt = self._format_prompt(request)
        command = [
            str(self._executable),
            "--model", str(self._model.path),
            "--prompt", prompt,
            "--max-tokens", str(request.max_tokens),
            "--ctx-size", str(settings.context_size),
            "--batch-size", str(self._batch_size),
            "--threads", str(threads),
            "--gpu-layers", str(gpu_layers),
            "--temperature", str(request.temperature),
            "--top-k", str(max(0, int(options.get("top_k", 40)))),
            "--top-p", str(float(options.get("top_p", 0.95))),
        ]

        creation_kwargs: dict[str, object] = {}
        if os.name == "nt":
            creation_kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)

        process: asyncio.subprocess.Process | None = None
        stderr_task: asyncio.Task[bytes] | None = None
        decoder = codecs.getincrementaldecoder("utf-8")("replace")
        header_buffer = bytearray()
        header_removed = False
        answer_seen = False

        try:
            process = await asyncio.create_subprocess_exec(
                *command,
                cwd=str(self._executable.parent),
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                **creation_kwargs,
            )
            assert process.stdout is not None and process.stderr is not None
            stderr_task = asyncio.create_task(process.stderr.read())

            while True:
                try:
                    chunk = await asyncio.wait_for(
                        process.stdout.read(4096),
                        timeout=self._generation_timeout,
                    )
                except asyncio.TimeoutError as exc:
                    process.kill()
                    await process.wait()
                    raise EngineError(
                        EngineErrorCode.GENERATION_FAILED,
                        "Brahma Engine exceeded the generation timeout.",
                        "generate",
                        {"timeout_seconds": self._generation_timeout},
                    ) from exc

                if not chunk:
                    break

                if not header_removed:
                    header_buffer.extend(chunk)
                    separator = header_buffer.find(b"\n\n")
                    if separator < 0:
                        # Guard against unexpected output that never contains the
                        # CLI's banner separator.
                        if len(header_buffer) > 8192:
                            raise EngineError(
                                EngineErrorCode.GENERATION_FAILED,
                                "Brahma Engine returned an unexpected output header.",
                                "generate",
                            )
                        continue
                    del header_buffer[:separator + 2]
                    header_removed = True
                    chunk = bytes(header_buffer)
                    header_buffer.clear()

                if chunk:
                    text = decoder.decode(chunk)
                    if text:
                        answer_seen = True
                        yield GenerationChunk(text=text)

            tail = decoder.decode(b"", final=True)
            if tail:
                answer_seen = True
                yield GenerationChunk(text=tail)

            return_code = await process.wait()
            stderr_bytes = await stderr_task if stderr_task else b""
            stderr_text = stderr_bytes.decode("utf-8", errors="replace").strip()
            if return_code != 0:
                self._last_message = stderr_text or f"brahma-engine.exe exited with code {return_code}."
                raise EngineError(
                    EngineErrorCode.GENERATION_FAILED,
                    "Brahma Engine failed to generate a response.",
                    "generate",
                    {"return_code": return_code, "reason": self._last_message},
                )
            if not answer_seen:
                raise EngineError(
                    EngineErrorCode.GENERATION_FAILED,
                    "Brahma Engine returned no response text.",
                    "generate",
                    {"stderr": stderr_text[:2000]},
                )

            yield GenerationChunk(text="", finished=True, finish_reason="stop")

        except EngineError:
            raise
        except (OSError, asyncio.SubprocessError) as exc:
            self._last_message = str(exc)
            raise EngineError(
                EngineErrorCode.GENERATION_FAILED,
                "Could not execute the native Brahma Engine.",
                "generate",
                {"reason": str(exc), "executable": str(self._executable)},
            ) from exc
        finally:
            if process is not None and process.returncode is None:
                process.kill()
                await process.wait()
            if stderr_task is not None and not stderr_task.done():
                stderr_task.cancel()
                try:
                    await stderr_task
                except asyncio.CancelledError:
                    pass

    async def stop(self) -> None:
        self._model = None
        self._settings = None
        self._last_message = None
