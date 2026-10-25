"""Public contracts for the Brahma Engine.

This module deliberately contains no runtime implementation or process side
effects. Engine implementations and installer services depend on these stable
types rather than depending directly on a particular inference server.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any, AsyncIterator, Protocol


class EngineErrorCode(StrEnum):
    MODEL_NOT_FOUND = "MODEL_NOT_FOUND"
    MODEL_INVALID_GGUF = "MODEL_INVALID_GGUF"
    MODEL_CHECKSUM_MISMATCH = "MODEL_CHECKSUM_MISMATCH"
    RUNTIME_NOT_INSTALLED = "RUNTIME_NOT_INSTALLED"
    RUNTIME_START_FAILED = "RUNTIME_START_FAILED"
    RESOURCE_BUDGET_EXCEEDED = "RESOURCE_BUDGET_EXCEEDED"
    ENGINE_NOT_READY = "ENGINE_NOT_READY"
    GENERATION_FAILED = "GENERATION_FAILED"
    CONFIGURATION_INVALID = "CONFIGURATION_INVALID"


@dataclass(slots=True)
class EngineError(Exception):
    """An expected, user-actionable engine failure with a stable machine code."""

    code: EngineErrorCode
    message: str
    stage: str
    details: dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        return f"{self.code.value} during {self.stage}: {self.message}"

    def as_response(self, correlation_id: str | None = None) -> dict[str, Any]:
        """Return a safe API payload; callers should log internal tracebacks separately."""
        result: dict[str, Any] = {
            "code": self.code.value,
            "message": self.message,
            "stage": self.stage,
        }
        if correlation_id:
            result["correlation_id"] = correlation_id
        if self.details:
            result["details"] = self.details
        return {"error": result}


class EngineState(StrEnum):
    STOPPED = "stopped"
    STARTING = "starting"
    READY = "ready"
    GENERATING = "generating"
    STOPPING = "stopping"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class ModelArtifact:
    model_id: str
    path: Path
    sha256: str | None = None
    verified: bool = False
    architecture: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class EngineSettings:
    context_size: int = 4096
    mode: str = "auto"
    threads: int | None = None
    gpu_layers: int | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.context_size < 512 or self.context_size > 131072:
            raise ValueError("context_size must be between 512 and 131072")
        if self.mode not in {"auto", "cpu", "hybrid", "gpu"}:
            raise ValueError("mode must be auto, cpu, hybrid, or gpu")
        if self.threads is not None and self.threads < 1:
            raise ValueError("threads must be positive")
        if self.gpu_layers is not None and self.gpu_layers < 0:
            raise ValueError("gpu_layers cannot be negative")


@dataclass(frozen=True, slots=True)
class ResourcePlan:
    selected_mode: str
    context_size: int
    estimated_ram_gb: float | None = None
    estimated_vram_gb: float | None = None
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class EngineHealth:
    state: EngineState
    backend: str
    model_id: str | None = None
    process_owned: bool = False
    generation_ready: bool = False
    message: str | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class GenerationRequest:
    prompt: str
    max_tokens: int = 256
    temperature: float = 0.7
    options: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.prompt.strip():
            raise ValueError("prompt cannot be empty")
        if not 1 <= self.max_tokens <= 131072:
            raise ValueError("max_tokens must be between 1 and 131072")
        if not 0.0 <= self.temperature <= 2.0:
            raise ValueError("temperature must be between 0 and 2")


@dataclass(frozen=True, slots=True)
class GenerationChunk:
    text: str
    finished: bool = False
    finish_reason: str | None = None


class InferenceBackend(Protocol):
    """Contract every Brahma Engine inference backend must implement."""

    @property
    def name(self) -> str: ...

    async def inspect_model(self, path: Path) -> ModelArtifact: ...

    async def plan(
        self, model: ModelArtifact, settings: EngineSettings
    ) -> ResourcePlan: ...

    async def start(
        self, model: ModelArtifact, settings: EngineSettings
    ) -> EngineHealth: ...

    async def health(self) -> EngineHealth: ...

    async def generate(
        self, request: GenerationRequest
    ) -> AsyncIterator[GenerationChunk]: ...

    async def stop(self) -> None: ...
