"""Opt-in runtime probe for Brahma Engine v1; not wired into production."""
from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from .contracts import EngineError, EngineSettings, EngineSupervisor, GenerationRequest
from .llama_cpp_backend import LlamaCppServerBackend


async def probe_runtime(
    executable: Path,
    model_path: Path,
    *,
    mode: str = "cpu",
    context_size: int = 2048,
    startup_timeout: float = 180.0,
) -> dict[str, Any]:
    """Run one short generation and clean up only the backend-owned process."""
    backend = LlamaCppServerBackend(executable, startup_timeout=startup_timeout)
    supervisor = EngineSupervisor(backend)
    result: dict[str, Any] = {"status": "failed", "mode": mode}
    try:
        health = await supervisor.start(
            model_path, EngineSettings(context_size=context_size, mode=mode)
        )
        chunks = [chunk async for chunk in supervisor.generate(
            GenerationRequest(
                prompt="Reply with exactly: BRAHMA_ENGINE_OK",
                max_tokens=48,
                temperature=0.0,
            )
        )]
        output = "".join(chunk.text for chunk in chunks).strip()
        if not output:
            raise RuntimeError("Runtime returned an empty response")
        plan = supervisor.resource_plan
        result.update({
            "status": "passed",
            "backend": health.backend,
            "model_id": health.model_id,
            "process_owned": health.process_owned,
            "generation_test": "passed",
            "sample_output": output[:500],
            "resource_plan": {
                "mode": plan.selected_mode if plan else mode,
                "context_size": plan.context_size if plan else context_size,
                "estimated_ram_gb": plan.estimated_ram_gb if plan else None,
                "estimated_vram_gb": plan.estimated_vram_gb if plan else None,
                "warnings": list(plan.warnings) if plan else [],
            },
        })
    except EngineError as exc:
        result["error"] = exc.as_response()["error"]
    except Exception as exc:
        result["error"] = {"message": str(exc)}
    finally:
        try:
            await supervisor.stop()
        except Exception as exc:
            result["status"] = "failed"
            result["cleanup_error"] = str(exc)
    return result


def run_probe(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Synchronous wrapper for scripts and local diagnostics."""
    return asyncio.run(probe_runtime(*args, **kwargs))
