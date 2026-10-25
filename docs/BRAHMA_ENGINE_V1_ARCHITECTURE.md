# Brahma Engine V1 — Architecture and Migration Plan

## Decision

Brahma Desk will own a first-class inference engine boundary integrated with the installer. The installer must not directly manage an inference process, and the UI must not need to know which runtime implementation is active.

This is a staged replacement, not a claim that a new transformer/GPU implementation already exists. The first milestone establishes the engine contract and moves lifecycle ownership behind it. A native inference backend can then replace the compatibility adapter without another installer redesign.

## Goals

- One engine API for runtime discovery, model validation, configuration, start/stop, health, and generation.
- Installer model records remain the source of truth for model path, verification metadata, and selected runtime settings.
- Explicit process ownership: never kill or reconfigure a server the user started independently.
- Actionable errors with stage, stable error code, user-facing message, and diagnostic details safe to log.
- CPU-first fallback; GPU acceleration is reported only when the selected backend actually supports it.
- Reproducible tests with fake runtimes, no network or model download required.
- A future native backend can be selected behind the same contract.

## Target architecture

```text
Electron UI
   |
   v
FastAPI installer + chat API
   |
   v
Brahma Engine Manager
   |-- Model Registry (GGUF path, hash, provenance, config)
   |-- Runtime Resolver (installed engine builds / capabilities)
   |-- Resource Planner (RAM, VRAM, context, offload estimate)
   |-- Lifecycle Supervisor (start, health, stop, ownership)
   |-- Generation API (streaming, cancellation, diagnostics)
   |
   +-- Backend adapter (migration compatibility only)
   +-- Native Brahma runtime (future implementation)
```

## Engine API responsibilities

1. `inspect_model(path)`: validate file existence, GGUF header, size, and metadata where available.
2. `plan(model, hardware, settings)`: return a resource plan and explicit warnings; do not overstate GPU compatibility.
3. `configure(model_id, settings)`: persist a validated configuration atomically.
4. `start(model_id)`: start only a managed runtime; return structured startup diagnostics.
5. `health()`: distinguish engine process health, model loaded state, and generation readiness.
6. `generate(request)`: stream tokens, surface cancellation and runtime failures.
7. `stop()`: stop only processes owned by Brahma Desk.

## Error contract

Errors use stable codes such as `MODEL_NOT_FOUND`, `MODEL_INVALID_GGUF`, `MODEL_CHECKSUM_MISMATCH`, `RUNTIME_NOT_INSTALLED`, `RUNTIME_START_FAILED`, `RESOURCE_BUDGET_EXCEEDED`, `ENGINE_NOT_READY`, and `GENERATION_FAILED`. HTTP handlers translate these to appropriate status codes and return a correlation ID. Raw tracebacks belong in local logs, not in the user-facing UI.

## Rollout stages

- **E1 — Contract and diagnostics:** engine types, error contract, fake-engine tests, installer-to-engine boundary.
- **E2 — Lifecycle ownership:** one supervisor owns start/stop/health and avoids external-port collisions.
- **E3 — Installer integration:** validate → plan → configure → start → smoke-generate as a single guided flow, with stage-specific results.
- **E4 — Native runtime prototype:** select one architecture (initially a dense decoder-only GGUF model), implement tokenizer/model loading and CPU inference, and benchmark against a fixed test suite.
- **E5 — Acceleration and compatibility:** add GPU backend only after numerical correctness and performance tests; keep runtime selection explicit.
- **E6 — Cutover:** remove compatibility adapter only after native runtime passes correctness, stability, packaging, and performance gates.

## Non-goals for the first milestone

- Claiming a from-scratch inference engine based solely on a new API wrapper.
- Supporting every GGUF architecture, quantization format, sampler, or multimodal model immediately.
- Replacing the existing chat storage, profiles, or UI.
- Silently killing external llama.cpp processes or silently switching to CPU after a failed GPU request.

## Acceptance gates

- Installer never returns “configured and ready” without a real generation smoke test.
- A failed stage returns a stable error code and actionable message.
- Existing external inference processes remain untouched.
- Unit tests use deterministic fake runtimes; integration tests exercise a real runtime separately.
- A native backend must pass tokenizer/model correctness checks, prompt/completion smoke tests, cancellation, repeated start/stop, and packaged Windows tests before becoming default.
