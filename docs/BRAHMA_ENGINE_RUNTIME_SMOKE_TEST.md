# Brahma Engine v1 — real-runtime smoke test

The engine contract and adapter currently have unit-test coverage, but they have not yet been verified against the installed llama.cpp binary. Run this test before connecting the supervisor to the production installer lifecycle.

## Safety boundary

The test must use a separate loopback port and must start a process owned by `LlamaCppServerBackend`. Do not point it at or terminate an already-running server. The adapter should only stop the exact process it created.

## Validation checklist

- Verify the configured `llama-server.exe` exists.
- Verify the model path is a readable GGUF file.
- Start the isolated backend with a conservative context size (2048 tokens).
- Generate a short deterministic response.
- Confirm a non-empty response and successful cleanup.
- Confirm the existing Brahma Desk backend and any pre-existing llama.cpp server remain unaffected.

## Acceptance criteria

1. Startup reports `ready` and `generation_ready=true`.
2. A short generation request returns non-empty text.
3. The test reports the selected model, backend, and resource-plan warnings.
4. The test shuts down only its own child process, including after a failed generation.
5. Existing backend tests still pass.

Do not switch installer activation to `EngineSupervisor` until these criteria pass on the target Windows machine. The current RAM/VRAM estimates are heuristics, not a hardware compatibility guarantee.