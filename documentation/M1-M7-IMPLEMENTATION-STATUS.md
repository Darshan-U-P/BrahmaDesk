# Brahma Desk Smart Installer — Implementation Status

Branch: \`docs/m0-repository-audit-m1-hardware-design\`

This document tracks implementation work for M1–M7. A code change is not considered release-complete until the latest Windows CI run and local/manual acceptance checks pass.

## Implemented in code

| Milestone | Implementation |
|---|---|
| M1 — Hardware detection | \`GET /api/installer/hardware\` reports OS, CPU, RAM, GPU adapters, model-directory free space, and detection limitations. Windows GPU VRAM is explicitly marked best-effort. |
| M2 — Model catalog | \`GET /api/installer/catalog\` provides a curated GGUF catalog and hardware-aware recommendations. Hugging Face metadata is resolved to an immutable commit SHA before a download. |
| M3 — Setup wizard | React Model Setup page is integrated into workspace navigation. It shows hardware, model recommendations, download state, configuration controls, and runtime selection. |
| M4 — Download manager | Async downloads use a temporary \`.part\` file, HTTP Range resume, disk-space checks, SHA-256 verification when the source exposes it, persisted job state, cancellation, retry, and progress polling. |
| M5 — Validation and registry | GGUF magic and size checks, source checksum comparison when available, SQLite model registry, and a local-GGUF scan for files in the managed model directory. Local-file origin and license are not asserted as trusted. |
| M6 — Adaptive inference | Saves context, CPU/hybrid/GPU placement, thread count and GPU-layer settings. Activation only stops a llama.cpp process owned by this backend, starts the selected model, and runs a short generation smoke test. Externally owned llama.cpp processes are left untouched. |
| M7 — Installer/QA integration | Added Windows CI for backend tests, TypeScript/Vite build, PyInstaller backend packaging, and a full NSIS installer artifact. |

## Backend endpoints

- \`GET /api/installer/status\`
- \`GET /api/installer/hardware\`
- \`GET /api/installer/catalog\`
- \`GET /api/installer/models\`
- \`POST /api/installer/scan-local\`
- \`POST /api/installer/downloads\`
- \`GET /api/installer/downloads\`
- \`GET /api/installer/downloads/{job_id}\`
- \`POST /api/installer/downloads/{job_id}/cancel\`
- \`POST /api/installer/downloads/{job_id}/retry\`
- \`POST /api/installer/validate/{model_id}\`
- \`POST /api/installer/configure\`
- \`GET /api/installer/configuration\`
- \`GET /api/installer/runtime\` and \`POST /api/installer/runtime\`
- \`POST /api/installer/activate/{model_id}\`

## Important boundaries

1. **llama.cpp is not bundled by these changes.** Users select an existing \`llama-server.exe\` using the native file picker. The selected path is stored in the existing per-user SQLite database and restored on backend startup.
2. GPU VRAM reported by Windows may be inaccurate. GPU compatibility depends on the actual llama.cpp build and driver; the activation smoke test is the final basic runtime check.
3. A local GGUF file can be structurally checked and hashed, but that does not prove its source, license, or authenticity.
4. The curated catalog must be kept current. Downloads fail closed if the exact expected filename is absent at the resolved Hugging Face revision.
5. The setup wizard lives inside Brahma Desk after the NSIS application install; it is not a custom NSIS page that runs before the application is installed.
6. Full release acceptance still requires current CI to pass, installing the generated NSIS artifact on a clean Windows machine, testing CPU-only and GPU/hybrid paths, and verifying that existing profiles/conversations survive an upgrade.

## Release gate

Do not merge or publish a release while any of the following remain unverified:
- Latest Windows CI for the final commit is green.
- The NSIS installer artifact is produced successfully.
- A clean-install smoke test passes.
- An upgrade preserves existing per-user data.
- A failed/interrupted download resumes safely.
- An incompatible model/runtime fails with an actionable error rather than replacing an externally managed server.
