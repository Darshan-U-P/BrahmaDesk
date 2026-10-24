# Brahma Desk Smart Installer — M2–M7 Implementation Specification

**Baseline:** Brahma Desk v0.2.0, Electron + React/Vite + FastAPI + SQLite + llama.cpp  
**Status:** Proposed specification; implementation has not started.  
**Prerequisite:** M0 audit and M1 hardware report contract must be reviewed and M1 must be implemented/tested before model recommendations can be trusted.

## 1. Product goal

Deliver a Windows-first setup flow that detects the machine, recommends a defensible set of approved local GGUF models, downloads and validates a chosen model, configures llama.cpp conservatively, and launches a verified chat session.

The installer must be resumable and recoverable. It must not mark a model ready merely because a download completed, and it must not overwrite existing user data or model files during upgrades.

## 2. Existing architecture and reuse constraints

- Electron main process owns the desktop lifecycle and privileged operations.
- React renderer owns user-facing setup screens.
- FastAPI backend is the local API and should own catalog evaluation, download jobs, model validation/registry and inference configuration.
- SQLite under the existing per-user data directory is the persistence layer; use versioned migrations.
- llama.cpp remains the inference runtime; reuse its existing process ownership, health checks and graceful shutdown behavior.
- Existing NSIS installer and electron-updater flows must remain separate from model downloads. Application updates and multi-gigabyte model acquisition have different lifecycle and recovery requirements.
- Do not bundle large GGUF files inside the application installer.
- Do not expose arbitrary filesystem or shell operations to the renderer. Use validated API requests and narrow Electron IPC only where necessary.

## 3. M2 — Model catalog and compatibility engine

### 3.1 Catalog source of truth

Store a curated, version-controlled catalog manifest in the repository initially, for example `backend/app/model_catalog/catalog.json`. The manifest is data, not executable code. Only maintainers can add or approve entries through code review.

Each model artifact entry should contain:
- Stable internal model ID and display name.
- Hugging Face repository ID and exact repository revision commit SHA.
- Exact GGUF filename and immutable download URL derived from the approved repository/revision.
- Format and quantization label (e.g. Q4_K_M) verified against the artifact metadata.
- Expected byte size and SHA-256 digest obtained from a trusted release process.
- License identifier, license URL and distribution/use notes.
- Architecture/family, parameter count where known, context capability and tokenizer/chat-template notes.
- Minimum supported llama.cpp/runtime version or required capabilities when known.
- Catalog schema version, review date and maintainer approval status.

Never trust a mutable branch name such as main as the installed revision. Do not use wildcard filenames, user-supplied URLs, or unreviewed repository IDs as catalog entries. A license is a release gate: omit artifacts whose redistribution/download terms have not been reviewed.

### 3.2 Compatibility inputs

Input is the normalized M1 hardware report plus the intended runtime/build capabilities and model metadata. Unknown fields remain unknown; the engine must not silently substitute optimistic defaults.

### 3.3 Memory estimate

Use an explicit estimate breakdown:
- Model artifact size is a useful proxy for quantized weight residency but is not the same as peak runtime memory.
- KV cache estimate depends on context length, layer count, KV head count, head dimension, K/V cache types and runtime implementation. Use model metadata where available.
- Add configurable allowances for compute buffers, graph/workspace allocations, allocator overhead and OS/application headroom.
- Estimate CPU RAM and GPU VRAM separately; account for partial GPU layer offload where the runtime supports it.
- When required metadata is missing, return a conservative range or “insufficient metadata”, not a fabricated exact number.
- Include a confidence label and a visible explanation of assumptions.

A first-pass rule may use a conservative safety margin, but thresholds must be calibrated using load tests on representative Windows hardware before they are described as guaranteed.

### 3.4 Decision classes and ranking

Return one of:
- `recommended`: expected to fit within measured resources and validated runtime constraints with configured headroom.
- `compatible_with_constraints`: likely usable only with explicit limitations, such as reduced context, CPU-only mode, partial offload or slower performance.
- `unsupported`: a known hard constraint is violated.
- `unknown`: evidence is insufficient for a safe claim.

Rank candidates using memory fit and headroom, runtime/backend support, context requirements, quality/use-case tags, latency expectations and download size. Parameter count alone must never determine rank. Include a structured list of reasons and warnings for every result.

### 3.5 M2 acceptance tests

- Manifest schema and required fields are validated.
- Missing or invalid checksum, revision, license or file size rejects an entry from the approved catalog.
- Memory estimates respond predictably to context and KV-cache changes.
- Unknown hardware fields cannot produce a high-confidence recommendation.
- Same hardware/catalog inputs yield deterministic ordering.
- Unit tests cover CPU-only, integrated GPU, discrete GPU, low RAM, low disk, unknown VRAM and conflicting runtime capability cases.
- Recommendations include explanations a user can understand.

## 4. M3 — Setup wizard

### 4.1 State model

Use an explicit state machine instead of independent booleans. Suggested states:
- `welcome`
- `scanning`
- `hardware_results`
- `loading_catalog`
- `model_selection`
- `confirm_download`
- `downloading`
- `validating`
- `configuring`
- `testing_model`
- `complete`
- `cancelled`
- `recoverable_error`

Transitions must be explicit and testable. A cancelled or failed state must preserve enough job/selection information to retry safely. The user can go back before a download begins; while a download is active, cancellation must go through the download manager.

### 4.2 UI requirements

- Show hardware scan progress and partial-data warnings.
- Show recommended model first, plus alternatives and why they differ.
- Display quantization, expected download size, estimated RAM/VRAM, context assumptions, license, source/revision and compatibility class.
- Require explicit confirmation before downloading a large artifact; display disk-space requirement.
- Show download progress and recovery choices.
- Never say “ready” until validation, registration and startup test succeed.
- Provide accessible labels, keyboard navigation, focus management and clear empty/error states.
- Avoid auto-launching downloads or changing the active model without user confirmation.

### 4.3 Integration

Add the wizard as a renderer flow using the existing application shell and API. Keep privileged filesystem and process actions in backend/main process. Existing users with a configured working model should not be forced through first-run setup on every application launch.

### 4.4 M3 acceptance tests

Test each state transition, back/cancel behavior, no catalog results, hardware detection partial failure, download failure, app restart during a job, and returning users with an existing valid model.

## 5. M4 — Hugging Face download manager

### 5.1 Approved source and revision

Downloads must be initiated by a catalog model ID/artifact ID, not an arbitrary URL. Resolve to the catalog's exact repository revision and filename. Use HTTPS and validate redirects/host policy. If Hugging Face authentication is required for an approved artifact, handle tokens securely and never log them; do not persist credentials in plain-text application settings.

### 5.2 Persistent jobs

Add a versioned SQLite migration for download jobs. Suggested fields:
- job ID, model/artifact ID, repository/revision, filename
- expected size and SHA-256
- destination and temporary partial path
- state (queued/downloading/paused/cancelled/failed/completed/validating)
- bytes received, total bytes, last error code/message
- created/updated timestamps and retry count

Use atomic updates and a single-writer policy per artifact/destination. On startup, reconcile jobs with temporary files rather than assuming that a persisted “downloading” state means a transfer is still active.

### 5.3 Resume, cancellation and recovery

- Use HTTP Range resume only when the server returns a valid range response and the partial file belongs to the same exact artifact/revision.
- Validate response status, Content-Range, expected length and ETag where available.
- If resume is unsupported or the remote artifact changed, safely restart from byte zero.
- Stream to a temporary file in the target model directory; never write directly to the final registered filename.
- Use bounded retries with backoff for transient errors. Distinguish retryable network failures from permanent HTTP/auth/manifest errors.
- Cancellation stops network I/O, flushes/closes the file, and records a recoverable state.
- On restart, reconcile partial size and persisted byte count; never trust the persisted count alone.
- Estimate remaining time only when the sample is meaningful; otherwise show an indeterminate value.

### 5.4 Disk safety and progress

Check free space on the target volume before starting, with explicit headroom for temporary data and finalization. Report downloaded bytes, total bytes when known, rate, percentage and ETA when calculable. Protect against path traversal, unexpected filenames, symlink/reparse-point surprises and disk-full conditions.

### 5.5 M4 acceptance tests

Simulate dropped connections, invalid Range responses, server ignores Range, cancellation, disk full, app shutdown/restart, HTTP 404/401/429/5xx, wrong content length and corrupted partial files. No failed transfer may become a ready model.

## 6. M5 — Model validation and registry

### 6.1 Validation pipeline

A model can become `ready` only after:
1. The download is complete or a user-selected existing file has been chosen.
2. The file size matches catalog expectations when catalog-controlled.
3. SHA-256 matches the approved manifest for downloaded catalog artifacts.
4. GGUF magic/header and metadata are readable by a maintained parser or compatible llama.cpp metadata tool.
5. Essential metadata matches the expected artifact/model family where those expectations are available.
6. The model is atomically moved/finalized to its managed destination and registered.

A checksum proves bytes match the approved manifest, not that the model is safe or its license is acceptable. Local user-selected GGUFs without a trusted expected checksum should be marked `local_unverified`, not falsely labeled checksum-verified. Offer structural validation and an explicit user confirmation for local files.

### 6.2 Registry and schema

Add a versioned SQLite migration with a model registry and artifact metadata. Suggested model fields:
- stable model ID, origin (catalog/local), catalog ID/revision where applicable
- canonical local path, filename, file size, SHA-256 when known
- format, quantization, architecture and metadata summary
- validation status and validation timestamp
- registration/last-used timestamps
- active/default selection marker or separate settings reference

Enforce unique canonical paths where appropriate. Use transactional registration and filesystem-safe finalization. Never delete a user's pre-existing model during cleanup or upgrades.

### 6.3 Existing model discovery

Start with explicit user selection of a GGUF file or model directory. Validate candidates without recursively scanning the entire disk. Let the user configure a model directory and rediscover it on startup. Missing paths should transition to a visible unavailable state, not silently delete the registry record.

### 6.4 M5 acceptance tests

Cover correct/incorrect checksums, truncated GGUF, invalid magic/header, metadata mismatch, existing local model, missing file after restart, duplicate path, path changes, atomic registration and database migration from existing v0.2.0 user data.

## 7. M6 — Adaptive inference configuration

### 7.1 Configuration plan

Create a pure configuration planner that accepts the hardware report, selected model metadata, validated runtime capabilities and user preferences. It should return:
- execution mode (CPU, hybrid CPU/GPU, GPU-heavy)
- GPU layer offload target only when supported
- context length within model/runtime limits and memory budget
- KV-cache types only when supported by the selected runtime/model
- thread count and batch/ubatch defaults where relevant
- estimated memory by resource and safety margin
- rationale and any degraded-mode warning

Use conservative defaults first. Do not assume that all reported VRAM is available to the model, that unified memory is fully available, or that an installed CUDA driver means the packaged llama.cpp binary supports CUDA.

### 7.2 Startup integration

- Persist the chosen model ID and user-overridable configuration in the existing per-user settings store or a dedicated versioned table.
- Replace developer-machine fallback paths with resolved registry paths when a managed model is selected. Preserve explicit environment overrides for development where safe.
- Avoid breaking the existing behavior for users who already configure a local llama.cpp server.
- Track whether Brahma Desk owns the launched server; never kill an external server during shutdown.
- Apply a bounded startup timeout and report actionable errors.
- Run a minimal test prompt after the model endpoint reports ready. Separate endpoint readiness from successful generation.
- If loading fails, offer safe alternatives such as lower context, fewer GPU layers or CPU mode only when those options are supported and the user approves. Do not enter an endless retry loop.

### 7.3 M6 acceptance tests

Test CPU-only mode, GPU backend missing, insufficient VRAM, reduced context fallback, corrupt/missing model, server exits early, endpoint returns ready but generation fails, and preservation of external server ownership.

## 8. M7 — Installer integration, QA and release

### 8.1 Upgrade and data safety

- Preserve the current app data directory, SQLite database, active profile, model registry and model files across application upgrades.
- Keep large models outside the application install directory so an app upgrade does not redownload or overwrite them.
- Do not delete model data during normal uninstall unless the user explicitly selects a separately explained data-removal option.
- Back up or use safe migration procedures for schema changes; test upgrades from existing v0.2.0 data.
- Application updater and model downloader must remain independent state machines.

### 8.2 Test matrix

Include:
- Fresh install on supported Windows versions.
- Existing v0.2.0 upgrade with preserved conversations and profiles.
- Low-RAM and low-free-disk systems.
- CPU-only and integrated-GPU systems.
- NVIDIA/AMD GPU systems where available.
- GPU listed but inference backend unavailable.
- Missing runtime tools, offline start, proxy/rate limiting and interrupted downloads.
- User cancels setup at each safe cancellation point.
- Existing local GGUF selection and invalid GGUF recovery.
- App crash/restart during download, validation and model startup.
- Update detection/install and uninstall data preservation.
- Clean machine/package test, not only development mode.

### 8.3 Release gate

Do not call the installer release-ready until:
- Unit and integration tests pass with recorded output.
- Packaged PyInstaller backend and NSIS installer build successfully.
- Fresh install, upgrade, download, validation, inference smoke test and recovery scenarios pass on real Windows machines.
- Catalog license/revision/hash review is complete for every included model.
- Release notes and system requirements match tested capabilities.
- Release assets and updater metadata are verified from a packaged older install.
- Known limitations are documented; signing status is stated accurately.

## 9. Cross-cutting security and reliability requirements

- Local-only API binding and strict request validation.
- No arbitrary URL download endpoint.
- No arbitrary shell command endpoint.
- No secrets in logs or SQLite diagnostics.
- Allowlist catalog IDs, revisions and artifact filenames.
- Safe path normalization and destination containment checks.
- Bounded network timeouts/retries and subprocess timeouts.
- Atomic temporary-file finalization and database transactions.
- Clear distinction between unknown, unsupported, failed and verified.
- User confirmation for large downloads, model changes and destructive actions.
- Tests must not use production model downloads by default; mock HTTP and use small fixtures.

## 10. Suggested module layout

Keep module names consistent with repository conventions, but prefer clear separations such as:

- backend/app/model_catalog/ — manifest loader and schema validation
- backend/app/compatibility/ — memory estimator and recommendation engine
- backend/app/downloads/ — download service, job state, HTTP range handling
- backend/app/models/ — GGUF validation and registry
- backend/app/inference/ — config planner and llama.cpp integration
- desktop/src/features/setup/ — wizard UI, typed API client and state machine
- backend/app/tests/ — deterministic unit and integration tests

Avoid a single oversized main.py or a frontend-owned downloader. FastAPI should own long-running download jobs and registry transitions; the renderer observes job state and sends explicit user actions.

## 11. Milestone dependency order

1. M0 audit accepted; resolve source/build/test gaps.
2. M1 hardware schema, collectors, API and Windows tests.
3. M2 catalog schema and compatibility engine; validate manifest and estimator.
4. M4 download service and M5 validation/registry can be developed as backend services, but the UI must not treat them as ready before their acceptance tests pass.
5. M3 wizard consumes stable M1/M2/M4/M5 APIs and state contracts.
6. M6 inference planner consumes hardware/model/runtime data and integrates with existing lifecycle.
7. M7 full installer integration, upgrade/recovery QA and release.

The user-facing experience is M3 before download, but implementation can build/test M4/M5 services before wiring the wizard. This reduces UI rework and ensures the UI never promises an unsafe or incomplete download flow.

## 12. Completion definition

A supported Windows machine can install Brahma Desk, produce a partial-but-honest hardware report, see deterministic catalog recommendations with transparent assumptions, explicitly select an approved model, download the exact pinned artifact with recoverable progress, validate and register it, choose a conservative inference configuration, and successfully generate a test response. Failures must preserve user data and present a clear recovery path.
