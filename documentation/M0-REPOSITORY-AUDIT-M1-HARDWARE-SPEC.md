# M0 — Repository Audit & M1 Hardware Detection Specification

**Repository:** Darshan-U-P/BrahmaDesk  
**Audit baseline:** main, v0.2.0 source tree inspected on 2026-10-25  
**Status:** Remote source inspection only. No local Windows build or runtime tests were executed during this audit.

## 1. Executive summary

Brahma Desk already has a useful foundation: Electron owns the desktop lifecycle, React provides the renderer, a packaged FastAPI executable provides the local API, SQLite stores user data, and the backend starts or reuses a llama.cpp server. Existing profile, conversation, settings, memory, and update mechanisms should be reused.

M1 should begin as a read-only diagnostic capability. It must report evidence and confidence for each hardware field, tolerate unavailable data, and never silently download, launch, or reconfigure an inference backend. Model recommendations and automatic configuration belong in a later milestone.

## 2. Inspected source map

| Area | Existing files | Findings |
|---|---|---|
| Desktop packaging | desktop/package.json | Electron 33, electron-builder 25, Windows x64 NSIS target, version 0.2.0. Backend EXE is an extra resource. |
| Electron main process | desktop/electron/main.cjs | Sets stable per-user app-data path, starts packaged backend, waits for API health, writes backend logs, requests token-protected shutdown, integrates updater. |
| Preload bridge | desktop/electron/preload.cjs | Uses contextBridge and narrowly exposed IPC for version and updater state/actions. |
| Renderer | desktop/src/App.tsx | Chat UI, backend/model health state, profile/settings UI and Update Center. Renderer calls the local API at 127.0.0.1:8000. Sampling/reasoning preferences use localStorage. |
| Backend API and inference | backend/app/main.py | FastAPI API, SQLite initialization/migrations, profile/conversation/memory context, llama.cpp health/start/stop lifecycle. |
| Backend dependencies | backend/requirements.txt | FastAPI, Uvicorn, HTTPX and Pydantic; no dedicated hardware-detection dependency is declared. |
| Packaged backend | backend/build_backend.ps1, backend/run_backend.py | PyInstaller one-file EXE and Python 3.12 build environment. |
| Desktop packaging | desktop/package.json, desktop/build_backend.ps1 | NSIS configuration and a second backend build script exist; confirm which script is canonical. |
| Regression tests | backend/app/tests/*, backend/tests/test_memory.py | Coverage includes stream failures, persistence, process restarts, profile isolation, settings, model-health recovery and memory. |
| Desktop E2E | desktop/e2e/run-electron-ui-tests.cjs, Test-BrahmaDesk-E2E.ps1 | Playwright/Electron test runner and combined build/test workflow. |
| Branding/docs | desktop/build/*, documentation/*, README.md | Branded v0.2.0 assets, patch notes, updater guidance and test instructions. |

## 3. Existing mechanisms to reuse

### Desktop lifecycle and security boundary
- Keep Electron as lifecycle owner for the packaged application and backend process.
- Keep Node/system access in the Electron main process; renderer features should use narrow preload IPC or the loopback API.
- Expose hardware diagnostics through an explicit, read-only API call initially. Do not expose arbitrary shell commands to the renderer.
- Continue binding the backend to loopback; do not turn hardware diagnostics into a remote system-information endpoint.

### Inference and model startup
- Reuse llama.cpp health checking and startup diagnostics as runtime status, separate from hardware capability.
- Existing configuration includes LLAMA_BASE_URL, LLAMA_SERVER_EXE, LLAMA_MODEL_PATH, LLAMA_HOST, LLAMA_PORT, LLAMA_CONTEXT_SIZE, LLAMA_GPU_LAYERS, LLAMA_LOAD_MODE and startup timeout.
- The backend checks whether the server is ready, validates configured executable/model paths, then starts the process with configured arguments.
- Current defaults include developer-machine paths. Hardware reporting must not depend on those paths existing and must not infer hardware from whether a model server starts.
- Current health status proves endpoint reachability, not exact accelerator backend or actual GPU offload. M1 must report unknown unless independently detected.

### Persistence
- SQLite defaults to %APPDATA%\\brahma-desk, with BRAHMA_DATA_DIR override.
- Versioned migrations cover profiles, profile-owned conversations, projects, project notes, memories and profile settings.
- M1 does not require database persistence initially. Return a fresh report on request; consider caching only if later requirements justify staleness/invalidation rules.
- Do not store serial numbers, machine identifiers, usernames or raw environment dumps in a diagnostic report.

### Updates and packaging
- NSIS supports install-directory selection, shortcuts, launch-after-setup and preservation of app data on uninstall.
- electron-updater is integrated through main-process IPC. Hardware detection should not modify updater behavior.
- Packaged Electron expects backend/brahma-backend.exe under resources.
- backend/build_backend.ps1 and desktop/build_backend.ps1 overlap. Confirm which is canonical in CI and the documented release workflow before modifying either.
- Pull request #2 is intended to validate Windows packaging artifacts. Packaging CI is not equivalent to clean-install, upgrade, model-download, updater-install or uninstall data-preservation testing.

## 4. Risks and gaps

1. No normalized hardware capability report or dedicated hardware module was identified in the inspected source.
2. GPU VRAM and inference-backend support cannot reliably be inferred from llama.cpp health alone.
3. Windows telemetry sources can fail or return partial data. Report per-field availability/warnings rather than fail the entire report.
4. CPU feature detection must distinguish OS/architecture from instruction-set support. CPU marketing names are not proof of AVX/AVX2/AVX-512 support.
5. Report physical RAM and currently available RAM separately. Report free space on the relevant data/model volume, not just the system drive.
6. GPU inventory is not equivalent to usable inference acceleration. Driver/runtime/build compatibility must be separately verified.
7. Collectors must work inside a PyInstaller one-file executable, not only under a developer interpreter.
8. Avoid admin-only probes, network requests, package installation, benchmarks, or launching a model during initial detection.
9. Findings are based on source inspection, not proof that tests, builds or installer flows pass on Windows.

## 5. Proposed M1 architecture

### Suggested module boundaries

- backend/app/hardware/__init__.py
- backend/app/hardware/schemas.py — normalized response models and enums.
- backend/app/hardware/detector.py — orchestration only.
- backend/app/hardware/windows.py — OS, CPU, memory, storage and GPU collectors.
- Tests should follow the existing backend/app/tests/ convention unless a dedicated test package is preferred.

Keep collectors independently testable; do not place all detection logic inside backend/app/main.py.

### Proposed API

GET /api/hardware — return a fresh normalized report.

The route should be read-only, use the existing local-only server setup, and return a valid partial report even when an individual collector fails. Use bounded subprocess timeouts where subprocesses are necessary.

### Normalized report contract (draft)

{
  "schema_version": 1,
  "collected_at": "ISO-8601 timestamp",
  "os": {
    "name": "Windows",
    "version": "string or null",
    "architecture": "x86_64",
    "supported": true
  },
  "cpu": {
    "name": "string or null",
    "architecture": "x86_64",
    "physical_cores": 8,
    "logical_processors": 16,
    "instruction_sets": {
      "sse2": "supported | unsupported | unknown",
      "avx": "supported | unsupported | unknown",
      "avx2": "supported | unsupported | unknown",
      "avx512": "supported | unsupported | unknown"
    }
  },
  "memory": {
    "physical_total_bytes": null,
    "physical_available_bytes": null
  },
  "gpus": [
    {
      "name": "adapter name",
      "vendor": "NVIDIA | AMD | Intel | Other | Unknown",
      "dedicated_vram_bytes": null,
      "shared_memory_bytes": null,
      "driver_version": null,
      "detection_source": "windows",
      "inference_backends": {
        "cuda": "supported | unsupported | unknown",
        "vulkan": "supported | unsupported | unknown",
        "directml": "supported | unsupported | unknown"
      }
    }
  ],
  "storage": {
    "data_directory": "Brahma Desk data path",
    "data_volume_free_bytes": null,
    "model_candidates": []
  },
  "diagnostics": {
    "complete": false,
    "warnings": [],
    "collector_status": {}
  }
}

This is a draft, not a frozen API contract. Implement with Pydantic models. Use null/unknown rather than invented defaults. Backend support must mean verified availability of the relevant runtime, not merely presence of a GPU vendor.

### Detection strategy

- OS and architecture: Python standard library plus Windows version APIs if needed.
- CPU identity and core counts: Windows system APIs/WMI or standard-library information; keep optional dependencies minimal.
- Instruction sets: use a reliable native/OS capability source. If reliable detection is unavailable, report unknown.
- RAM: Windows memory status API or a maintained helper if justified. Return bytes.
- GPU inventory and VRAM: documented Windows adapter/driver inventory, with optional vendor tooling where available. Fall back gracefully when tooling is absent.
- Inference backend capability: separately inspect the selected llama.cpp binary/build and relevant runtime availability. GPU inventory alone is insufficient.
- Storage: query free bytes for the data directory and configured model locations only; do not recursively scan whole disks.
- Failure isolation: each collector returns data plus warnings/status; one failure must not prevent other sections from being reported.

## 6. M1 test and acceptance plan

### Unit tests with mocked system probes
- Complete report on a representative Windows fixture.
- No GPU present.
- GPU present but VRAM/driver details unavailable.
- Missing WMI or vendor CLI.
- Unknown CPU instruction support.
- Missing/inaccessible storage path.
- A collector exception still yields a partial report.
- Byte units and unknown/null semantics remain consistent.
- Schema version and response fields remain stable.

### Windows integration tests
- Compare CPU/core counts, RAM, GPU model/VRAM and free space with Windows tools on a known PC.
- Verify no administrator elevation is required.
- Verify source and packaged PyInstaller execution.
- Verify scan completes within an agreed latency budget and does not start a model or download data.
- Test both integrated-GPU/no-vendor-tool and NVIDIA configurations where available.

### Completion criteria
- Every required field is measured from a documented source or explicitly unknown.
- GPU absence and partial telemetry do not fail the endpoint.
- Hardware inventory is distinct from verified inference-backend support.
- Tests cover missing tools, exceptions and packaged execution.
- README/API documentation explains collected data and limitations.

## 7. Suggested implementation sequence

1. Approve report schema and privacy boundaries.
2. Implement schemas and collector interfaces.
3. Implement OS/CPU/RAM/storage collectors.
4. Add GPU inventory with graceful fallback.
5. Add GET /api/hardware and focused unit tests.
6. Add a diagnostics UI only after the API contract is stable.
7. Verify Windows source and packaged builds; record actual command output.
8. Keep model recommendations and automatic configuration out of M1 unless separately scoped.

## 8. Open decisions

- Windows 10/11 is the currently documented baseline; confirm whether both remain required.
- Decide whether GPU inventory uses WMI/PowerShell only or may call vendor utilities when present.
- Decide whether M1 is API-only or includes a diagnostics panel.
- Keep model candidate paths limited to configured locations initially; user-selected folder discovery can be scoped later.

## 9. Audit checklist

- [x] Repository tree and key source files inspected remotely.
- [x] Electron/backend/inference/persistence/updater/packaging mechanisms mapped.
- [x] M1 module boundaries, response design, failure handling and test plan drafted.
- [ ] Run backend regression tests locally.
- [ ] Run desktop production build and Windows installer validation.
- [ ] Validate installer and packaged backend on Windows.
- [ ] Approve open decisions before M1 implementation.
