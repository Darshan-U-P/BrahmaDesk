# Brahma Desk

Brahma Desk is a local-first Windows desktop application for chatting with a locally hosted language model. It uses Electron, React, TypeScript, and Vite for the desktop interface, with a FastAPI backend, llama.cpp for local inference, and SQLite for persistent conversations and workspace data.

The project is designed for a single machine. It does not currently provide cloud chat synchronization or account sign-in.

## What It Does

- Streams assistant responses from a llama.cpp server implementing the OpenAI-compatible Chat Completions API.
- Stores conversations, titles, and messages in SQLite.
- Supports conversation listing, reopening, renaming, message editing, answer regeneration, importing, and deletion.
- Provides local profiles with profile-specific conversations and personalization settings.
- Enforces profile ownership when accessing or modifying conversations.
- Includes database structures for projects, notes, and global or project-scoped memories.
- Retrieves relevant active memories, available project context, and earlier conversation messages within a bounded context budget.
- Exposes model sampling controls and reasoning preferences in the desktop interface.
- Includes a user-controlled Update Center for checking, downloading, and installing packaged releases.
- Packages the desktop application and FastAPI backend into a Windows x64 NSIS installer.

The llama.cpp executable and GGUF model are not bundled. They must be supplied and configured on each machine.

## Project Layout

| Path | Description |
|---|---|
| `backend/app/main.py` | FastAPI APIs, SQLite initialization and migrations, profile and conversation management, memory handling, context assembly, and llama.cpp process management |
| `backend/app/tests/` | Backend regression tests |
| `backend/requirements.txt` | Python dependencies |
| `backend/build_backend.ps1` | Builds the backend executable using PyInstaller |
| `desktop/src/` | React and TypeScript interface |
| `desktop/electron/` | Electron main process, lifecycle, updater IPC, and preload bridge |
| `desktop/e2e/` | Playwright-based Electron UI tests |
| `Test-BrahmaDesk-E2E.ps1` | Runs selected backend tests, builds the desktop application, checks backend/model health, and runs UI tests |
| `Test-BrahmaDesk-Backend.ps1` | Backend API smoke tests |

## Requirements

- Windows 10/11 and PowerShell
- Python 3.10 or later for backend development
- Python 3.12 for the configured backend packaging workflow
- Node.js and npm
- A compatible llama.cpp `llama-server` executable
- A compatible GGUF model

The end-to-end test suite also requires the project dependencies and a working local model for its live chat test.

## Configure and Run from Source

### 1. Prepare the Backend

From the project root:

```powershell
Set-Location .\backend
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Configure your local inference executable and model:

```powershell
$env:LLAMA_SERVER_EXE = "C:\path\to\llama-server.exe"
$env:LLAMA_MODEL_PATH = "C:\path\to\model.gguf"
```

Start the API:

```powershell
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

Health endpoint: http://127.0.0.1:8000/api/health

The backend reuses a ready llama.cpp server at `LLAMA_BASE_URL`. If no ready server is available, it attempts to start the configured executable and model.

### 2. Prepare the Desktop Application

Open a second PowerShell window at the project root:

```powershell
Set-Location .\desktop
npm ci
npm run dev
```

The development frontend targets `http://127.0.0.1:8000`. Keep the backend running while using the development application.

## Configuration

| Variable | Purpose | Default |
|---|---|---|
| `BRAHMA_DATA_DIR` | Override the local data directory | `%APPDATA%\brahma-desk` |
| `BRAHMA_BACKEND_PORT` | Backend port | `8000` |
| `LLAMA_BASE_URL` | llama.cpp API base URL | `http://127.0.0.1:8080` |
| `LLAMA_SERVER_EXE` | Path to the llama.cpp server executable | Developer-machine fallback in source |
| `LLAMA_MODEL_PATH` | Path to the GGUF model | Developer-machine fallback in source |
| `LLAMA_HOST` | Bind address for a backend-managed model server | `127.0.0.1` |
| `LLAMA_PORT` | Port for a backend-managed model server | `8080` |
| `LLAMA_CONTEXT_SIZE` | llama.cpp context size | `4096` |
| `LLAMA_GPU_LAYERS` | Number of GPU layers passed to llama.cpp | `20` |
| `LLAMA_LOAD_MODE` | Model loading mode | `auto` |
| `LLAMA_STARTUP_TIMEOUT` | Startup timeout in seconds | `180` |
| `LLAMA_MODEL_ID` | Fallback model identifier | `local-model` |
| `BRAHMA_CONTEXT_TOKEN_BUDGET` | Approximate context assembly budget | `3000`, minimum `1200` |

The backend passes `--no-webui` when it starts llama.cpp. Use a compatible llama.cpp build that supports this option.

## Local Data and Privacy

By default, the SQLite database, `brahma_desk.sqlite3`, and active profile selection are stored under `%APPDATA%\brahma-desk`. The `BRAHMA_DATA_DIR` environment variable overrides the backend data directory.

The database stores local conversations, profiles, settings, projects, notes, and memories. Back up the data directory before testing migrations or manually changing database contents.

Brahma Desk does not upload this data to a cloud service as part of its documented local workflow.

Profiles provide application-level data separation, not operating-system authentication. The backend is intended for local use and should not be exposed to untrusted networks without appropriate authentication and security controls.

## Backend API Overview

The interactive API documentation is available at http://127.0.0.1:8000/docs while the backend is running.

| Endpoint | Purpose |
|---|---|
| `GET /api/health` | Backend and model-server status |
| `POST /api/chat/stream` | Stream chat completions using server-sent events |
| `GET`, `POST /api/conversations` | List and create profile-owned conversations |
| `GET`, `PATCH`, `DELETE /api/conversations/{conversation_id}` | Read, rename, and delete owned conversations |
| `PATCH /api/conversations/{conversation_id}/messages/{message_id}` | Edit a user message and remove subsequent messages |
| `POST /api/conversations/import` | Import conversations and messages |
| `GET`, `POST /api/memories` | List and create or update profile-scoped global memories |
| `DELETE /api/memories/{memory_id}` | Delete an owned global memory |
| `GET`, `POST /api/profiles` | List and create profiles |
| `GET`, `PATCH`, `DELETE /api/profiles/{profile_id}` | Read, update, and delete profiles |
| `GET`, `PUT /api/profiles/{profile_id}/settings` | Read and update personalization settings |
| `GET`, `PUT /api/active-profile` | Read or change the active profile |
| `POST /api/shutdown` | Request controlled backend/model shutdown when a shutdown token is configured |

Project and note database structures exist, but project/note CRUD endpoints and a project-management screen are not yet implemented.

### Streaming Failure Behavior

Chat responses use server-sent events (SSE). An HTTP 200 response does not necessarily mean the model completed successfully. Clients must inspect SSE events for an `error` field and treat incomplete streams as failures.

The backend persists the exchange only after receiving the upstream `[DONE]` marker. Connection failures, upstream HTTP errors, read errors, and unexpected end-of-stream must not be treated as completed answers.

Partial text may already have appeared in the interface when an error occurs. An incomplete request must not be recorded as a successfully completed exchange.

## Tests and Checks

### Backend Regression Tests

From the project root:

```powershell
Set-Location .\backend
.\.venv\Scripts\Activate.ps1
Set-Location ..
python -m pytest .\backend\app\tests -q
```

These tests cover settings and payload validation, SQLite persistence, process restarts, profile isolation, model health recovery, and stream-failure behavior.

The separate memory regression suite can be run with:

```powershell
python -m unittest discover -s .\backend\tests -p "test_memory.py"
```

### Desktop Production Build

```powershell
Set-Location .\desktop
npm run build
```

### Electron End-to-End Tests

```powershell
.\Test-BrahmaDesk-E2E.ps1
```

Available options:

```powershell
.\Test-BrahmaDesk-E2E.ps1 -SkipBuild
.\Test-BrahmaDesk-E2E.ps1 -SkipUnitTests
.\Test-BrahmaDesk-E2E.ps1 -KeepStartedBackend
```

The script runs selected backend tests, builds the application, and exercises the UI against the configured live model. The live model request can take up to two minutes.

### Backend API Smoke Test

Start the backend, then run from the project root:

```powershell
.\Test-BrahmaDesk-Backend.ps1
```

Optional chat and memory checks:

```powershell
.\Test-BrahmaDesk-Backend.ps1 -IncludeChat -IncludeMemoryTests
```

The memory smoke option lists memories only; it avoids writes and deletes.

### Standalone Electron UI Tests

With a healthy backend and model running:

```powershell
Set-Location .\desktop
.\RUN-UI-TESTS.ps1
```

The runner starts its own Vite/Electron test instance and saves screenshots under `desktop\test-results\e2e`. Avoid running another Vite instance on the same port simultaneously.

## Build a Windows Installer

Build the backend executable:

```powershell
Set-Location .\backend
.\build_backend.ps1
```

Build the desktop application and Windows installer:

```powershell
Set-Location ..\desktop
npm ci
npm run build
npm run dist:win
```

The installer and update metadata are generated under `desktop\release\`. The NSIS installer targets Windows x64 and is configured to preserve user data when the application is uninstalled.

The installer does not bundle the GGUF model or `llama-server`; their paths must be configured on the target PC.

### Releases and Updates

The Update Center supports user-controlled update checks, downloads, and installation in packaged builds. Automatic downloading and automatic installation on application quit are disabled.

The configured GitHub repository is `Darshan-U-P/BrahmaDesk`. Publishing an update requires a versioned GitHub Release and the electron-builder update metadata, including `latest.yml`.

Keep release credentials in secure environment variables or CI secrets. Never commit GitHub tokens, private signing keys, PFX certificates, or certificate passwords.

Test update checks, downloads, and installation with a packaged release before distributing an update. Unsigned Windows installers may trigger Microsoft Defender SmartScreen warnings.

## Upcoming Updates

The following improvements are proposed for upcoming development. They are not claims about functionality currently available in the application.

- **Project and Notes Management:** Add frontend screens and backend CRUD APIs for creating, viewing, editing, and deleting projects and notes.
- **Improved Memory Retrieval:** Improve how relevant profile memories and project context are selected and incorporated into prompts.
- **Model Configuration:** Make local model setup easier, including executable/model path selection, connection checks, and clearer startup diagnostics.
- **Update Center Improvements:** Improve update error messages, download progress, release information, and recovery from failed update attempts.
- **Release Signing:** Add a repeatable code-signing workflow for packaged Windows executables and installers. Initially, this may use a self-signed certificate for development and testing; public distribution will require a certificate trusted by Windows.
- **Backup and Restore:** Provide a safer way to export, back up, and restore local conversations and profile data.
- **Expanded Regression Coverage:** Extend automated testing for project operations, memory behavior, model configuration, and packaged update workflows.
- **Installation and First-Run Experience:** Improve initial setup guidance so users can configure llama.cpp and their GGUF model more easily.

## Future Roadmap

Brahma Desk aims to evolve into a more complete local AI workspace while retaining its local-first design.

### Phase 1 — Stability and Reliability

- Improve installer and update reliability.
- Strengthen error handling and diagnostic logging.
- Expand backend, desktop, and end-to-end regression tests.
- Improve first-run setup and model health checks.

### Phase 2 — Workspace and Memory

- Implement project and note management.
- Improve project-scoped context and memory retrieval.
- Add reliable local backup, export, and restore workflows.
- Improve conversation search and workspace organization.

### Phase 3 — Local Model Experience

- Improve model selection and configuration.
- Provide clearer information about context size, GPU offloading, and model availability.
- Improve compatibility checks for llama.cpp builds and GGUF models.
- Explore optional support for additional local inference backends.

### Phase 4 — Advanced AI Workspace

- Explore document-aware chat and local knowledge retrieval.
- Explore workspace-aware assistance across conversations, notes, and projects.
- Investigate optional developer tools and productivity workflows.
- Evaluate additional local-first AI features based on testing and user feedback.

Roadmap priorities may change as development progresses. Planned features will be marked as implemented only after they are developed and tested.

## Known Limitations

- Context selection uses a character-based approximation rather than the model's tokenizer or semantic/vector search.
- Reasoning levels are prompt/template preferences, not portable hard limits on internal reasoning tokens. The `Off` option depends on the model's chat template and llama.cpp version.
- The backend requires a compatible llama.cpp binary supporting `--no-webui`.
- Project and note database structures exist, but their management APIs and UI are incomplete.
- Update checks, downloads, and installation require a packaged build and correctly published GitHub release metadata.
- Self-signed certificates do not establish public trust and will not reliably eliminate SmartScreen warnings.
- Profile separation is not a replacement for authentication when exposing the backend remotely.
- Passing unit tests does not guarantee that every model, llama.cpp configuration, installer, or published update works on every target machine. Run relevant build, integration, and live smoke tests before each release.

## Documentation

- [E2E Automation](documentation/README-E2E-AUTOMATION.md)
- [Update Center Mini-Update](documentation/UPDATE-CENTER-MINI-UPDATE.md)
- [Patch Notes](documentation/PATCH-NOTES.md)

## Contributing and Feedback

Bug reports, feature suggestions, and improvement ideas are welcome. When reporting a problem, include the Brahma Desk version, relevant logs, operating system details, and steps to reproduce the issue. Do not include API credentials, private keys, certificate passwords, or other secrets.
