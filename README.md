# Brahma Desk

Brahma Desk is a local-first Windows desktop application for chatting with a locally hosted language model. The desktop app is built with Electron, React, TypeScript, and Vite. A FastAPI backend manages local inference through llama.cpp and persists conversations and workspace data in SQLite.

The project is designed for a single machine and does not currently provide cloud chat synchronization or account sign-in.

## What it does

- Streams assistant responses from a llama.cpp server that implements the OpenAI-compatible chat completions API.
- Persists conversations, titles, and messages in SQLite; supports listing, reopening, renaming, editing user messages, regenerating answers, importing, and deleting conversations.
- Provides local profiles with profile-specific chats and personalization settings. The backend checks profile ownership when conversations are accessed or changed.
- Includes database tables for projects, notes, and global or project-scoped memories. The current backend exposes global memory management per profile, but does not yet expose project/note CRUD endpoints or a project-management screen.
- Retrieves relevant active memories, profile-owned project context when present in the database, and earlier messages from the current conversation, subject to a bounded context budget.
- Exposes model sampling controls and reasoning/thinking preferences in the desktop interface.
- Includes a user-controlled Update Center for checking, downloading, and installing releases in packaged builds. Automatic download and automatic installation on quit are disabled.
- Can package the desktop app as a Windows x64 NSIS installer with the FastAPI backend executable.

The llama.cpp executable and GGUF model are not bundled. They must be provided and configured on each machine running the backend.

## Project layout

| Path | Description |
| --- | --- |
| `backend/app/main.py` | FastAPI app, SQLite initialization and migrations, profile/conversation/memory APIs, context building, and llama.cpp process management |
| `backend/app/tests/` | Pytest regression tests for settings, persistence, process restart, profile isolation, model health recovery, and chat stream failures |
| `backend/requirements.txt` | Python runtime dependencies |
| `backend/build_backend.ps1` | Builds `backend\dist\brahma-backend.exe` with PyInstaller |
| `desktop/src/` | React and TypeScript interface |
| `desktop/electron/` | Electron main process, lifecycle and updater IPC, and preload bridge |
| `desktop/e2e/` | Playwright-based Electron UI test runner |
| `Test-BrahmaDesk-E2E.ps1` | Runs focused backend tests, builds the desktop app, checks backend/model health, and runs Electron UI tests |
| `Test-BrahmaDesk-Backend.ps1` | Backend API smoke test, with optional live chat and memory checks |

## Requirements

- Windows 10/11 and PowerShell
- Python 3.10 or later for backend development; Python 3.12 is used by the backend packaging script
- Node.js and npm
- A compatible llama.cpp `llama-server` executable and GGUF model

The E2E suite also requires the project Python dependencies, Playwright/Electron desktop dependencies, and a working local model for its live chat check.

## Configure and run from source

### 1. Prepare the backend

Open PowerShell at the project root:

```powershell
Set-Location .\backend
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Set the locations of the local inference executable and model. The source currently contains developer-machine fallback paths; override them for your system:

```powershell
$env:LLAMA_SERVER_EXE = "C:\path\to\llama-server.exe"
$env:LLAMA_MODEL_PATH = "C:\path\to\model.gguf"
```

Start the API:

```powershell
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

On startup the backend reuses a ready llama.cpp server at `LLAMA_BASE_URL`. If none is ready, it attempts to start the configured executable and model. The health endpoint is <http://127.0.0.1:8000/api/health>. Its response reports backend status, model-server status, model ID, whether the model is managed by this backend, and startup errors.

### 2. Prepare and start the desktop app

In a second PowerShell window at the project root:

```powershell
Set-Location .\desktop
npm ci
npm run dev
```

The desktop development server and backend are intended to listen on loopback addresses. Keep the backend running while using the development app. The renderer currently targets `http://127.0.0.1:8000`; if you change the backend port, update the renderer's API base URL to match.

## Configuration

| Variable | Purpose | Default |
| --- | --- | --- |
| `BRAHMA_DATA_DIR` | Override the local data directory | `%APPDATA%\brahma-desk` on Windows |
| `BRAHMA_BACKEND_PORT` | Backend port (also passed by Electron to the packaged backend) | `8000` |
| `LLAMA_BASE_URL` | llama.cpp API base URL | `http://127.0.0.1:8080` |
| `LLAMA_SERVER_EXE` | Path to llama.cpp server executable | Developer-machine path in source |
| `LLAMA_MODEL_PATH` | Path to GGUF model | Developer-machine path in source |
| `LLAMA_HOST` / `LLAMA_PORT` | Bind address and port if the backend starts llama.cpp | `127.0.0.1` / `8080` |
| `LLAMA_CONTEXT_SIZE` | llama.cpp context size | `4096` |
| `LLAMA_GPU_LAYERS` | Number of GPU layers passed to llama.cpp | `20` |
| `LLAMA_LOAD_MODE` | llama.cpp load mode | `auto` |
| `LLAMA_STARTUP_TIMEOUT` | Seconds to wait for a newly started model server | `180` |
| `LLAMA_MODEL_ID` | Fallback model identifier | `local-model` |
| `BRAHMA_CONTEXT_TOKEN_BUDGET` | Approximate budget for context assembly | `3000` (minimum `1200`) |

The backend passes `--no-webui` when it starts llama.cpp. Use a llama.cpp build that supports this option.

## Local data and privacy

On Windows, the SQLite database (`brahma_desk.sqlite3`) and active profile selection are stored under `%APPDATA%\brahma-desk` by default. `BRAHMA_DATA_DIR` changes the backend data directory. The packaged Electron app also uses the `brahma-desk` directory under the Windows application data path and passes it to its backend process.

The database contains local conversations, profiles, profile settings, projects, notes, and memories. Back up this directory before testing migrations or manually modifying data. The app does not upload this data to a cloud service.

Profiles provide application-level data separation, not operating-system authentication. The backend is intended for local use and must not be exposed to an untrusted network without adding appropriate authentication and security controls.

## Backend API overview

The interactive OpenAPI description is available at <http://127.0.0.1:8000/docs> while the backend is running.

| Endpoint | Purpose |
| --- | --- |
| `GET /api/health` | Backend and model-server status |
| `POST /api/chat/stream` | Stream a chat completion using server-sent events |
| `GET`, `POST /api/conversations` | List and create profile-owned conversations |
| `GET`, `PATCH`, `DELETE /api/conversations/{conversation_id}` | Load, rename, and delete an owned conversation |
| `PATCH /api/conversations/{conversation_id}/messages/{message_id}` | Edit a user message and remove later messages |
| `POST /api/conversations/import` | Import a conversation and its messages |
| `GET`, `POST /api/memories` | List and create/update profile-scoped global memories |
| `DELETE /api/memories/{memory_id}` | Delete an owned global memory |
| `GET`, `POST /api/profiles` | List and create profiles |
| `GET`, `PATCH`, `DELETE /api/profiles/{profile_id}` | Read, update, and delete profiles |
| `GET`, `PUT /api/profiles/{profile_id}/settings` | Read and update profile personalization settings |
| `GET`, `PUT /api/active-profile` | Read or change the active profile |
| `POST /api/shutdown` | Request a controlled backend/model shutdown when a shutdown token is configured |

### Streaming failure behavior

Chat responses use server-sent events. The API may already have returned HTTP 200 when a model or stream error occurs; clients must inspect the SSE events for an `error` field and treat the response as incomplete. The backend persists the exchange only after it receives the upstream `[DONE]` marker. Connection failures, upstream HTTP errors, read errors, and unexpected end-of-stream must not be treated as successful completed answers.

The desktop UI displays a connection error when it receives an error event; some partial deltas may already have appeared in the active view. It may create an empty conversation record before sending the request; an incomplete request must not create persisted user/assistant message rows or a successful completion record.

## Tests and checks

### Backend regression tests

From the project root, activate the backend environment and run:

```powershell
Set-Location .\backend
.\.venv\Scripts\Activate.ps1
Set-Location ..
python -m pytest .\backend\app\tests -q
```

These tests cover settings and payload validation, SQLite conversation operations, persistence across separate backend processes, cross-profile conversation isolation, health recovery after model outages, and chat stream error/incomplete-stream behavior. Tests use local test data and mocked model calls where possible.

The separate memory regression suite can be run with:

```powershell
python -m unittest discover -s .\backend\tests -p "test_memory.py"
```

### Desktop production build

```powershell
Set-Location .\desktop
npm run build
```

This runs the TypeScript project build and Vite production build.

### Electron UI and live-model end-to-end test

The test runner can run focused settings tests, build the desktop app, reuse a healthy backend or start one, and exercise the Electron UI against the configured live model:

```powershell
.\Test-BrahmaDesk-E2E.ps1
```

Options:

```powershell
.\Test-BrahmaDesk-E2E.ps1 -SkipBuild
.\Test-BrahmaDesk-E2E.ps1 -SkipUnitTests
.\Test-BrahmaDesk-E2E.ps1 -KeepStartedBackend
```

The script currently runs `test_settings_pipeline.py` rather than the complete backend regression suite. Run the pytest command above separately for persistence, ownership, and stream-failure coverage. The UI E2E includes a real model request and can take up to two minutes; model/server behavior can affect its result. Screenshots and logs are saved under test-results directories.

### Backend API smoke test

Start the backend first, then from the project root:

```powershell
.\Test-BrahmaDesk-Backend.ps1
```

This script checks the OpenAPI document, health, safe GET routes, and a temporary conversation CRUD lifecycle. Live chat and memory listing are optional:

```powershell
.\Test-BrahmaDesk-Backend.ps1 -IncludeChat -IncludeMemoryTests
```

The memory smoke option lists memories only; it intentionally skips writes/deletes to avoid changing persistent user data.

### Standalone Electron UI test

With a healthy backend and model already running:

```powershell
Set-Location .\desktop
.\RUN-UI-TESTS.ps1
```

The runner starts its own Vite/Electron test instance and writes screenshots under `desktop\test-results\e2e`. Do not run another Vite development instance on the same port at the same time.

## Build a Windows installer

Build the backend executable first:

```powershell
Set-Location .\backend
.\build_backend.ps1
```

Then build the desktop app and installer:

```powershell
Set-Location ..\desktop
npm ci
npm run build
npm run dist:win
```

The installer and supporting update metadata are written under `desktop\release\`. The NSIS installer is configured for Windows x64 and preserves the user-data directory when the app is uninstalled.

The build publishes updates from the GitHub repository configured in `desktop\package.json` (`Darshan-U-P/BrahmaDesk`). Publishing requires a versioned GitHub Release and the electron-builder update files, including `latest.yml`. Configure release credentials through a secure environment or CI secret; **never put a GitHub token in `package.json` or source control**. Test update checks, downloads, and installation with a packaged release before distributing them.

Neither the model nor `llama-server` is included in the installer. Their paths must be made available on the target PC. Unsigned Windows installers may show SmartScreen warnings.

## Known limitations

- Context selection uses a character-based approximation, not the model's tokenizer or semantic/vector search.
- Reasoning levels are prompt/template preferences, not portable hard limits on internal reasoning tokens. The `Off` option depends on the model chat template and llama.cpp version.
- `--no-webui` is required by the backend's model startup command and must be supported by the installed llama.cpp binary.
- Update downloads and installation require a packaged build and correctly published GitHub release metadata.
- The backend API's profile separation is not a substitute for authentication if the service is made remotely accessible.
- A passing unit test suite does not prove the configured model, llama.cpp flags, installer, or published update feed works on every target machine; run the desktop build and relevant live smoke tests before release.

## Documentation

- [README-E2E-AUTOMATION.md](./README-E2E-AUTOMATION.md) describes the end-to-end test kit.
- [UPDATE-CENTER-MINI-UPDATE.md](./UPDATE-CENTER-MINI-UPDATE.md) documents the Update Center increment; its repository-owner placeholder note predates the current publisher setting in `desktop\package.json`.
- [PATCH-NOTES.md](./PATCH-NOTES.md) records a focused E2E runner fix and its validation scope.
