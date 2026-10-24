
from __future__ import annotations

import asyncio
import json
import os
import secrets
import sqlite3
import subprocess
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import httpx
from fastapi import BackgroundTasks, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Paths and configuration
# ---------------------------------------------------------------------------

APP_DIR = Path(__file__).resolve().parent.parent
# Packaged builds set BRAHMA_DATA_DIR to Electron's per-user app-data directory.
DATA_DIR = Path(os.getenv("BRAHMA_DATA_DIR", str(APP_DIR / "data"))).expanduser()
DATA_DIR.mkdir(parents=True, exist_ok=True)
DATABASE_PATH = DATA_DIR / "brahma_desk.sqlite3"

def db_connect() -> sqlite3.Connection:
    connection = sqlite3.connect(DATABASE_PATH, timeout=10)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection

def init_database() -> None:
    """Create the original chat tables, then apply versioned local migrations.

    Each migration is idempotent where practical. Existing conversations are
    assigned to a single default profile; existing conversation/message IDs
    and message contents are preserved.
    """
    with db_connect() as db:
        db.execute("PRAGMA journal_mode = WAL")
        db.execute("""CREATE TABLE IF NOT EXISTS conversations (
            id TEXT PRIMARY KEY, title TEXT NOT NULL, created_at REAL NOT NULL,
            updated_at REAL NOT NULL)""")
        db.execute("""CREATE TABLE IF NOT EXISTS messages (
            id TEXT PRIMARY KEY, conversation_id TEXT NOT NULL,
            role TEXT NOT NULL CHECK(role IN ('user','assistant','system')),
            content TEXT NOT NULL, created_at REAL NOT NULL,
            FOREIGN KEY(conversation_id) REFERENCES conversations(id) ON DELETE CASCADE)""")
        db.execute("CREATE INDEX IF NOT EXISTS idx_messages_conversation ON messages(conversation_id, created_at)")
        db.execute("""CREATE TABLE IF NOT EXISTS schema_migrations (
            version INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            applied_at REAL NOT NULL
        )""")

        applied = {row[0] for row in db.execute("SELECT version FROM schema_migrations").fetchall()}

        # Migration 1: local profiles. A stable default profile keeps this
        # migration repeatable and gives legacy chats a single owner.
        if 1 not in applied:
            db.execute("""CREATE TABLE IF NOT EXISTS profiles (
                id TEXT PRIMARY KEY,
                display_name TEXT NOT NULL,
                avatar_color TEXT NOT NULL DEFAULT '#7968E8',
                is_default INTEGER NOT NULL DEFAULT 0 CHECK(is_default IN (0,1)),
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL
            )""")
            now = time.time()
            db.execute("""INSERT OR IGNORE INTO profiles
                (id, display_name, avatar_color, is_default, created_at, updated_at)
                VALUES (?, ?, ?, 1, ?, ?)""",
                ("default-profile", "Default Profile", "#7968E8", now, now))
            db.execute("INSERT INTO schema_migrations(version, name, applied_at) VALUES(1, ?, ?)",
                       ("create_profiles", now))

        # Migration 2: add profile ownership without rebuilding the existing
        # conversations table or changing any existing conversation/message IDs.
        applied = {row[0] for row in db.execute("SELECT version FROM schema_migrations").fetchall()}
        if 2 not in applied:
            columns = {row[1] for row in db.execute("PRAGMA table_info(conversations)").fetchall()}
            if "profile_id" not in columns:
                db.execute("ALTER TABLE conversations ADD COLUMN profile_id TEXT REFERENCES profiles(id) ON DELETE CASCADE")
            db.execute("UPDATE conversations SET profile_id = ? WHERE profile_id IS NULL", ("default-profile",))
            db.execute("CREATE INDEX IF NOT EXISTS idx_conversations_profile_updated ON conversations(profile_id, updated_at DESC)")
            now = time.time()
            db.execute("INSERT INTO schema_migrations(version, name, applied_at) VALUES(2, ?, ?)",
                       ("assign_conversations_to_default_profile", now))

        # Migration 3: project workspace, notes, memory scopes, and per-profile settings.
        applied = {row[0] for row in db.execute("SELECT version FROM schema_migrations").fetchall()}
        if 3 not in applied:
            db.execute("""CREATE TABLE IF NOT EXISTS projects (
                id TEXT PRIMARY KEY,
                profile_id TEXT NOT NULL,
                name TEXT NOT NULL,
                description TEXT NOT NULL DEFAULT '',
                instructions TEXT NOT NULL DEFAULT '',
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                archived_at REAL,
                FOREIGN KEY(profile_id) REFERENCES profiles(id) ON DELETE CASCADE
            )""")
            db.execute("CREATE INDEX IF NOT EXISTS idx_projects_profile_updated ON projects(profile_id, updated_at DESC)")
            db.execute("""CREATE TABLE IF NOT EXISTS project_notes (
                id TEXT PRIMARY KEY,
                project_id TEXT NOT NULL,
                title TEXT NOT NULL,
                content TEXT NOT NULL DEFAULT '',
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                FOREIGN KEY(project_id) REFERENCES projects(id) ON DELETE CASCADE
            )""")
            db.execute("CREATE INDEX IF NOT EXISTS idx_project_notes_project_updated ON project_notes(project_id, updated_at DESC)")
            db.execute("""CREATE TABLE IF NOT EXISTS memories (
                id TEXT PRIMARY KEY,
                profile_id TEXT NOT NULL,
                project_id TEXT,
                category TEXT NOT NULL DEFAULT 'general',
                content TEXT NOT NULL,
                source TEXT NOT NULL DEFAULT 'manual',
                status TEXT NOT NULL DEFAULT 'active' CHECK(status IN ('pending','active','archived')),
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                last_used_at REAL,
                FOREIGN KEY(profile_id) REFERENCES profiles(id) ON DELETE CASCADE,
                FOREIGN KEY(project_id) REFERENCES projects(id) ON DELETE CASCADE
            )""")
            db.execute("CREATE INDEX IF NOT EXISTS idx_memories_profile_status_updated ON memories(profile_id, status, updated_at DESC)")
            db.execute("CREATE INDEX IF NOT EXISTS idx_memories_project_status ON memories(project_id, status)")
            db.execute("""CREATE TABLE IF NOT EXISTS profile_settings (
                profile_id TEXT PRIMARY KEY,
                settings_json TEXT NOT NULL DEFAULT '{}',
                updated_at REAL NOT NULL,
                FOREIGN KEY(profile_id) REFERENCES profiles(id) ON DELETE CASCADE
            )""")
            now = time.time()
            db.execute("INSERT OR IGNORE INTO profile_settings(profile_id, settings_json, updated_at) VALUES(?, '{}', ?)",
                       ("default-profile", now))
            db.execute("INSERT INTO schema_migrations(version, name, applied_at) VALUES(3, ?, ?)",
                       ("create_projects_memories_and_settings", now))

        # Keep the default profile/settings present if a previous run was
        # interrupted between migration steps. No user data is overwritten.
        now = time.time()
        db.execute("""INSERT OR IGNORE INTO profiles
            (id, display_name, avatar_color, is_default, created_at, updated_at)
            VALUES (?, ?, ?, 1, ?, ?)""",
            ("default-profile", "Default Profile", "#7968E8", now, now))
        db.execute("INSERT OR IGNORE INTO profile_settings(profile_id, settings_json, updated_at) VALUES(?, '{}', ?)",
                   ("default-profile", now))

init_database()

LLAMA_BASE_URL = os.getenv(
    "LLAMA_BASE_URL",
    "http://127.0.0.1:8080",
).rstrip("/")

LLAMA_SERVER_EXE = os.getenv(
    "LLAMA_SERVER_EXE",
    r"D:\all project\llama-b11552-bin-win-cuda-12.4-x64\llama-server.exe",
)

LLAMA_MODEL_PATH = os.getenv(
    "LLAMA_MODEL_PATH",
    r"D:\all project\Projects\AI projects\BrahmaDesk\BrahmaDesk-Phase0\Models\Qwen_Qwen3-4B-Instruct-2507-Q4_K_M.gguf",
)

LLAMA_HOST = os.getenv("LLAMA_HOST", "127.0.0.1")
LLAMA_PORT = int(os.getenv("LLAMA_PORT", "8080"))
BACKEND_PORT = int(os.getenv("BRAHMA_BACKEND_PORT", "8000"))
LLAMA_CONTEXT_SIZE = int(os.getenv("LLAMA_CONTEXT_SIZE", "4096"))
LLAMA_GPU_LAYERS = os.getenv("LLAMA_GPU_LAYERS", "20")
LLAMA_LOAD_MODE = os.getenv("LLAMA_LOAD_MODE", "auto")
LLAMA_STARTUP_TIMEOUT = float(os.getenv("LLAMA_STARTUP_TIMEOUT", "180"))
MODEL_ID = os.getenv("LLAMA_MODEL_ID", "local-model")

# Electron generates a random token and passes it to this backend.
# The shutdown endpoint is disabled unless a token is configured.
SHUTDOWN_TOKEN = os.getenv("BRAHMA_SHUTDOWN_TOKEN", "")

# Track only a model process launched by this backend.
llama_process: subprocess.Popen | None = None
llama_process_owned = False
llama_start_error: str | None = None

# Prevent simultaneous shutdown requests from stopping the process twice.
shutdown_lock = asyncio.Lock()


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------

class Message(BaseModel):
    role: str = Field(pattern="^(system|user|assistant)$")
    content: str = Field(min_length=1, max_length=100_000)


class ChatRequest(BaseModel):
    conversation_id: str | None = None
    messages: list[Message] = Field(min_length=1, max_length=100)
    temperature: float = Field(default=0.7, ge=0, le=2)
    max_tokens: int = Field(default=512, ge=1, le=8192)


# ---------------------------------------------------------------------------
# llama.cpp process management
# ---------------------------------------------------------------------------

async def llama_is_ready() -> bool:
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            response = await client.get(f"{LLAMA_BASE_URL}/v1/models")
            return response.status_code == 200
    except httpx.HTTPError:
        return False


def stop_owned_llama() -> None:
    """Stop llama.cpp only if this backend started and owns the process."""
    global llama_process, llama_process_owned, llama_start_error

    process = llama_process

    if not llama_process_owned or process is None:
        return

    try:
        if process.poll() is None:
            print(
                "[Brahma Desk] Stopping llama.cpp server "
                "started by this backend...",
                flush=True,
            )

            process.terminate()

            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                print(
                    "[Brahma Desk] llama.cpp did not stop in time; "
                    "forcing termination.",
                    flush=True,
                )
                process.kill()
                process.wait(timeout=5)

        print("[Brahma Desk] Owned llama.cpp process stopped.", flush=True)

    except (OSError, subprocess.SubprocessError) as exc:
        print(
            f"[Brahma Desk] Error stopping llama.cpp: {exc}",
            flush=True,
        )
        raise

    finally:
        # Do not retain a stale process reference after shutdown.
        if process.poll() is not None:
            llama_process = None
            llama_process_owned = False


async def start_llama_if_needed() -> None:
    global llama_process, llama_process_owned, llama_start_error

    # Reuse a server that was already running. We do not own it and must
    # never terminate it during Brahma Desk shutdown.
    if await llama_is_ready():
        print(
            f"[Brahma Desk] Reusing existing llama.cpp server "
            f"at {LLAMA_BASE_URL}",
            flush=True,
        )
        llama_start_error = None
        return

    executable = Path(LLAMA_SERVER_EXE)
    model = Path(LLAMA_MODEL_PATH)

    if not executable.is_file():
        llama_start_error = f"llama-server.exe not found: {executable}"
        print(f"[Brahma Desk] ERROR: {llama_start_error}", flush=True)
        return

    if not model.is_file():
        llama_start_error = f"GGUF model not found: {model}"
        print(f"[Brahma Desk] ERROR: {llama_start_error}", flush=True)
        return

    command = [
        str(executable),
        "-m",
        str(model),
        "--host",
        LLAMA_HOST,
        "--port",
        str(LLAMA_PORT),
        "-c",
        str(LLAMA_CONTEXT_SIZE),
        "-ngl",
        str(LLAMA_GPU_LAYERS),
        "--load-mode",
        LLAMA_LOAD_MODE,
    ]

    try:
        process = subprocess.Popen(
            command,
            cwd=str(executable.parent),
        )

        llama_process = process
        llama_process_owned = True
        llama_start_error = None

        print(
            f"[Brahma Desk] Starting local model: {model.name}",
            flush=True,
        )

        deadline = time.monotonic() + LLAMA_STARTUP_TIMEOUT

        while time.monotonic() < deadline:
            if process.poll() is not None:
                llama_start_error = (
                    "llama-server.exe exited early with code "
                    f"{process.returncode}"
                )
                print(
                    f"[Brahma Desk] ERROR: {llama_start_error}",
                    flush=True,
                )
                return

            if await llama_is_ready():
                print(
                    f"[Brahma Desk] Local model is ready at "
                    f"{LLAMA_BASE_URL}",
                    flush=True,
                )
                return

            await asyncio.sleep(1)

        llama_start_error = (
            f"Timed out after {LLAMA_STARTUP_TIMEOUT:.0f}s "
            f"waiting for llama.cpp at {LLAMA_BASE_URL}"
        )
        print(f"[Brahma Desk] ERROR: {llama_start_error}", flush=True)

    except OSError as exc:
        llama_start_error = f"Could not start llama-server.exe: {exc}"
        print(f"[Brahma Desk] ERROR: {llama_start_error}", flush=True)


# ---------------------------------------------------------------------------
# FastAPI lifecycle
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(_: FastAPI):
    await start_llama_if_needed()

    try:
        yield
    finally:
        # This also handles a normal backend shutdown.
        async with shutdown_lock:
            try:
                stop_owned_llama()
            except (OSError, subprocess.SubprocessError) as exc:
                print(
                    f"[Brahma Desk] Model shutdown error: {exc}",
                    flush=True,
                )


app = FastAPI(
    title="Brahma Desk API",
    version="0.2.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5173",
        "http://localhost:5173",
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST", "PATCH", "DELETE"],
    allow_headers=["Content-Type"],
)


# ---------------------------------------------------------------------------
# Health endpoint
# ---------------------------------------------------------------------------

@app.get("/api/health")
async def health() -> dict[str, Any]:
    status: dict[str, Any] = {
        "backend": "ok",
        "llama_server": "unreachable",
        "llama_base_url": LLAMA_BASE_URL,
        "model": None,
        "managed_by_backend": llama_process_owned,
        "llama_process_running": (
            llama_process is not None and llama_process.poll() is None
        ),
        "startup_error": llama_start_error,
    }

    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            response = await client.get(f"{LLAMA_BASE_URL}/v1/models")
            response.raise_for_status()
            payload = response.json()

            status["llama_server"] = "ok"

            data = payload.get("data") or payload.get("models") or []
            if data:
                status["model"] = data[0].get("id") or data[0].get("name")

    except (httpx.HTTPError, ValueError):
        pass

    return status


# ---------------------------------------------------------------------------
# Authenticated shutdown endpoint
# ---------------------------------------------------------------------------

async def request_uvicorn_shutdown() -> None:
    # Allow the HTTP response to leave the server before requesting exit.
    await asyncio.sleep(0.25)
    server = getattr(app.state, "uvicorn_server", None)
    if server is not None:
        print("[Brahma Desk] Graceful Uvicorn shutdown requested.", flush=True)
        server.should_exit = True

@app.post("/api/shutdown")
async def shutdown_backend(
    background_tasks: BackgroundTasks,
    x_brahma_shutdown_token: str = Header(default=""),
) -> dict[str, str]:
    if not SHUTDOWN_TOKEN or not secrets.compare_digest(
        x_brahma_shutdown_token,
        SHUTDOWN_TOKEN,
    ):
        raise HTTPException(status_code=403, detail="Forbidden")

    async with shutdown_lock:
        try:
            stop_owned_llama()
        except (OSError, subprocess.SubprocessError) as exc:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to stop the owned model process: {exc}",
            ) from exc

    background_tasks.add_task(request_uvicorn_shutdown)
    return {"status": "shutdown_requested"}

# ---------------------------------------------------------------------------
# Streaming chat endpoint
# ---------------------------------------------------------------------------

@app.post("/api/chat/stream")
async def chat_stream(request: ChatRequest) -> StreamingResponse:
    request_id = str(uuid.uuid4())

    payload = {
        "model": MODEL_ID,
        "messages": [message.model_dump() for message in request.messages],
        "temperature": request.temperature,
        "max_tokens": request.max_tokens,
        "stream": True,
    }

    async def generate():
        answer = ""
        started = time.time()
        conversation_id = request.conversation_id

        try:
            async with httpx.AsyncClient(timeout=None) as client:
                async with client.stream(
                    "POST",
                    f"{LLAMA_BASE_URL}/v1/chat/completions",
                    json=payload,
                    headers={"Accept": "text/event-stream"},
                ) as upstream:
                    if upstream.status_code >= 400:
                        body = await upstream.aread()
                        yield (
                            f"data: {json.dumps({'error': body.decode('utf-8', errors='replace')})}\n\n"
                        )
                        yield "data: [DONE]\n\n"
                        return

                    async for line in upstream.aiter_lines():
                        if not line.startswith("data:"):
                            continue

                        raw = line[5:].strip()

                        if raw == "[DONE]":
                            break

                        try:
                            event = json.loads(raw)
                        except json.JSONDecodeError:
                            continue

                        choices = event.get("choices") or []
                        if not choices:
                            continue

                        delta = (choices[0].get("delta") or {}).get("content")

                        if delta:
                            answer += delta
                            yield f"data: {json.dumps({'delta': delta})}\n\n"

            # Persist the complete exchange only after generation finishes.
            if not conversation_id:
                conversation_id = str(uuid.uuid4())
                first_user = next((m.content for m in request.messages if m.role == "user"), "New chat")
                title = first_user.strip().replace("\n", " ")[:60] or "New chat"
                with db_connect() as db:
                    db.execute("INSERT INTO conversations(id,title,created_at,updated_at) VALUES(?,?,?,?)",
                               (conversation_id, title, started, started))
            with db_connect() as db:
                exists = db.execute("SELECT 1 FROM conversations WHERE id=?", (conversation_id,)).fetchone()
                if exists:
                    # Avoid duplicating messages already stored for this conversation.
                    stored = db.execute("SELECT COUNT(*) FROM messages WHERE conversation_id=?", (conversation_id,)).fetchone()[0]
                    incoming = [m for m in request.messages if m.role in ("user", "assistant")]
                    for message in incoming[stored:]:
                        db.execute("INSERT INTO messages(id,conversation_id,role,content,created_at) VALUES(?,?,?,?,?)",
                                   (str(uuid.uuid4()), conversation_id, message.role, message.content, time.time()))
                    if answer:
                        db.execute("INSERT INTO messages(id,conversation_id,role,content,created_at) VALUES(?,?,?,?,?)",
                                   (str(uuid.uuid4()), conversation_id, "assistant", answer, time.time()))
                    existing_title = db.execute("SELECT title FROM conversations WHERE id=?", (conversation_id,)).fetchone()
                    if existing_title and existing_title["title"] == "New chat":
                        first_user = next((m.content for m in request.messages if m.role == "user"), "New chat")
                        suggested_title = first_user.strip().replace("\n", " ")[:60] or "New chat"
                        db.execute("UPDATE conversations SET title=?, updated_at=? WHERE id=?", (suggested_title, time.time(), conversation_id))
                    else:
                        db.execute("UPDATE conversations SET updated_at=? WHERE id=?", (time.time(), conversation_id))

            yield (
                f"data: {json.dumps({'done': True, 'request_id': request_id, 'conversation_id': conversation_id})}\n\n"
            )

        except httpx.HTTPError as exc:
            yield (
                f"data: {json.dumps({'error': f'Cannot reach llama.cpp server: {exc}'})}\n\n"
            )

        finally:
            yield "data: [DONE]\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


# ---------------------------------------------------------------------------
# Persistent conversation API (SQLite)
# ---------------------------------------------------------------------------

class ConversationCreate(BaseModel):
    title: str = Field(default="New chat", min_length=1, max_length=200)

class ConversationRename(BaseModel):
    title: str = Field(min_length=1, max_length=200)

class MessageEdit(BaseModel):
    content: str = Field(min_length=1, max_length=100_000)

@app.get("/api/conversations")
async def conversations() -> dict[str, Any]:
    with db_connect() as db:
        rows = db.execute("""SELECT c.id, c.title, c.created_at, c.updated_at,
            (SELECT content FROM messages m WHERE m.conversation_id=c.id
             ORDER BY m.created_at DESC LIMIT 1) AS preview
            FROM conversations c ORDER BY c.updated_at DESC""").fetchall()
    return {"items": [dict(row) for row in rows]}

@app.post("/api/conversations")
async def create_conversation(request: ConversationCreate) -> dict[str, Any]:
    now = time.time(); conversation_id = str(uuid.uuid4())
    with db_connect() as db:
        db.execute("INSERT INTO conversations(id,title,created_at,updated_at) VALUES(?,?,?,?)",
                   (conversation_id, request.title.strip() or "New chat", now, now))
    return {"id": conversation_id, "title": request.title.strip() or "New chat", "created_at": now, "updated_at": now, "messages": []}

@app.get("/api/conversations/{conversation_id}")
async def get_conversation(conversation_id: str) -> dict[str, Any]:
    with db_connect() as db:
        conv = db.execute("SELECT * FROM conversations WHERE id=?", (conversation_id,)).fetchone()
        if conv is None: raise HTTPException(status_code=404, detail="Conversation not found")
        messages = db.execute("SELECT id, role, content, created_at FROM messages WHERE conversation_id=? ORDER BY created_at, rowid", (conversation_id,)).fetchall()
    return {**dict(conv), "messages": [dict(m) for m in messages]}

@app.patch("/api/conversations/{conversation_id}")
async def rename_conversation(conversation_id: str, request: ConversationRename) -> dict[str, Any]:
    title = request.title.strip()
    if not title: raise HTTPException(status_code=422, detail="Title cannot be empty")
    with db_connect() as db:
        result = db.execute("UPDATE conversations SET title=?, updated_at=? WHERE id=?", (title, time.time(), conversation_id))
        if result.rowcount == 0: raise HTTPException(status_code=404, detail="Conversation not found")
    return {"id": conversation_id, "title": title}


@app.patch("/api/conversations/{conversation_id}/messages/{message_id}")
async def edit_message(conversation_id: str, message_id: str, request: MessageEdit) -> dict[str, Any]:
    content = request.content.strip()
    if not content:
        raise HTTPException(status_code=422, detail="Message cannot be empty")
    with db_connect() as db:
        row = db.execute("SELECT rowid, role FROM messages WHERE id=? AND conversation_id=?", (message_id, conversation_id)).fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail="Message not found")
        if row["role"] != "user":
            raise HTTPException(status_code=400, detail="Only user messages can be edited")
        db.execute("UPDATE messages SET content=? WHERE id=?", (content, message_id))
        db.execute("DELETE FROM messages WHERE conversation_id=? AND rowid>?", (conversation_id, row["rowid"]))
        db.execute("UPDATE conversations SET updated_at=? WHERE id=?", (time.time(), conversation_id))
    return {"status": "updated", "conversation_id": conversation_id, "message_id": message_id}

@app.delete("/api/conversations/{conversation_id}")
async def delete_conversation(conversation_id: str) -> dict[str, str]:
    with db_connect() as db:
        result = db.execute("DELETE FROM conversations WHERE id=?", (conversation_id,))
        if result.rowcount == 0: raise HTTPException(status_code=404, detail="Conversation not found")
    return {"status": "deleted", "id": conversation_id}

