"""Regression tests for Brahma Desk's SQLite-backed conversation APIs.

Uses the isolated BRAHMA_DATA_DIR configured by test_settings_pipeline.py.
These tests do not start llama.cpp or call the live model.
"""

import importlib
import os
import sys
import uuid
from pathlib import Path

from fastapi.testclient import TestClient

BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))
os.environ.setdefault("BRAHMA_DATA_DIR", str(Path(__file__).resolve().parent / ".test-data"))
main = importlib.import_module("main")

client = TestClient(main.app)


def unique_title(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:10]}"


def test_create_conversation_is_listed_and_can_be_reopened():
    title = unique_title("persistence-create")

    created = client.post("/api/conversations", json={"title": title})
    assert created.status_code == 200, created.text
    conversation_id = created.json()["id"]

    listing = client.get("/api/conversations")
    assert listing.status_code == 200, listing.text
    assert any(item["id"] == conversation_id and item["title"] == title
               for item in listing.json()["items"])

    reopened = client.get(f"/api/conversations/{conversation_id}")
    assert reopened.status_code == 200, reopened.text
    assert reopened.json()["id"] == conversation_id
    assert reopened.json()["title"] == title
    assert reopened.json()["messages"] == []


def test_imported_messages_persist_in_order_and_survive_new_db_connection():
    title = unique_title("persistence-messages")
    user_text = f"user-{uuid.uuid4().hex}"
    assistant_text = f"assistant-{uuid.uuid4().hex}"

    imported = client.post("/api/conversations/import", json={
        "conversation": {
            "title": title,
            "messages": [
                {"role": "user", "content": user_text},
                {"role": "assistant", "content": assistant_text},
            ],
        }
    })
    assert imported.status_code == 200, imported.text
    conversation_id = imported.json()["id"]

    # Read using a new SQLite connection, not an in-memory test copy.
    with main.db_connect() as db:
        conversation = db.execute(
            "SELECT title FROM conversations WHERE id = ?", (conversation_id,)
        ).fetchone()
        rows = db.execute(
            "SELECT role, content FROM messages WHERE conversation_id = ? ORDER BY created_at, rowid",
            (conversation_id,),
        ).fetchall()

    assert conversation is not None
    assert conversation["title"] == title
    assert [(row["role"], row["content"]) for row in rows] == [
        ("user", user_text),
        ("assistant", assistant_text),
    ]

    reopened = client.get(f"/api/conversations/{conversation_id}")
    assert reopened.status_code == 200, reopened.text
    assert [m["content"] for m in reopened.json()["messages"]] == [user_text, assistant_text]


def test_rename_conversation_persists():
    created = client.post("/api/conversations", json={"title": unique_title("before-rename")})
    assert created.status_code == 200, created.text
    conversation_id = created.json()["id"]
    new_title = unique_title("after-rename")

    renamed = client.patch(
        f"/api/conversations/{conversation_id}", json={"title": new_title}
    )
    assert renamed.status_code == 200, renamed.text
    assert renamed.json()["title"] == new_title

    reopened = client.get(f"/api/conversations/{conversation_id}")
    assert reopened.status_code == 200, reopened.text
    assert reopened.json()["title"] == new_title


def test_delete_conversation_removes_its_messages():
    imported = client.post("/api/conversations/import", json={
        "conversation": {
            "title": unique_title("delete-cascade"),
            "messages": [{"role": "user", "content": "delete-cascade-test"}],
        }
    })
    assert imported.status_code == 200, imported.text
    conversation_id = imported.json()["id"]

    deleted = client.delete(f"/api/conversations/{conversation_id}")
    assert deleted.status_code == 200, deleted.text
    assert client.get(f"/api/conversations/{conversation_id}").status_code == 404

    with main.db_connect() as db:
        count = db.execute(
            "SELECT COUNT(*) FROM messages WHERE conversation_id = ?", (conversation_id,)
        ).fetchone()[0]
    assert count == 0


def test_conversation_import_rejects_missing_messages_list():
    response = client.post("/api/conversations/import", json={
        "conversation": {"title": unique_title("invalid-import")}
    })
    assert response.status_code == 422
