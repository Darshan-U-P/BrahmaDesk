import importlib
import json
import os
import sys
import uuid
from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx
from fastapi.testclient import TestClient

BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))
os.environ.setdefault(
    "BRAHMA_DATA_DIR",
    str(Path(__file__).resolve().parent / ".test-data"),
)

main = importlib.import_module("main")
client = TestClient(main.app)


def make_request():
    return {
        "messages": [
            {"role": "user", "content": f"stream-failure-{uuid.uuid4().hex}"}
        ]
    }


def read_sse(response):
    events = []
    for line in response.text.splitlines():
        if line.startswith("data: "):
            value = line[6:]
            if value == "[DONE]":
                continue
            try:
                events.append(json.loads(value))
            except json.JSONDecodeError:
                pass
    return events


def configure_client_mock(mock_client, stream_context):
    instance = MagicMock()
    instance.__aenter__.return_value = instance
    instance.__aexit__.return_value = False
    instance.stream.return_value = stream_context
    mock_client.return_value = instance


def test_stream_reports_upstream_http_error():
    user_text = f"http-error-chat-{uuid.uuid4().hex}"

    class MockUpstream:
        status_code = 503

        async def aread(self):
            return b"simulated model outage"

    class MockStream:
        async def __aenter__(self):
            return MockUpstream()

        async def __aexit__(self, *args):
            return False

    request = {"messages": [{"role": "user", "content": user_text}]}
    with patch.object(main.httpx, "AsyncClient") as mock_client:
        configure_client_mock(mock_client, MockStream())
        response = client.post("/api/chat/stream", json=request)

    assert response.status_code == 200
    events = read_sse(response)
    assert any("error" in event for event in events)
    assert "[DONE]" in response.text
    with main.db_connect() as db:
        rows = db.execute(
            "SELECT id FROM conversations WHERE title = ?",
            (user_text[:60],),
        ).fetchall()
    assert rows == []


def test_stream_reports_connection_failure():
    class MockStream:
        async def __aenter__(self):
            raise httpx.ConnectError("simulated connection failure")

        async def __aexit__(self, *args):
            return False

    with patch.object(main.httpx, "AsyncClient") as mock_client:
        configure_client_mock(mock_client, MockStream())
        response = client.post("/api/chat/stream", json=make_request())

    assert response.status_code == 200
    events = read_sse(response)
    assert any(
        "error" in event
        and "Cannot reach llama.cpp server" in event["error"]
        for event in events
    )
    assert "[DONE]" in response.text


def test_connection_failure_does_not_create_conversation():
    user_text = f"failed-chat-{uuid.uuid4().hex}"
    request = {"messages": [{"role": "user", "content": user_text}]}

    class MockStream:
        async def __aenter__(self):
            raise httpx.ConnectError("offline")

        async def __aexit__(self, *args):
            return False

    with patch.object(main.httpx, "AsyncClient") as mock_client:
        configure_client_mock(mock_client, MockStream())
        response = client.post("/api/chat/stream", json=request)

    assert response.status_code == 200

    with main.db_connect() as db:
        rows = db.execute(
            "SELECT id FROM conversations WHERE title = ?",
            (user_text[:60],),
        ).fetchall()

    assert rows == []


def test_unexpected_stream_end_does_not_save_partial_answer():
    user_text = f"interrupted-chat-{uuid.uuid4().hex}"
    request = {"messages": [{"role": "user", "content": user_text}]}

    class MockUpstream:
        status_code = 200

        async def aiter_lines(self):
            yield 'data: {"choices":[{"delta":{"content":"partial answer"}}]}'
            # Simulate an upstream stream ending without data: [DONE].

    class MockStream:
        async def __aenter__(self):
            return MockUpstream()

        async def __aexit__(self, *args):
            return False

    with patch.object(main.httpx, "AsyncClient") as mock_client:
        configure_client_mock(mock_client, MockStream())
        response = client.post("/api/chat/stream", json=request)

    assert response.status_code == 200
    with main.db_connect() as db:
        rows = db.execute(
            "SELECT c.id, m.role, m.content "
            "FROM conversations c "
            "LEFT JOIN messages m ON m.conversation_id = c.id "
            "WHERE c.title = ?",
            (user_text[:60],),
        ).fetchall()
    assert rows == []


def test_read_error_during_stream_reports_error_and_does_not_persist():
    user_text = f"read-error-chat-{uuid.uuid4().hex}"
    request = {"messages": [{"role": "user", "content": user_text}]}

    class MockUpstream:
        status_code = 200

        async def aiter_lines(self):
            yield 'data: {"choices":[{"delta":{"content":"partial answer"}}]}'
            raise httpx.ReadError("simulated interrupted stream")

    class MockStream:
        async def __aenter__(self):
            return MockUpstream()

        async def __aexit__(self, *args):
            return False

    with patch.object(main.httpx, "AsyncClient") as mock_client:
        configure_client_mock(mock_client, MockStream())
        response = client.post("/api/chat/stream", json=request)

    assert response.status_code == 200
    events = read_sse(response)
    assert any(
        "error" in event
        and "Cannot reach llama.cpp server" in event["error"]
        for event in events
    )
    assert "[DONE]" in response.text
    with main.db_connect() as db:
        rows = db.execute(
            "SELECT id FROM conversations WHERE title = ?",
            (user_text[:60],),
        ).fetchall()
    assert rows == []
