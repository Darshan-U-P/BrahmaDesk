"""Regression tests for explicit, profile-scoped persistent memory."""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any

# Keep test data separate from the user's real %APPDATA% database.
_TEST_DATA_DIR = tempfile.mkdtemp(prefix="brahma-desk-tests-")
os.environ["BRAHMA_DATA_DIR"] = _TEST_DATA_DIR
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import main  # noqa: E402


class FakeUpstream:
    status_code = 200

    async def __aenter__(self) -> "FakeUpstream":
        return self

    async def __aexit__(self, *args: Any) -> None:
        return None

    async def aread(self) -> bytes:
        return b""

    async def aiter_lines(self):
        yield 'data: ' + json.dumps({"choices": [{"delta": {"content": "Memory handled."}}]})
        yield "data: [DONE]"


class FakeAsyncClient:
    last_payload: dict[str, Any] | None = None

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    async def __aenter__(self) -> "FakeAsyncClient":
        return self

    async def __aexit__(self, *args: Any) -> None:
        return None

    def stream(self, method: str, url: str, *, json: dict[str, Any], **kwargs: Any) -> FakeUpstream:
        type(self).last_payload = json
        return FakeUpstream()


class GlobalMemoryTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        with main.db_connect() as db:
            db.execute("DELETE FROM memories")
            db.execute("DELETE FROM messages")
            db.execute("DELETE FROM conversations")
        self.original_async_client = main.httpx.AsyncClient
        main.httpx.AsyncClient = FakeAsyncClient
        FakeAsyncClient.last_payload = None

    def tearDown(self) -> None:
        main.httpx.AsyncClient = self.original_async_client

    def active_profile_id(self) -> str:
        with main.db_connect() as db:
            return main.get_active_profile_record(db)["id"]

    def memory_rows(self) -> list[dict[str, Any]]:
        with main.db_connect() as db:
            return [dict(row) for row in db.execute("SELECT * FROM memories").fetchall()]

    def test_duplicate_memory_updates_existing_record(self) -> None:
        profile_id = self.active_profile_id()
        content = "My current AI project is called Project Orion."
        first = main.save_global_memory(profile_id, content, "project")
        second = main.save_global_memory(profile_id, content, "project")

        self.assertEqual(first["status"], "created")
        self.assertEqual(second["status"], "updated")
        self.assertEqual(first["memory"]["id"], second["memory"]["id"])
        self.assertEqual(len(self.memory_rows()), 1)

    def test_regular_chat_message_does_not_create_memory(self) -> None:
        parsed = main.extract_explicit_memory_request("I am working on an AI project called Orion.")
        self.assertIsNone(parsed)

    def test_memory_survives_reopening_database_connection(self) -> None:
        profile_id = self.active_profile_id()
        saved = main.save_global_memory(profile_id, "Persistent memory survives restarts.", "general")
        # A new SQLite connection reads the same committed record.
        with main.db_connect() as db:
            row = db.execute("SELECT content FROM memories WHERE id = ?", (saved["memory"]["id"],)).fetchone()
        self.assertIsNotNone(row)
        self.assertEqual(row["content"], "Persistent memory survives restarts.")

    def test_chat_only_saves_when_user_explicitly_opts_in(self) -> None:
        request = main.ChatRequest(messages=[main.Message(role="user", content="I am building a local AI assistant.")])

        async def run_request() -> str:
            response = await main.chat_stream(request)
            return "".join([chunk async for chunk in response.body_iterator])

        stream = self.run_async(run_request())
        self.assertIn('"done": true', stream)
        self.assertEqual(self.memory_rows(), [])

    def test_explicit_chat_instruction_saves_memory_before_model_context(self) -> None:
        prompt = (
            "Remember globally that my current AI project is called Project Orion. "
            "Its goal is to build a local-first AI assistant. Save this as persistent "
            "memory, not just as part of this conversation."
        )
        request = main.ChatRequest(messages=[main.Message(role="user", content=prompt)])

        async def run_request() -> str:
            response = await main.chat_stream(request)
            return "".join([chunk async for chunk in response.body_iterator])

        stream = self.run_async(run_request())
        self.assertIn('"done": true', stream)
        rows = self.memory_rows()
        self.assertEqual(len(rows), 1)
        self.assertIn("Project Orion", rows[0]["content"])
        self.assertEqual(rows[0]["category"], "project")
        payload = FakeAsyncClient.last_payload
        self.assertIsNotNone(payload)
        assert payload is not None
        system_text = "\n".join(m["content"] for m in payload["messages"] if m["role"] == "system")
        self.assertIn("successfully saved", system_text)
        self.assertIn("Project Orion", system_text)

    @staticmethod
    def run_async(coro):
        # IsolatedAsyncioTestCase already runs in an event loop, so tests call this helper
        # only from the synchronous wrapper in this test method.
        import asyncio
        return asyncio.run(coro)


if __name__ == "__main__":
    unittest.main()


class ReasoningBudgetTests(unittest.TestCase):
    def test_reasoning_budget_values_are_validated(self) -> None:
        for budget in ("off", "low", "medium", "high", "max"):
            request = main.ChatRequest(
                messages=[main.Message(role="user", content="Hello")],
                reasoning_budget=budget,
            )
            self.assertEqual(request.reasoning_budget, budget)

    def test_invalid_reasoning_budget_is_rejected(self) -> None:
        from pydantic import ValidationError
        with self.assertRaises(ValidationError):
            main.ChatRequest(
                messages=[main.Message(role="user", content="Hello")],
                reasoning_budget="extreme",
            )
