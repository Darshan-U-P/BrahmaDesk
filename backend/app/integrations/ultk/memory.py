"""BrahmaDesk-backed memory tools; BrahmaDesk remains the memory system of record."""

from __future__ import annotations

import time
import uuid
from collections.abc import Callable
from typing import Any

from ultk.exceptions import ToolExecutionError
from ultk.models import ToolContext, ToolSpec


class BrahmaDeskMemoryTool:
    def __init__(
        self,
        spec: ToolSpec,
        handler: Callable[[dict[str, Any], ToolContext], Any],
    ) -> None:
        self.spec = spec
        self._handler = handler

    async def invoke(
        self,
        arguments: dict[str, Any],
        context: ToolContext,
    ) -> Any:
        return self._handler(arguments, context)


def create_brahmadesk_memory_tools(
    db_connect: Callable[[], Any],
    save_global_memory: Callable[[str, str, str], dict[str, Any]],
) -> tuple[BrahmaDeskMemoryTool, ...]:
    """Build profile-scoped tools over BrahmaDesk's existing memories table."""

    def profile_id_from(context: ToolContext) -> str:
        profile_id = context.metadata.get("profile_id")
        if not isinstance(profile_id, str) or not profile_id:
            raise ToolExecutionError("A validated BrahmaDesk profile context is required.")
        return profile_id

    def project_id_from(context: ToolContext, db: Any) -> str | None:
        project_id = context.metadata.get("project_id")
        if project_id is None:
            return None
        if not isinstance(project_id, str) or not project_id:
            raise ToolExecutionError("The host supplied an invalid project context.")
        project = db.execute(
            "SELECT id FROM projects WHERE id = ? AND profile_id = ?",
            (project_id, profile_id_from(context)),
        ).fetchone()
        if project is None:
            raise ToolExecutionError("The selected project does not belong to this profile.")
        return project_id

    def search(arguments: dict[str, Any], context: ToolContext) -> dict[str, Any]:
        profile_id = profile_id_from(context)
        query = str(arguments.get("query", "")).strip()
        limit = int(arguments.get("limit", 10))
        if len(query) > 200 or not 1 <= limit <= 50:
            raise ToolExecutionError("Memory search query or result limit is outside allowed bounds.")
        with db_connect() as db:
            project_id = project_id_from(context, db)
            rows = db.execute(
                """
                SELECT id, category, content, source, status, created_at, updated_at
                FROM memories
                WHERE profile_id = ?
                  AND status = 'active'
                  AND (project_id IS NULL OR project_id = ?)
                  AND (? = '' OR content LIKE ? OR category LIKE ?)
                ORDER BY updated_at DESC
                LIMIT ?
                """,
                (
                    profile_id,
                    project_id,
                    query,
                    f"%{query}%",
                    f"%{query}%",
                    limit,
                ),
            ).fetchall()
        return {"memories": [dict(row) for row in rows], "count": len(rows)}

    def save(arguments: dict[str, Any], context: ToolContext) -> dict[str, Any]:
        profile_id = profile_id_from(context)
        content = str(arguments.get("content", "")).strip()
        category = str(arguments.get("category", "general")).strip() or "general"
        if not content or len(content) > 4000 or len(category) > 50:
            raise ToolExecutionError("Memory content or category exceeds allowed bounds.")
        with db_connect() as db:
            project_id = project_id_from(context, db)
            if project_id is None:
                existing = None
            else:
                existing = db.execute(
                """
                SELECT id FROM memories
                WHERE profile_id = ? AND project_id = ? AND status = 'active'
                  AND LOWER(TRIM(content)) = LOWER(TRIM(?))
                LIMIT 1
                    """,
                    (profile_id, project_id, content),
                ).fetchone()
            if project_id is None:
                return save_global_memory(profile_id, content, category)
            now = time.time()
            if existing:
                db.execute(
                    "UPDATE memories SET category = ?, updated_at = ?, last_used_at = ? WHERE id = ?",
                    (category, now, now, existing["id"]),
                )
                memory_id = existing["id"]
                status = "updated"
            else:
                memory_id = str(uuid.uuid4())
                db.execute(
                    """
                    INSERT INTO memories(
                        id, profile_id, project_id, category, content, source,
                        status, created_at, updated_at, last_used_at
                    ) VALUES (?, ?, ?, ?, ?, 'agent', 'active', ?, ?, NULL)
                    """,
                    (memory_id, profile_id, project_id, category, content, now, now),
                )
                status = "created"
            row = db.execute(
                """
                SELECT id, category, content, source, status, created_at, updated_at
                FROM memories WHERE id = ? AND profile_id = ?
                """,
                (memory_id, profile_id),
            ).fetchone()
        return {"status": status, "memory": dict(row)}

    def archive(arguments: dict[str, Any], context: ToolContext) -> dict[str, Any]:
        profile_id = profile_id_from(context)
        memory_id = str(arguments.get("memory_id", "")).strip()
        if not memory_id or len(memory_id) > 128:
            raise ToolExecutionError("A valid memory identifier is required.")
        with db_connect() as db:
            project_id = project_id_from(context, db)
            cursor = db.execute(
                """
                UPDATE memories SET status = 'archived', updated_at = ?
                WHERE id = ? AND profile_id = ? AND status = 'active'
                  AND ((project_id IS NULL AND ? IS NULL) OR project_id = ?)
                """,
                (time.time(), memory_id, profile_id, project_id, project_id),
            )
            if cursor.rowcount != 1:
                raise ToolExecutionError("The memory does not exist in the active profile/project scope.")
        return {"status": "archived", "memory_id": memory_id}

    return (
        BrahmaDeskMemoryTool(
            ToolSpec(
                name="memory.search",
                description="Search active memories belonging only to the current BrahmaDesk profile and selected project. Use this to recall relevant saved context.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "maxLength": 200},
                        "limit": {"type": "integer", "minimum": 1, "maximum": 50},
                    },
                    "additionalProperties": False,
                },
                capabilities=frozenset({"memory:read"}),
                timeout_seconds=5,
                max_output_chars=12000,
            ),
            search,
        ),
        BrahmaDeskMemoryTool(
            ToolSpec(
                name="memory.save",
                description="Save a durable user memory only when the user explicitly asks you to remember or save this information. Saves within the current profile and selected project.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "content": {"type": "string", "minLength": 1, "maxLength": 4000},
                        "category": {"type": "string", "maxLength": 50},
                    },
                    "required": ["content"],
                    "additionalProperties": False,
                },
                capabilities=frozenset({"memory:write"}),
                timeout_seconds=5,
                max_output_chars=3000,
            ),
            save,
        ),
        BrahmaDeskMemoryTool(
            ToolSpec(
                name="memory.archive",
                description="Archive a saved memory in the current BrahmaDesk profile and selected project. Only do this when the user explicitly asks to remove or archive it.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "memory_id": {"type": "string", "minLength": 1, "maxLength": 128},
                    },
                    "required": ["memory_id"],
                    "additionalProperties": False,
                },
                capabilities=frozenset({"memory:write"}),
                timeout_seconds=5,
                max_output_chars=1000,
            ),
            archive,
        ),
    )
