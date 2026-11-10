"""Provider adapter that routes ULTK turns through the Brahma Engine."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from app.engine.contracts import GenerationRequest
from app.engine.supervisor import EngineSupervisor


class BrahmaEngineProvider:
    """Adapt Brahma Engine generation to ULTK's provider-neutral contract.

    This adapter intentionally uses non-streaming structured completions for agent
    turns so tool-call arguments are complete and validated before dispatch. The
    ordinary BrahmaDesk chat endpoint continues to use its existing SSE path.
    """

    def __init__(
        self,
        supervisor: EngineSupervisor,
        *,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        generation_options: Mapping[str, Any] | None = None,
    ) -> None:
        if not 1 <= max_tokens <= 131072:
            raise ValueError("max_tokens must be between 1 and 131072")
        if not 0.0 <= temperature <= 2.0:
            raise ValueError("temperature must be between 0 and 2")
        self._supervisor = supervisor
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._generation_options = dict(generation_options or {})

    async def complete(
        self,
        messages: Sequence[Mapping[str, object]],
        tools: Sequence[Mapping[str, object]],
    ):
        """Return one structured assistant turn from the active local model."""
        from ultk.models import AssistantTurn, ToolCallRequest

        normalized_messages = [dict(message) for message in messages]
        latest_user = next(
            (
                message.get("content")
                for message in reversed(normalized_messages)
                if message.get("role") == "user"
            ),
            None,
        )
        if not isinstance(latest_user, str) or not latest_user.strip():
            raise ValueError("The agent conversation must contain a non-empty user message")

        request = GenerationRequest(
            prompt=latest_user,
            max_tokens=self._max_tokens,
            temperature=self._temperature,
            options=dict(self._generation_options),
            messages=normalized_messages,
            stream=False,
            tools=[dict(tool) for tool in tools] if tools else None,
        )
        chunks = [chunk async for chunk in self._supervisor.generate(request)]
        completed = next((chunk for chunk in reversed(chunks) if chunk.finished), None)
        if completed is None:
            raise RuntimeError("Brahma Engine ended the agent turn without a completion marker")

        tool_calls = tuple(
            ToolCallRequest(
                id=call.id,
                name=call.name,
                arguments=dict(call.arguments),
            )
            for call in completed.tool_calls
        )
        content = completed.text or None
        if not content and not tool_calls:
            raise RuntimeError("Brahma Engine returned neither assistant text nor tool calls")
        return AssistantTurn(content=content, tool_calls=tool_calls)
