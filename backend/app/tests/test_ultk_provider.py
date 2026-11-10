from __future__ import annotations

import asyncio

from app.engine.contracts import GenerationChunk, GenerationRequest, GenerationToolCall
from app.integrations.ultk.provider import BrahmaEngineProvider


class FakeSupervisor:
    def __init__(self, chunks):
        self.chunks = chunks
        self.request = None

    async def generate(self, request: GenerationRequest):
        self.request = request
        for chunk in self.chunks:
            yield chunk


def test_brahma_provider_maps_tool_calls_and_preserves_messages():
    supervisor = FakeSupervisor([
        GenerationChunk(
            text="",
            finished=True,
            finish_reason="tool_calls",
            tool_calls=(
                GenerationToolCall(
                    id="call-1",
                    name="system.get_platform_info",
                    arguments={},
                ),
            ),
        )
    ])
    provider = BrahmaEngineProvider(supervisor)

    async def run():
        return await provider.complete(
            [
                {"role": "system", "content": "Be careful."},
                {"role": "user", "content": "Inspect this machine."},
            ],
            [
                {
                    "type": "function",
                    "function": {
                        "name": "system.get_platform_info",
                        "description": "Inspect platform details",
                        "parameters": {"type": "object", "properties": {}},
                    },
                }
            ],
        )

    turn = asyncio.run(run())

    assert turn.content is None
    assert turn.tool_calls[0].id == "call-1"
    assert turn.tool_calls[0].name == "system.get_platform_info"
    assert turn.tool_calls[0].arguments == {}
    assert supervisor.request.stream is False
    assert supervisor.request.messages[0]["role"] == "system"
    assert supervisor.request.tools[0]["function"]["name"] == "system.get_platform_info"


def test_brahma_provider_maps_final_text():
    supervisor = FakeSupervisor([
        GenerationChunk(text="Local answer.", finished=True, finish_reason="stop")
    ])
    provider = BrahmaEngineProvider(supervisor)

    async def run():
        return await provider.complete(
            [{"role": "user", "content": "Hello"}],
            [],
        )

    turn = asyncio.run(run())
    assert turn.content == "Local answer."
    assert turn.tool_calls == ()
