"""
ai/client.py — async LLM client using Groq (OpenAI-compatible API).
"""

from __future__ import annotations

import json
import logging
from typing import Any, Callable, Awaitable

from openai import AsyncOpenAI

from config import settings
from ai.prompts import SYSTEM_PROMPT, build_music_context
from ai.tools import MUSIC_TOOLS_OPENAI

log = logging.getLogger("nova.ai.client")

_client = AsyncOpenAI(
    api_key=settings.llm_api_key,
    base_url=settings.llm_base_url,
)

ToolExecutor = Callable[[str, dict], Awaitable[Any]]


async def process_message(
    conversation: list[dict],
    music_state: dict | None,
    tool_executor: ToolExecutor,
) -> str:
    music_ctx = build_music_context(music_state)
    system = SYSTEM_PROMPT
    if music_ctx:
        system = f"{SYSTEM_PROMPT}\n\n{music_ctx}"

    messages = [{"role": "system", "content": system}] + list(conversation)

    for attempt in range(6):
        log.debug("LLM request attempt %d", attempt + 1)

        response = await _client.chat.completions.create(
            model=settings.llm_model,
            max_tokens=512,
            tools=MUSIC_TOOLS_OPENAI,
            tool_choice="auto",
            messages=messages,
        )

        choice = response.choices[0]
        msg = choice.message

        # Pure text response
        if choice.finish_reason == "stop" or not msg.tool_calls:
            return msg.content or ""

        # Tool calls requested
        messages.append(msg)  # append assistant message with tool_calls

        for tool_call in msg.tool_calls:
            tool_name = tool_call.function.name
            try:
                tool_input = json.loads(tool_call.function.arguments)
            except json.JSONDecodeError:
                tool_input = {}

            log.info("[AI] Tool call: %s %s", tool_name, tool_input)

            try:
                result = await tool_executor(tool_name, tool_input)
                result_content = json.dumps(result)
            except Exception as exc:
                log.error("Tool %s raised an exception", tool_name, exc_info=True)
                result_content = json.dumps({"error": str(exc)})
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result_content,
                }
            )

    return "I got stuck in a loop. Try again?"
