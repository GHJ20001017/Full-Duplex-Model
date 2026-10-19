"""Normalize standalone DSML and stop protocol text before sentence splitting/TTS.

This operates on provider events, so Responses, Chat and audio fallback share
one speech boundary. Only potential marker prefixes are delayed in normal text;
standalone DSML is buffered until EOF and validated atomically. Native calls
keep their existing streaming behavior; mixed DSML cannot add fallback calls.
"""
from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import TYPE_CHECKING, Any

from openai.types.responses import ResponseFunctionToolCall

from speech_to_speech.LLM.deepseek_dsml import _candidate, _invalid, _looks_like_dsml, _parse
from speech_to_speech.utils.utils import _generate_id

if TYPE_CHECKING:
    from speech_to_speech.LLM.base_openai_compatible_language_model import ProviderEvent

_MARKERS = ("<DSML", "</DSML", "<｜DSML｜", "</｜DSML｜")
_MAX_BUFFER = 65536


def _safe_prefix(text: str, *, final: bool = False) -> tuple[str, str]:
    """Withhold split marker prefixes; reject markers even in quotes/examples."""
    if any(marker in text for marker in _MARKERS):
        _invalid()
    for size in range(min(len(text), max(map(len, _MARKERS)) - 1), 0, -1):
        suffix = text[-size:]
        if any(marker.startswith(suffix) for marker in _MARKERS):
            if final:
                # A bare '<' or '</' is normal text, identifiable prefixes aren't.
                if suffix not in ("<", "</"):
                    _invalid()
                break
            return text[:-size], suffix
    return text, ""


def normalize_speech_events(
    events: Iterator[ProviderEvent],
    options: dict[str, Any],
    *,
    cancelled: Callable[[], bool] = lambda: False,
) -> Iterator[ProviderEvent]:
    # Local import avoids a cycle with the base handler's event definitions.
    from speech_to_speech.LLM.base_openai_compatible_language_model import (
        AssistantMessage,
        TextDelta,
        ToolCall,
        Usage,
    )

    leading = True
    text = ""
    pending_messages = []
    native_seen = False
    for event in events:
        if isinstance(event, Usage):
            # Account for provider tokens even after a cancelled generation.
            yield event
            continue
        if cancelled():
            return
        if isinstance(event, ToolCall):
            if _looks_like_dsml(text) or any(
                _looks_like_dsml("".join(c.text or "" for c in msg.content)) for msg in pending_messages
            ):
                _invalid()
            native_seen = True
            yield event
        elif isinstance(event, AssistantMessage):
            raw = "".join(c.text or "" for c in event.content)
            if leading:
                pending_messages.append(event)
                if _candidate(raw) and sum(
                    len(c.text or "") for m in pending_messages for c in m.content
                ) > _MAX_BUFFER:
                    _invalid()
            else:
                _safe_prefix(raw, final=True)
                yield event
        elif isinstance(event, TextDelta):
            text += event.text
            if leading and _candidate(text):
                if len(text) > _MAX_BUFFER:
                    _invalid()
                continue
            leading = False
            safe, text = _safe_prefix(text)
            for message in pending_messages:
                _safe_prefix("".join(c.text or "" for c in message.content), final=True)
                yield message
            pending_messages.clear()
            if safe:
                yield TextDelta(text=safe)

    if cancelled():
        return
    if leading and _looks_like_dsml(text):
        if native_seen:
            _invalid()
        for message in pending_messages:
            raw = "".join(c.text or "" for c in message.content)
            if raw.strip() != text.strip() or any(c.type != "output_text" for c in message.content):
                _invalid()
        # Parse and validate every call before releasing the first one.
        calls = _parse(text, options)
        reasoning = "".join(message.reasoning_content for message in pending_messages)
        for call in calls:
            yield ToolCall(item=ResponseFunctionToolCall(
                type="function_call", id=_generate_id("fc"), call_id=_generate_id("call"),
                **call["function"],
            ), reasoning_content=reasoning)
        return
    safe, _ = _safe_prefix(text, final=True)
    for message in pending_messages:
        _safe_prefix("".join(c.text or "" for c in message.content), final=True)
        yield message
    if safe:
        yield TextDelta(text=safe)
