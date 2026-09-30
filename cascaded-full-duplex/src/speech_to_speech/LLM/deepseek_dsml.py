"""Conservative wire-level fallback for DSML and XML-style tool containers.

Only a container at the beginning of content is executable; quoted/embedded
examples remain text. Marked responses are validated in full before any call
is released. DSML string values are raw text, not XML entities.

Streaming responses are wrapped in :class:`_NormalizedStream`, which forwards
``close`` (idempotently) to the underlying HTTP response even when it was never
iterated or is closed concurrently. All normalization state is per-wrapper, so
nothing is shared on the handler.
"""

from __future__ import annotations

import copy
import json
import re
from collections.abc import Iterator
from threading import Lock
from typing import Any

from openai.types.chat.chat_completion_chunk import ChoiceDeltaToolCall
from openai.types.chat.chat_completion_message_tool_call import ChatCompletionMessageFunctionToolCall

_XML_MARKER = "<tool_call>"
_XML_FUNCTION = re.compile(r"<function=([^<>\s]+)>")
_XML_PARAMETER = re.compile(r"<parameter=([^<>\s]+)>")
_MARKER = "<｜DSML｜"
# A lone "<" is ordinary text; "<｜" (or a longer marker prefix) is identifiable.
_MARKER_PREFIX_MIN = len("<｜")
_CONTAINER = re.compile(r"<｜DSML｜(tool_calls|function_calls)>")
_INVOKE = re.compile(r'<｜DSML｜invoke name="([^"<>]+)">')
_PARAMETER = re.compile(r'<｜DSML｜parameter name="([^"<>]+)" string="(true|false)">')


def _candidate(text: str) -> bool:
    """Whether a streaming prefix could still grow into a DSML container.

    A lone ``<`` is a candidate (it may begin a marker split across chunks) but
    is not *identified* as DSML here; :func:`_looks_like_dsml` decides at EOF so
    ordinary ``<``/``whitespace + <`` text is preserved rather than rejected.
    """
    text = text.lstrip()
    return not text or any(marker.startswith(text) or text.startswith(marker) for marker in (_MARKER, _XML_MARKER))


def _looks_like_dsml(text: str) -> bool:
    """True for an identifiable DSML or XML tool prefix, including truncation."""
    text = text.strip()
    return (
        text.startswith(_MARKER)
        or (len(text) >= _MARKER_PREFIX_MIN and _MARKER.startswith(text))
        or text.startswith(_XML_MARKER)
        or (len(text) >= len("<tool_") and _XML_MARKER.startswith(text))
    )


def _invalid() -> None:
    raise ValueError("Invalid, truncated, or unauthorized DeepSeek DSML tool response")


def _parse_xml_calls(text: str, options: dict[str, Any], allowed: set[str]) -> list[dict[str, Any]]:
    schemas = {tool["function"]["name"]: tool["function"].get("parameters", {})
               for tool in options.get("tools", []) if tool.get("type") == "function"}
    calls = []
    while text:
        if not text.startswith(_XML_MARKER):
            _invalid()
        body = text[len(_XML_MARKER):].lstrip()
        function = _XML_FUNCTION.match(body)
        if not function or function[1] not in allowed:
            _invalid()
        name = function[1]
        properties = schemas[name].get("properties", {})
        body = body[function.end():].lstrip()
        arguments = {}
        while not body.startswith("</function>"):
            parameter = _XML_PARAMETER.match(body)
            if not parameter or parameter[1] in arguments:
                _invalid()
            key = parameter[1]
            value, separator, body = body[parameter.end():].partition("</parameter>")
            if not separator or any(tag in value for tag in ("<tool_call", "<function=", "<parameter=", "</function>", "</tool_call>")):
                _invalid()
            # This format has no DSML string flag. Only the advertised schema
            # can distinguish a string such as "123" from a JSON number.
            value = value.strip()
            kind = properties.get(key, {}).get("type")
            if kind != "string":
                if kind not in ("integer", "number", "boolean", "null", "array", "object"):
                    _invalid()
                try:
                    value = json.loads(value)
                    json.dumps(value, allow_nan=False)
                except (ValueError, TypeError):
                    _invalid()
                valid = {
                    "integer": type(value) is int,
                    "number": type(value) in (int, float),
                    "boolean": type(value) is bool,
                    "null": value is None,
                    "array": isinstance(value, list),
                    "object": isinstance(value, dict),
                }
                if not valid[kind]:
                    _invalid()
            arguments[key] = value
            body = body.lstrip()
        if not set(schemas[name].get("required", [])).issubset(arguments):
            _invalid()
        body = body[len("</function>"):].lstrip()
        if not body.startswith("</tool_call>"):
            _invalid()
        text = body[len("</tool_call>"):].lstrip()
        calls.append({"type": "function", "function": {"name": name, "arguments": json.dumps(arguments, ensure_ascii=False)}})
    return calls


def _parse(text: str, options: dict[str, Any]) -> list[dict[str, Any]]:
    allowed = {
        tool["function"]["name"]
        for tool in options.get("tools", [])
        if tool.get("type") == "function"
    }
    choice = options.get("tool_choice")
    if choice == "none":
        allowed.clear()
    elif isinstance(choice, dict):
        allowed.intersection_update([choice.get("function", {}).get("name")])
    text = text.strip()
    if text.startswith(_XML_MARKER):
        return _parse_xml_calls(text, options, allowed)
    match = _CONTAINER.match(text)
    if not match or not allowed:
        _invalid()
    end = f"</｜DSML｜{match[1]}>"
    if not text.endswith(end):
        _invalid()
    body = text[match.end():-len(end)].strip()
    calls = []
    while body:
        invoke = _INVOKE.match(body)
        if not invoke or invoke[1] not in allowed:
            _invalid()
        name = invoke[1]
        body = body[invoke.end():].lstrip()
        arguments = {}
        while not body.startswith("</｜DSML｜invoke>"):
            parameter = _PARAMETER.match(body)
            if not parameter or parameter[1] in arguments:
                _invalid()
            body = body[parameter.end():]
            value, separator, body = body.partition("</｜DSML｜parameter>")
            if not separator or "｜DSML｜" in value:
                _invalid()
            if parameter[2] == "false":
                try:
                    value = json.loads(value)
                    # Reject non-JSON NaN/Infinity accepted by Python's decoder.
                    json.dumps(value, allow_nan=False)
                except (ValueError, TypeError):
                    _invalid()
            arguments[parameter[1]] = value
            body = body.lstrip()
        body = body[len("</｜DSML｜invoke>"):].lstrip()
        calls.append({"type": "function", "function": {"name": name, "arguments": json.dumps(arguments, ensure_ascii=False)}})
    if not calls:
        _invalid()
    return calls


def _replace(obj: Any, **updates: Any) -> Any:
    result = copy.copy(obj)
    for key, value in updates.items():
        setattr(result, key, value)
    return result


def normalize_response(response: Any, options: dict[str, Any]) -> Any:
    if not response.choices:
        return response
    message = response.choices[0].message
    text = message.content or ""
    if not _looks_like_dsml(text):
        return response
    if message.tool_calls or getattr(message, "refusal", None):
        _invalid()
    calls = [ChatCompletionMessageFunctionToolCall(id=f"dsml_{i}", **call) for i, call in enumerate(_parse(text, options))]
    choice = _replace(response.choices[0], message=_replace(message, content=None, tool_calls=calls))
    return _replace(response, choices=[choice, *response.choices[1:]])


def _normalize_chunks(response: Any, options: dict[str, Any]) -> Iterator[Any]:
    """Buffer the leading content until it is known text or a DSML container."""
    pending = []
    text = ""
    passthrough = False
    for chunk in response:
        if passthrough:
            yield chunk
            continue
        pending.append(chunk)
        if chunk.choices:
            text += chunk.choices[0].delta.content or ""
        if not _candidate(text):
            passthrough = True
            yield from pending
            pending.clear()
    if passthrough:
        return
    if not _looks_like_dsml(text):
        # Ordinary text (including a lone "<") or a native/refusal-only turn.
        yield from pending
        return
    # Do not release native calls before ruling out a mixed/ambiguous turn.
    for chunk in pending:
        if chunk.choices:
            delta = chunk.choices[0].delta
            if delta.tool_calls or getattr(delta, "refusal", None):
                _invalid()
    calls = [ChoiceDeltaToolCall(index=i, id=f"dsml_{i}", **call) for i, call in enumerate(_parse(text, options))]
    last = max(i for i, chunk in enumerate(pending) if chunk.choices)
    for i, chunk in enumerate(pending):
        if chunk.choices:
            delta = _replace(chunk.choices[0].delta, content=None, tool_calls=calls if i == last else None)
            choice = _replace(chunk.choices[0], delta=delta)
            chunk = _replace(chunk, choices=[choice, *chunk.choices[1:]])
        yield chunk


class _NormalizedStream:
    """Iterator over a normalized chat-completions stream.

    ``close`` forwards directly to the underlying HTTP response even when this
    iterator was never started (a bare generator would not run its ``finally``
    in that case), and it is idempotent and safe under concurrent cancellation.
    """

    def __init__(self, response: Any, options: dict[str, Any]) -> None:
        self._response = response
        self._generator = _normalize_chunks(response, options)
        self._close_lock = Lock()
        self._closed = False

    def __iter__(self) -> _NormalizedStream:
        return self

    def __next__(self) -> Any:
        try:
            return next(self._generator)
        except BaseException:
            self._close_response()
            raise

    def close(self) -> None:
        try:
            self._generator.close()
        except ValueError:
            # The generator is executing on another thread (concurrent
            # cancellation); the underlying stream is still closed below.
            pass
        self._close_response()

    def _close_response(self) -> None:
        with self._close_lock:
            if self._closed:
                return
            self._closed = True
        close = getattr(self._response, "close", None)
        if close is not None:
            close()


def normalize_stream(response: Any, options: dict[str, Any]) -> _NormalizedStream:
    """Wrap a raw streaming response in a close-forwarding normalizing iterator."""
    return _NormalizedStream(response, options)
