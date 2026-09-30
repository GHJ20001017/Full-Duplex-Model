"""Opt-in, best-effort JSONL diagnostics for Chat Completions delegation.

Payloads contain conversation text: use a private directory and remove the file
when done. No files or directories are created unless explicitly enabled.
"""
from __future__ import annotations

import json
import logging
import os
import re
import stat
import time
import uuid
from contextvars import ContextVar
from threading import Lock
from typing import Any

trace_context: ContextVar[dict[str, Any] | None] = ContextVar("delegation_trace_context", default=None)
_LOCK = Lock()
_NOTICE_LOCK = Lock()
_NOTICES: set[str] = set()
logger = logging.getLogger(__name__)


def _notice(status: str) -> None:
    with _NOTICE_LOCK:
        if status in _NOTICES:
            return
        _NOTICES.add(status)
    if status == "written":
        logger.warning("Delegation trace active: writing private diagnostic JSONL (contains conversation text).")
    else:
        logger.warning("Delegation trace write failed; check S2S_DELEGATION_TRACE_PATH and file permissions. Service continues.")


_SECRET_KEY = re.compile(r"api[_-]?key|authorization|credential|password|passwd|secret|access[_-]?token|refresh[_-]?token|cookie", re.I)
_SECRET_TEXT = re.compile(
    r"(?i)(?:bearer\s+[^\s\"'<>]+|sk-[\w-]+|"
    r"(?:api[_-]?key|authorization|password|passwd|secret|credential|access[_-]?token|refresh[_-]?token)"
    r"[\"']?\s*[:=]\s*(?:\"[^\"]*\"|'[^']*'|(?:bearer\s+)?[^\s\"',;}<>]+))"
)
_AUDIO_DATA = re.compile(r"data:audio/[^\s\"'<>]+", re.I)
_MAX_STRING = 65536
_MAX_RECORDS = 4096


def _sanitize(value: Any, secrets: tuple[str, ...], depth: int = 0) -> Any:
    if depth > 24:
        return "[depth limit]"
    if isinstance(value, (bytes, bytearray, memoryview)):
        return "[binary omitted]"
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json", exclude_none=True)
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            key = str(key)
            if _SECRET_KEY.search(key):
                result[key] = "[redacted]"
            elif "audio" in key.lower() and key.lower() not in {"audio_tokens"}:
                result[key] = "[audio omitted]"
            else:
                result[key] = _sanitize(item, secrets, depth + 1)
        return result
    if isinstance(value, (list, tuple)):
        return [_sanitize(item, secrets, depth + 1) for item in value]
    if isinstance(value, str):
        # Parse encoded tool arguments before textual redaction changes JSON.
        if value.lstrip().startswith(("{", "[")):
            try:
                return json.dumps(_sanitize(json.loads(value), secrets, depth + 1), ensure_ascii=False)
            except (ValueError, TypeError):
                pass
        for secret in secrets:
            value = value.replace(secret, "[redacted]")
        value = _AUDIO_DATA.sub("[audio omitted]", value)
        value = re.sub(r"(https?://)[^/@\s]+:[^/@\s]+@", r"\1[redacted]@", value)
        value = _SECRET_TEXT.sub("[redacted]", value)
        return value if len(value) <= _MAX_STRING else value[:_MAX_STRING] + "[truncated]"
    if value is None or isinstance(value, (int, float, bool)):
        return value
    return f"[{type(value).__name__} omitted]"


class DelegationTrace:
    def __init__(self, path: str) -> None:
        self.path = path
        self.request_id = uuid.uuid4().hex
        self.context = dict(trace_context.get() or {})
        self.secrets = tuple(v for k, v in os.environ.items() if _SECRET_KEY.search(k) and len(v) >= 4)
        self.counts: dict[str, int] = {}

    @classmethod
    def from_env(cls) -> DelegationTrace | None:
        try:
            path = os.environ.get("S2S_DELEGATION_TRACE_PATH", "")
            return cls(path) if path and os.path.isabs(path) else None
        except Exception:
            return None

    def protect_client(self, client: Any) -> None:
        try:
            key = getattr(client, "api_key", None)
            if isinstance(key, str) and key:
                self.secrets += (key,)
        except Exception:
            pass

    def emit(self, event: str, payload: Any = None) -> None:
        # Never allow filesystem/serialization errors to affect the provider.
        try:
            count = self.counts.get(event, 0)
            self.counts[event] = count + 1
            if count > _MAX_RECORDS:
                return
            if count == _MAX_RECORDS:
                payload = {"truncated_event": event, "limit": _MAX_RECORDS}
                event = "trace_limit"
            record = _sanitize({
                "timestamp": time.time(), "request_id": self.request_id,
                **self.context, "event": event, "sequence": count, "payload": payload,
            }, self.secrets)
            data = (json.dumps(record, ensure_ascii=False) + "\n").encode("utf-8")
            if len(data) > 262144:
                record["payload"] = {"truncated": True, "original_bytes": len(data),
                                     "preview": json.dumps(record["payload"], ensure_ascii=False)[:32768]}
                data = (json.dumps(record, ensure_ascii=False) + "\n").encode("utf-8")
            with _LOCK:
                fd = os.open(self.path, os.O_WRONLY | os.O_APPEND | os.O_CREAT | os.O_NOFOLLOW | os.O_NONBLOCK, 0o600)
                try:
                    info = os.fstat(fd)
                    if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or info.st_nlink != 1:
                        raise OSError("Unsafe diagnostic file")
                    os.fchmod(fd, 0o600)
                    while data:
                        written = os.write(fd, data)
                        if written <= 0:
                            raise OSError("Incomplete diagnostic write")
                        data = data[written:]
                finally:
                    os.close(fd)
            _notice("written")
        except Exception:
            _notice("failed")


def trace_realtime_event(event: str, payload: Any) -> None:
    """Record wire IDs independently of mutable active-response state."""
    try:
        path = os.environ.get("S2S_DELEGATION_TRACE_PATH", "")
        if not path or not os.path.isabs(path):
            return
        if event == "realtime_outbound":
            kind = payload.get("type", "") if isinstance(payload, dict) else getattr(payload, "type", "")
            if kind.endswith(".delta") or kind in {"response.audio.done", "response.output_audio.done"}:
                return
        trace = DelegationTrace.from_env()
        if trace is not None:
            trace.emit(event, payload)
    except Exception:
        _notice("failed")


class TracedStream:
    """Passive lazy tap; close remains owned by the wrapped stream."""
    def __init__(self, source: Any, trace: DelegationTrace, event: str) -> None:
        self.source = source
        self.trace = trace
        self.event = event
        self.iterator: Any = None

    def __iter__(self) -> TracedStream:
        return self

    def __next__(self) -> Any:
        try:
            if self.iterator is None:
                self.iterator = iter(self.source)
            chunk = next(self.iterator)
        except StopIteration:
            self.trace.emit(self.event + "_end")
            raise
        except BaseException as exc:
            self.trace.emit(self.event + "_error", {"type": type(exc).__name__})
            raise
        self.trace.emit(self.event, chunk)
        return chunk

    def close(self) -> None:
        self.trace.emit(self.event + "_close")
        close = getattr(self.source, "close", None)
        if close is not None:
            close()
