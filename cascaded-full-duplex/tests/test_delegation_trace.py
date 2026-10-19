"""Delegation diagnostics are passive, private, and opt-in."""
import json
import os
from types import SimpleNamespace as NS

import pytest

from speech_to_speech.LLM.chat_completions_language_model import _request_chat_completions
from speech_to_speech.LLM.delegation_trace import DelegationTrace, trace_context
from tests.test_chat_completions_backend import _drive, _FakeStream, _make_handler
from tests.test_deepseek_dsml import DSML, TOOLS, chunk, request


def records(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


@pytest.mark.parametrize("path", [None, "relative.jsonl"])
def test_disabled(monkeypatch, path):
    monkeypatch.delenv("S2S_DELEGATION_TRACE_PATH", raising=False)
    if path:
        monkeypatch.setenv("S2S_DELEGATION_TRACE_PATH", path)
    monkeypatch.setattr(os, "open", lambda *a, **k: pytest.fail("must not open any file"))
    assert DelegationTrace.from_env() is None
    list(request())


@pytest.mark.parametrize("stream", [True, False])
def test_request_raw_and_normalized_share_ids(tmp_path, monkeypatch, stream):
    path = tmp_path / "trace.jsonl"
    monkeypatch.setenv("S2S_DELEGATION_TRACE_PATH", str(path))
    token = trace_context.set({"turn_id": "turn-a", "turn_revision": 2, "response_key": "response-a"})
    try:
        events = request(stream=stream)
    finally:
        trace_context.reset(token)
    list(events)
    rows = records(path)
    assert len({row["request_id"] for row in rows}) == 1
    assert all(row["turn_id"] == "turn-a" and row["response_key"] == "response-a" for row in rows)
    assert rows[0]["payload"]["tools"] == TOOLS["tools"]
    assert rows[0]["payload"]["model"] == "test"
    assert "DSML" in json.dumps([r for r in rows if r["event"].startswith("raw_")], ensure_ascii=False)
    normalized = [r for r in rows if r["event"] in ("normalized_chunk", "normalized_response")]
    assert "spawn_thinking" in json.dumps(normalized)
    assert path.stat().st_mode & 0o777 == 0o600
    list(request(stream=stream))
    assert len({row["request_id"] for row in records(path)}) == 2


def test_redaction_and_audio(tmp_path, monkeypatch):
    path = tmp_path / "trace.jsonl"
    monkeypatch.setenv("TEST_API_KEY", "environment-secret-value")
    trace = DelegationTrace(str(path))
    trace.emit("request", {
        "Authorization": "private-auth", "credentials": {"user": "private-user"},
        "input_audio": {"data": "private-audio"}, "binary": b"private-bytes",
        "content": "Bearer private-bearer environment-secret-value api_key=private-key",
        "url": "https://user:private-password@example.org/path",
        "arguments": json.dumps({"password": "private-password", "audio": "private-encoded-audio"}),
    })
    assert "private-" not in path.read_text()
    assert "environment-secret-value" not in path.read_text()
    path.chmod(0o666)
    trace.emit("next")
    assert path.stat().st_mode & 0o777 == 0o600


@pytest.mark.parametrize("failure", ["missing_parent", "symlink", "write", "serialization"])
def test_fail_open(tmp_path, monkeypatch, failure):
    path = tmp_path / "trace.jsonl"
    if failure == "missing_parent":
        path = tmp_path / "missing" / "trace.jsonl"
    elif failure == "symlink":
        target = tmp_path / "target"
        target.write_text("untouched")
        path.symlink_to(target)
    elif failure == "write":
        monkeypatch.setattr(os, "write", lambda *a: (_ for _ in ()).throw(OSError("full")))
    else:
        monkeypatch.setattr("speech_to_speech.LLM.delegation_trace._sanitize", lambda *a: 1 / 0)
    monkeypatch.setenv("S2S_DELEGATION_TRACE_PATH", str(path))
    assert list(request())
    if failure == "symlink":
        assert target.read_text() == "untouched"


@pytest.mark.parametrize("consume", [False, True])
def test_laziness_and_close(tmp_path, monkeypatch, consume):
    path = tmp_path / "trace.jsonl"
    monkeypatch.setenv("S2S_DELEGATION_TRACE_PATH", str(path))

    class Source:
        reads = 0
        closes = 0

        def __iter__(self):
            self.reads += 1
            yield chunk(DSML)

        def close(self):
            self.closes += 1

    source = Source()
    client = NS(chat=NS(completions=NS(create=lambda **kw: source)))
    response = _request_chat_completions(
        client=client, model_name="test", messages=[], stream=True,
        extra_body=None, timeout=1, optional_kwargs=TOOLS,
    )
    assert source.reads == 0
    assert [r["event"] for r in records(path)] == ["request"]
    if consume:
        list(response)
    response.close()
    response.close()
    assert source.closes == 1


def test_normalization_error_is_visible(tmp_path, monkeypatch):
    path = tmp_path / "trace.jsonl"
    monkeypatch.setenv("S2S_DELEGATION_TRACE_PATH", str(path))
    with pytest.raises(ValueError):
        list(request(DSML[:-10]))
    rows = records(path)
    assert any(r["event"] == "raw_chunk" for r in rows)
    assert any(r["event"] == "normalized_chunk_error" for r in rows)


def test_real_turn_correlation(tmp_path, monkeypatch):
    path = tmp_path / "trace.jsonl"
    monkeypatch.setenv("S2S_DELEGATION_TRACE_PATH", str(path))
    handler = _make_handler()
    handler.client.chat.completions.next_result = _FakeStream([chunk("hello")])
    text, _, _, _, end = _drive(handler)
    assert text == "hello"
    rows = records(path)
    assert all(row["turn_id"] == "t" and row["turn_revision"] == 0 for row in rows)
    assert all(row["response_key"] == end.response_key for row in rows)
    assert trace_context.get() is None


def test_stream_record_limit(tmp_path, monkeypatch):
    monkeypatch.setattr("speech_to_speech.LLM.delegation_trace._MAX_RECORDS", 2)
    path = tmp_path / "trace.jsonl"
    trace = DelegationTrace(str(path))
    for _ in range(20):
        trace.emit("raw_chunk", "delta")
    assert [r["event"] for r in records(path)] == ["raw_chunk", "raw_chunk", "trace_limit"]


def test_notices_are_once_across_requests(tmp_path, monkeypatch, caplog):
    import speech_to_speech.LLM.delegation_trace as module

    monkeypatch.setattr(module, "_NOTICES", set())
    path = tmp_path / "trace.jsonl"
    for _ in range(3):
        DelegationTrace(str(path)).emit("request")
        DelegationTrace(str(tmp_path / "missing" / "trace.jsonl")).emit("request")
    assert sum("trace active" in r.message for r in caplog.records) == 1
    assert sum("trace write failed" in r.message for r in caplog.records) == 1


def test_realtime_helper_filters_audio_and_deltas(tmp_path, monkeypatch):
    from speech_to_speech.LLM.delegation_trace import trace_realtime_event

    path = tmp_path / "trace.jsonl"
    monkeypatch.setenv("S2S_DELEGATION_TRACE_PATH", str(path))
    trace_realtime_event("realtime_response_mapping", {
        "conn_id": "connection-a", "response_key": "internal-a", "response_id": "resp_a",
    })
    trace_realtime_event("realtime_outbound", {"type": "response.audio.delta", "delta": "audio-secret"})
    trace_realtime_event("realtime_outbound", {"type": "response.text.delta", "delta": "text"})
    trace_realtime_event("realtime_outbound", {"type": "response.done", "response": {"id": "resp_a"}})
    rows = records(path)
    assert [r["event"] for r in rows] == ["realtime_response_mapping", "realtime_outbound"]
    assert rows[0]["payload"]["response_id"] == rows[1]["payload"]["response"]["id"]
    assert "audio-secret" not in path.read_text()
