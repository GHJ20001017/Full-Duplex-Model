"""Diagnostics correlate pipeline IDs and only successfully delivered wire events."""

import asyncio
import json
from queue import Queue
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from fastapi import WebSocketDisconnect
from openai.types.realtime import ResponseCreateEvent
from starlette.websockets import WebSocketState

from speech_to_speech.api.openai_realtime import transports
from speech_to_speech.api.openai_realtime.handlers import response
from speech_to_speech.api.openai_realtime.service import RealtimeService
from speech_to_speech.pipeline.messages import GenerateResponseRequest, ResponsePrefetchTransaction


@pytest.fixture
def session():
    queue = Queue()
    service = RealtimeService(text_prompt_queue=queue)
    conn_id = service.register()
    yield service, conn_id, queue
    service.unregister(conn_id)


@pytest.mark.parametrize("creation", ["implicit", "late_key", "explicit", "prefetch"])
def test_response_mapping_at_identity_assignment(session, monkeypatch, creation):
    service, conn_id, queue = session
    trace = Mock()
    monkeypatch.setattr(response, "trace_realtime_event", trace)
    state = service._state(conn_id)
    key = "pipeline-response-key"
    if creation == "implicit":
        response_id, _ = service.response._ensure_response(conn_id, key)
    elif creation == "late_key":
        response_id, _ = service.response._ensure_response(conn_id)
        trace.assert_not_called()
        service.response._ensure_response(conn_id, key)
    else:
        if creation == "prefetch":
            request = GenerateResponseRequest(
                runtime_config=state.runtime_config,
                prefetch_transaction=ResponsePrefetchTransaction(),
            )
            state.tool_followup_prefetch_request = request
            state.mark_response_pending(request.response_key)
            key = request.response_key
        created = service.response.handle_response_create(conn_id, ResponseCreateEvent(type="response.create"))
        response_id = created.response.id
        if creation == "explicit":
            key = queue.get_nowait().response_key
    trace.assert_called_once_with(
        "realtime_response_mapping",
        {"conn_id": conn_id, "response_key": key, "response_id": response_id},
    )
    service.response._ensure_response(conn_id, key)
    assert trace.call_count == 1
    with pytest.raises(RuntimeError, match="different active response"):
        service.response._ensure_response(conn_id, "another-key")
    assert trace.call_count == 1


def test_response_done_keeps_wire_id_after_state_reset(session, monkeypatch):
    service, conn_id, _ = session
    records = []
    monkeypatch.setattr(response, "trace_realtime_event", lambda *args: records.append(args))
    monkeypatch.setattr(transports, "trace_realtime_event", lambda *args: records.append(args))
    response_id, _ = service.response._ensure_response(conn_id, "pipeline-key")
    events = service.response.finish_response(conn_id)
    assert service._state(conn_id).current_response_id is None
    assert service._state(conn_id).current_response_key is None
    ws = SimpleNamespace(application_state=WebSocketState.CONNECTED, send_json=AsyncMock())
    asyncio.run(transports.WebSocketTransport(ws).send_events(events))
    done = next(payload for event, payload in records if event == "realtime_outbound" and payload["type"] == "response.done")
    assert done["response"]["id"] == response_id
    assert records[0] == (
        "realtime_response_mapping",
        {"conn_id": conn_id, "response_key": "pipeline-key", "response_id": response_id},
    )
    assert ws.send_json.await_count == len(events)


def test_jsonl_joins_model_request_to_old_response_after_new_response_starts(session, monkeypatch, tmp_path):
    from speech_to_speech.LLM.delegation_trace import trace_context
    from tests.test_deepseek_dsml import request

    path = tmp_path / "trace.jsonl"
    monkeypatch.setenv("S2S_DELEGATION_TRACE_PATH", str(path))
    service, conn_id, _ = session
    token = trace_context.set({"turn_id": "turn-old", "response_key": "key-old"})
    try:
        list(request())
    finally:
        trace_context.reset(token)
    old_id, _ = service.response._ensure_response(conn_id, "key-old")
    old_events = service.response.finish_response(conn_id)
    new_id, _ = service.response._ensure_response(conn_id, "key-new")
    ws = SimpleNamespace(application_state=WebSocketState.CONNECTED, send_json=AsyncMock())
    asyncio.run(transports.WebSocketTransport(ws).send_events(old_events))
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    model = next(row for row in rows if row["event"] == "request")
    mappings = {row["payload"]["response_key"]: row["payload"]
                for row in rows if row["event"] == "realtime_response_mapping"}
    done = next(row["payload"] for row in rows
                if row["event"] == "realtime_outbound" and row["payload"]["type"] == "response.done")
    assert model["turn_id"] == "turn-old"
    assert mappings[model["response_key"]]["response_id"] == done["response"]["id"] == old_id
    assert mappings["key-new"]["response_id"] == new_id != old_id
    assert mappings["key-old"]["conn_id"] == conn_id


@pytest.mark.parametrize("failure", [None, WebSocketDisconnect(), RuntimeError("websocket.close"), ValueError("send failed")])
@pytest.mark.parametrize("connected", [True, False])
def test_websocket_only_traces_successful_send(monkeypatch, failure, connected):
    payload = {"type": "response.done", "response": {"id": "resp-offline"}}
    event = SimpleNamespace(type=payload["type"], model_dump=Mock(return_value=payload))
    trace = Mock()
    ws = SimpleNamespace(
        application_state=WebSocketState.CONNECTED if connected else WebSocketState.DISCONNECTED,
        send_json=AsyncMock(side_effect=failure),
    )

    def record(name, sent):
        assert ws.send_json.await_count == 1
        assert sent is payload
        trace(name, sent)

    monkeypatch.setattr(transports, "trace_realtime_event", record)
    asyncio.run(transports.send_ws_event(ws, event))
    if connected and failure is None:
        trace.assert_called_once_with("realtime_outbound", payload)
    else:
        trace.assert_not_called()
    assert event.model_dump.call_count == int(connected)


@pytest.mark.parametrize("event_type", ["response.output_audio.delta", "response.output_text.delta", "response.function_call_arguments.delta"])
def test_websocket_deltas_are_not_traced(monkeypatch, event_type):
    trace = Mock()
    monkeypatch.setattr(transports, "trace_realtime_event", trace)
    event = SimpleNamespace(type=event_type, model_dump=Mock(return_value={"type": event_type, "delta": "data"}))
    ws = SimpleNamespace(application_state=WebSocketState.CONNECTED, send_json=AsyncMock())
    asyncio.run(transports.send_ws_event(ws, event))
    ws.send_json.assert_awaited_once()
    event.model_dump.assert_called_once_with()
    trace.assert_not_called()


@pytest.mark.parametrize("channel_state", ["open", "closed", "missing"])
@pytest.mark.parametrize("failure", [False, True])
def test_webrtc_only_traces_successful_send(monkeypatch, channel_state, failure):
    rtc = pytest.importorskip("speech_to_speech.api.openai_realtime.webrtc_session")
    trace = Mock()
    monkeypatch.setattr(rtc, "trace_realtime_event", trace)
    dc = SimpleNamespace(readyState=channel_state, send=Mock(side_effect=ValueError("send failed") if failure else None))
    transport = object.__new__(rtc.WebRTCSession)
    transport._dc = None if channel_state == "missing" else dc
    payload = {"type": "response.done", "response": {"id": "resp-after-reset"}}
    event = SimpleNamespace(type=payload["type"], model_dump=Mock(return_value=payload))

    def record(name, sent):
        dc.send.assert_called_once_with(json.dumps(payload))
        trace(name, sent)

    monkeypatch.setattr(rtc, "trace_realtime_event", record)
    asyncio.run(transport.send_events([event]))
    if channel_state == "open" and not failure:
        trace.assert_called_once_with("realtime_outbound", payload)
    else:
        trace.assert_not_called()
    assert event.model_dump.call_count == int(channel_state == "open")


def test_webrtc_deltas_are_not_traced(monkeypatch):
    rtc = pytest.importorskip("speech_to_speech.api.openai_realtime.webrtc_session")
    trace = Mock()
    monkeypatch.setattr(rtc, "trace_realtime_event", trace)
    transport = object.__new__(rtc.WebRTCSession)
    transport._dc = SimpleNamespace(readyState="open", send=Mock())
    events = [
        SimpleNamespace(type=kind, model_dump=Mock(return_value={"type": kind, "delta": "data"}))
        for kind in ("response.output_audio.delta", "response.output_text.delta", "response.function_call_arguments.delta")
    ]
    asyncio.run(transport.send_events(events))
    assert transport._dc.send.call_count == len(events)
    for event in events:
        event.model_dump.assert_called_once_with()
    trace.assert_not_called()
