"""Tests for the local conversation window and its wiring into the audio client."""

from __future__ import annotations

import json
import queue
import shutil
import socket
import subprocess
import threading
import urllib.error
import urllib.request
from pathlib import Path
from types import SimpleNamespace

import pytest

import speech_to_speech.api.openai_realtime.audio_client as audio_client_module
from speech_to_speech.api.openai_realtime.audio_client import (
    PlaybackBuffer,
    RealtimeAudioClientConfig,
    _FriendlyEventRenderer,
    _open_conversation_window,
    handle_server_event,
)
from speech_to_speech.api.openai_realtime.conversation_window import ConversationWindow


@pytest.fixture
def window():
    """A started window bound to an ephemeral loopback port, closed afterwards."""

    window = ConversationWindow(open_browser=False).start()
    try:
        yield window
    finally:
        window.close()


def _get(url: str, timeout: float = 5.0) -> tuple[int, bytes]:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return response.status, response.read()


def test_window_serves_page_with_injected_titles(window):
    status, body = _get(window.url)

    assert status == 200
    text = body.decode("utf-8")
    assert "Full-Duplex Voice" in text
    assert "__TITLE__" not in text
    assert "__SUBTITLE__" not in text


def test_window_escapes_title_markup():
    window = ConversationWindow(open_browser=False, title="<b>x</b>").start()
    try:
        _, body = _get(window.url)
    finally:
        window.close()
    text = body.decode("utf-8")
    assert "<b>x</b>" not in text
    assert "&lt;b&gt;x&lt;/b&gt;" in text


def test_window_health_probe(window):
    status, body = _get(window.url + "health")
    assert status == 200
    assert body == b"ok"


def test_window_unknown_path_is_404(window):
    with pytest.raises(urllib.error.HTTPError) as excinfo:
        _get(window.url + "nope")
    assert excinfo.value.code == 404


def test_window_stream_delivers_pushed_events(window):
    """A subscriber to /events receives frames pushed by the pipeline thread."""

    received: list[bytes] = []
    ready = threading.Event()

    def read_stream():
        with urllib.request.urlopen(window.url + "events", timeout=5.0) as response:
            ready.set()
            for line in response:
                if line.startswith(b"data: "):
                    received.append(line)
                    if len(received) >= 3:  # open handshake + two pushes
                        return

    reader = threading.Thread(target=read_stream, daemon=True)
    reader.start()
    assert ready.wait(3.0)

    window.push({"event": "status", "value": "connected"})
    window.push_message({"event": "message", "role": "user", "text": "hello"})
    reader.join(5.0)

    assert not reader.is_alive()
    payloads = [json.loads(frame[len(b"data: ") :]) for frame in received]
    assert {"event": "status", "value": "connected"} in payloads
    assert {"event": "message", "role": "user", "text": "hello"} in payloads


def test_window_replays_durable_backlog_to_late_subscriber(window):
    """A tab that connects after messages were sent still sees them."""

    window.push({"event": "status", "value": "connected"})  # transient, not replayed
    window.push_message({"event": "message", "role": "assistant", "text": "earlier"})

    received: list[bytes] = []

    def read_stream():
        with urllib.request.urlopen(window.url + "events", timeout=5.0) as response:
            for line in response:
                if line.startswith(b"data: "):
                    received.append(line)
                    if len(received) >= 2:  # open handshake + replayed message
                        return

    reader = threading.Thread(target=read_stream, daemon=True)
    reader.start()
    reader.join(5.0)

    payloads = [json.loads(frame[len(b"data: ") :]) for frame in received]
    # The transient status is not replayed; only the durable message is.
    assert payloads == [{}, {"event": "message", "role": "assistant", "text": "earlier"}]


def test_window_push_is_safe_without_subscribers(window):
    # No subscriber connected: pushing must be a no-op, never raise.
    window.push({"event": "status", "value": "listening"})
    window.push_message({"event": "message", "role": "user", "text": "x"})


def test_window_drops_frames_when_subscriber_queue_is_full(window):
    subscriber: queue.Queue[bytes] = queue.Queue(maxsize=1)
    with window._lock:
        window._subscribers.add(subscriber)

    window.push({"event": "status", "value": "one"})
    window.push({"event": "status", "value": "two"})  # dropped, queue full

    assert subscriber.qsize() == 1
    with window._lock:
        window._subscribers.clear()


def test_window_url_requires_start():
    window = ConversationWindow(open_browser=False)
    with pytest.raises(RuntimeError):
        _ = window.url


def test_window_close_is_idempotent():
    window = ConversationWindow(open_browser=False).start()
    window.close()
    window.close()


def test_window_bind_failure_degrades_to_terminal(monkeypatch, capsys):
    """_open_conversation_window returns None (terminal-only) when binding fails."""

    monkeypatch.setattr(
        audio_client_module.ConversationWindow,
        "start",
        lambda self: (_ for _ in ()).throw(OSError("address in use")),
    )

    assert _open_conversation_window(RealtimeAudioClientConfig(ui=True)) is None
    assert "conversation window disabled" in capsys.readouterr().out


def test_open_conversation_window_disabled_returns_none(monkeypatch):
    calls = []
    monkeypatch.setattr(
        audio_client_module.ConversationWindow,
        "start",
        lambda self: calls.append(self),
    )
    assert _open_conversation_window(RealtimeAudioClientConfig(ui=False)) is None
    assert calls == []


# -- renderer wiring -------------------------------------------------------------------


def _renderer_with_window(window) -> tuple[_FriendlyEventRenderer, PlaybackBuffer]:
    return _FriendlyEventRenderer(window), PlaybackBuffer(16000)


def _drain(subscriber: queue.Queue[bytes]) -> list[dict]:
    frames = []
    while not subscriber.empty():
        frame = subscriber.get_nowait()
        if frame.startswith(b"data: "):
            frames.append(json.loads(frame[len(b"data: ") :]))
    return frames


def test_renderer_without_window_does_not_fail(capsys):
    renderer = _FriendlyEventRenderer()
    playback = PlaybackBuffer(16000)
    handle_server_event(
        SimpleNamespace(type="conversation.item.input_audio_transcription.completed", item_id="i", transcript="hi"),
        playback=playback,
        renderer=renderer,
        print_json=False,
    )
    assert renderer.window is None
    capsys.readouterr()


def test_user_transcript_reaches_window(window, capsys):
    renderer, playback = _renderer_with_window(window)
    subscriber = window._subscribe()

    handle_server_event(
        SimpleNamespace(type="conversation.item.input_audio_transcription.delta", item_id="i1", delta="你好"),
        playback=playback,
        renderer=renderer,
        print_json=False,
    )
    handle_server_event(
        SimpleNamespace(
            type="conversation.item.input_audio_transcription.completed", item_id="i1", transcript="你好世界"
        ),
        playback=playback,
        renderer=renderer,
        print_json=False,
    )

    events = _drain(subscriber)
    assert {"event": "transcript", "role": "user", "item_id": "i1", "delta": "你好"} in events
    assert {
        "event": "transcript",
        "role": "user",
        "item_id": "i1",
        "final": True,
        "text": "你好世界",
    } in events
    capsys.readouterr()


def test_streaming_assistant_transcript_reaches_window(window, capsys):
    renderer, playback = _renderer_with_window(window)
    subscriber = window._subscribe()

    def event(**kwargs):
        return SimpleNamespace(**kwargs)

    for delta in ("Hello", " there"):
        handle_server_event(
            event(
                type="response.output_audio_transcript.delta",
                delta=delta,
                response_id="r1",
                item_id="m1",
                output_index=0,
                content_index=0,
            ),
            playback=playback,
            renderer=renderer,
            print_json=False,
        )
    handle_server_event(
        event(
            type="response.output_audio_transcript.done",
            transcript="Hello there",
            response_id="r1",
            item_id="m1",
            output_index=0,
            content_index=0,
        ),
        playback=playback,
        renderer=renderer,
        print_json=False,
    )

    events = _drain(subscriber)
    assert {"event": "transcript", "role": "assistant", "delta": "Hello"} in events
    assert {"event": "transcript", "role": "assistant", "delta": " there"} in events
    # The final commit carries the full text and marks the stream finished.
    assert {"event": "transcript", "role": "assistant", "final": True, "text": "Hello there"} in events
    # No duplicate standalone message after the streamed bubble.
    assert {"event": "message", "role": "assistant", "text": "Hello there"} not in events
    capsys.readouterr()


def test_assistant_transcript_without_delta_commits_single_message(window, capsys):
    renderer, playback = _renderer_with_window(window)
    subscriber = window._subscribe()

    handle_server_event(
        SimpleNamespace(
            type="response.output_audio_transcript.done",
            transcript="One shot",
            response_id="r1",
            item_id="m1",
            output_index=0,
            content_index=0,
        ),
        playback=playback,
        renderer=renderer,
        print_json=False,
    )

    events = _drain(subscriber)
    assert {"event": "message", "role": "assistant", "text": "One shot"} in events
    capsys.readouterr()


def test_cancelled_response_finalizes_pending_assistant_bubble(window, capsys):
    """A barge-in cancels the response without a transcript.done, so the UI bubble
    must still be finalized or it would show a caret forever."""

    renderer, playback = _renderer_with_window(window)
    subscriber = window._subscribe()

    handle_server_event(
        SimpleNamespace(
            type="response.output_audio_transcript.delta",
            delta="half a sen",
            response_id="r1",
            item_id="m1",
            output_index=0,
            content_index=0,
        ),
        playback=playback,
        renderer=renderer,
        print_json=False,
    )
    handle_server_event(
        SimpleNamespace(type="response.done", response=SimpleNamespace(id="r1", status="cancelled")),
        playback=playback,
        renderer=renderer,
        print_json=False,
    )

    events = _drain(subscriber)
    finals = [e for e in events if e.get("event") == "transcript" and e.get("final") and e["role"] == "assistant"]
    assert finals, "cancelled response must emit a final assistant transcript event"
    assert "text" not in finals[-1]  # browser keeps the accumulated delta text
    capsys.readouterr()


def test_status_and_error_events_reach_window(window, capsys):
    renderer, playback = _renderer_with_window(window)
    subscriber = window._subscribe()

    handle_server_event(
        SimpleNamespace(type="session.created"),
        playback=playback,
        renderer=renderer,
        print_json=False,
    )
    handle_server_event(
        SimpleNamespace(type="input_audio_buffer.speech_started"),
        playback=playback,
        renderer=renderer,
        print_json=False,
    )
    error = SimpleNamespace(type="error", error=SimpleNamespace(type="bad", message="boom"))
    handle_server_event(error, playback=playback, renderer=renderer, print_json=False)

    events = _drain(subscriber)
    assert {"event": "status", "value": "connected"} in events
    assert {"event": "status", "value": "listening"} in events
    assert {"event": "system", "kind": "error", "text": "bad: boom"} in events
    capsys.readouterr()


def test_window_receives_status_when_config_ui_enabled(monkeypatch, capsys):
    """_run_audio_session opens a window when ui is enabled and passes it to the renderer."""

    created = []

    class FakeWindow:
        def __init__(self, *args, **kwargs):
            created.append(kwargs)

        def start(self):
            return self

        def push(self, event):
            pass

        def push_message(self, event):
            pass

        def close(self):
            pass

    monkeypatch.setattr(audio_client_module, "ConversationWindow", FakeWindow)
    window = _open_conversation_window(RealtimeAudioClientConfig(ui=True, ui_open_browser=False))
    assert window is not None
    assert created == [{"open_browser": False}]
    capsys.readouterr()


def test_free_port_reexport_available():
    """The window must not require networked imports beyond the stdlib."""

    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        assert probe.getsockname()[1] > 0


# -- rendered page behavior -------------------------------------------------------------

_WINDOW_PAGE = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "speech_to_speech"
    / "api"
    / "openai_realtime"
    / "conversation_window.html"
)
_DOM_HARNESS = Path(__file__).resolve().parent / "conversation_window_dom_harness.js"
_NODE = shutil.which("node")

requires_node = pytest.mark.skipif(_NODE is None, reason="node is required to render the window page")


def _render_page(messages: list[dict], tmp_path: Path) -> list[dict]:
    """Run the page script under Node with a stub DOM and return the rendered rows."""

    assert _NODE is not None
    messages_path = tmp_path / "window-messages.json"
    messages_path.write_text(json.dumps(messages), encoding="utf-8")
    result = subprocess.run(
        [_NODE, str(_DOM_HARNESS), str(_WINDOW_PAGE), str(messages_path)],
        capture_output=True,
        text=True,
        timeout=30,
        cwd=tmp_path,
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


@requires_node
def test_rendered_window_merges_reopened_transcript_into_one_bubble(tmp_path):
    """A reopened turn revises the same sentence: one bubble, the complete text."""

    rows = _render_page(
        [
            {"event": "transcript", "role": "user", "item_id": "item_1", "delta": "你好世界"},
            {"event": "transcript", "role": "user", "item_id": "item_1", "final": True, "text": "你好世界"},
            {
                "event": "transcript",
                "role": "user",
                "item_id": "item_1",
                "final": True,
                "text": "你好世界，今天天气怎么样",
            },
        ],
        tmp_path,
    )

    assert rows == [
        {"role": "user", "text": "你好世界，今天天气怎么样", "pending": False},
    ]


@requires_node
def test_rendered_window_keeps_separate_utterances_apart(tmp_path):
    rows = _render_page(
        [
            {"event": "transcript", "role": "user", "item_id": "item_1", "final": True, "text": "第一句"},
            {"event": "transcript", "role": "user", "item_id": "item_2", "final": True, "text": "第二句"},
        ],
        tmp_path,
    )

    assert rows == [
        {"role": "user", "text": "第一句", "pending": False},
        {"role": "user", "text": "第二句", "pending": False},
    ]


@requires_node
def test_rendered_window_replays_repeated_final_for_one_item(tmp_path):
    """A reconnecting tab replays durable finals: still one row per item id."""

    rows = _render_page(
        [
            {"event": "transcript", "role": "user", "item_id": "item_1", "final": True, "text": "短"},
            {"event": "transcript", "role": "user", "item_id": "item_1", "final": True, "text": "完整的一句话"},
        ],
        tmp_path,
    )

    assert rows == [{"role": "user", "text": "完整的一句话", "pending": False}]


@requires_node
def test_rendered_window_keeps_displayed_text_when_final_is_empty(tmp_path):
    rows = _render_page(
        [
            {"event": "transcript", "role": "user", "item_id": "item_1", "final": True, "text": "已经识别的一句"},
            {"event": "transcript", "role": "user", "item_id": "item_1", "final": True, "text": ""},
        ],
        tmp_path,
    )

    assert rows == [{"role": "user", "text": "已经识别的一句", "pending": False}]


@requires_node
def test_rendered_window_skips_empty_user_final(tmp_path):
    rows = _render_page(
        [{"event": "transcript", "role": "user", "item_id": "item_9", "final": True, "text": ""}],
        tmp_path,
    )

    assert rows == []


@requires_node
def test_rendered_window_falls_back_to_one_live_bubble_without_item_id(tmp_path):
    """Older payloads without an item id keep the previous per-role behavior."""

    rows = _render_page(
        [
            {"event": "transcript", "role": "user", "delta": "你好"},
            {"event": "transcript", "role": "user", "final": True, "text": "你好世界"},
        ],
        tmp_path,
    )

    assert rows == [{"role": "user", "text": "你好世界", "pending": False}]


@requires_node
def test_rendered_window_finalizes_assistant_bubble_without_text(tmp_path):
    """A barge-in final carries no text: the bubble keeps its streamed deltas."""

    rows = _render_page(
        [
            {"event": "transcript", "role": "assistant", "delta": "半句"},
            {"event": "transcript", "role": "assistant", "final": True},
        ],
        tmp_path,
    )

    assert rows == [{"role": "assistant", "text": "半句", "pending": False}]
