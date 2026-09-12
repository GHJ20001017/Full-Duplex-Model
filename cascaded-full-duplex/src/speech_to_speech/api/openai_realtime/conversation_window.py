"""Local conversation window for the packaged ``talk`` / ``local`` client.

The native client drives audio through ``sounddevice`` and, until now, rendered the
conversation as terminal lines. This module adds an optional local browser window:
a tiny threaded HTTP server serves one self-contained page and streams conversation
events to it over Server-Sent Events.

Design constraints:

- No new dependencies. The server is stdlib ``http.server``; the page is one static
  HTML asset with no build step and no external network calls, so it keeps working
  offline and never becomes a second thing to maintain.
- The window is a *view*. It renders the same user / assistant transcripts the
  terminal already shows and owns no protocol state, so the terminal path keeps
  working unchanged when the window is disabled.
- Publishing is non-blocking. ``push`` runs on the asyncio event-loop thread while
  the HTTP server lives on its own thread, so events are handed across a lock-guarded
  queue and dropped rather than blocking the audio pipeline.
"""

from __future__ import annotations

import json
import logging
import queue
import threading
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import TracebackType
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Path to the single static page served by the window.
_PAGE_PATH = Path(__file__).with_name("conversation_window.html")
# Per-subscriber buffer. Large enough to absorb a burst of transcript deltas, small
# enough that a wedged browser tab cannot grow memory without bound.
_SUBSCRIBER_QUEUE_SIZE = 512
# Recent durable messages replayed to a tab that connects late (or reloads), so a
# refresh does not blank the conversation. Transient deltas are not kept.
_BACKLOG_SIZE = 200
# Comment frame that keeps idle SSE connections (and any intermediary) from timing out.
_HEARTBEAT = b": keep-alive\n\n"


def _escape_html_bytes(text: str) -> bytes:
    """Escape a title/subtitle for safe insertion into the HTML template."""

    escaped = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")
    return escaped.encode("utf-8")


class ConversationWindow:
    """Serve a local chat page and broadcast conversation events to it.

    The window binds to loopback on an ephemeral port by default, starts a daemon
    server thread, and optionally opens the page in the user's browser. Call
    :meth:`push` from any thread; call :meth:`close` on shutdown.
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 0,
        *,
        open_browser: bool = True,
        title: str = "Full-Duplex Voice",
        subtitle: str = "Cascaded speech-to-speech",
    ) -> None:
        self._host = host
        self._requested_port = port
        self._open_browser = open_browser
        self._server: Optional[ThreadingHTTPServer] = None
        self._thread: Optional[threading.Thread] = None
        self._subscribers: set[queue.Queue[bytes]] = set()
        self._backlog: list[bytes] = []
        self._lock = threading.Lock()
        # Static pieces are injected at serve time so the HTML asset stays free of
        # format placeholders and can be opened directly by tests.
        page = _PAGE_PATH.read_bytes()
        page = page.replace(b"__TITLE__", _escape_html_bytes(title))
        page = page.replace(b"__SUBTITLE__", _escape_html_bytes(subtitle))
        self._page = page

    # -- lifecycle -----------------------------------------------------------------

    def start(self) -> "ConversationWindow":
        """Bind the socket, start serving, and return self for convenient chaining."""

        if self._server is not None:
            return self
        server = _WindowHTTPServer((self._host, self._requested_port), _PageHandler, self)
        server.daemon_threads = True
        self._server = server
        self._thread = threading.Thread(target=server.serve_forever, name="conversation-window", daemon=True)
        self._thread.start()
        logger.info("Conversation window available at %s", self.url)
        if self._open_browser:
            # Open off the calling thread: ``start`` runs on the asyncio loop thread,
            # and a slow browser launch must not delay the audio session's startup.
            threading.Thread(target=self._open_browser_safely, name="conversation-window-browser", daemon=True).start()
        return self

    def _open_browser_safely(self) -> None:
        try:
            webbrowser.open(self.url)
        except Exception:
            # A missing/blocked browser must never abort the session.
            logger.debug("Could not open the conversation window automatically", exc_info=True)

    def close(self) -> None:
        """Stop the server and release every waiting client connection."""

        server = self._server
        self._server = None
        if server is not None:
            server.shutdown()
            server.server_close()
        with self._lock:
            subscribers = list(self._subscribers)
            self._subscribers.clear()
        for subscriber in subscribers:
            try:
                subscriber.put_nowait(b"event: closed\ndata: {}\n\n")
            except queue.Full:
                pass
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def __enter__(self) -> "ConversationWindow":
        return self.start()

    def __exit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc: Optional[BaseException],
        tb: Optional[TracebackType],
    ) -> None:
        self.close()

    @property
    def url(self) -> str:
        server = self._server
        if server is None:
            raise RuntimeError("Conversation window has not been started")
        host, port = server.server_address[:2]
        display_host = "localhost" if host in {"127.0.0.1", "::1", "0.0.0.0"} else host
        return f"http://{display_host}:{port}/"

    # -- broadcasting ---------------------------------------------------------------

    def push(self, event: dict[str, Any]) -> None:
        """Broadcast one transient event (a partial delta or status). Never blocks."""

        frame = self._encode(event)
        if frame is not None:
            self._broadcast(frame)

    def push_message(self, event: dict[str, Any]) -> None:
        """Broadcast a durable message and remember it for late-connecting tabs."""

        frame = self._encode(event)
        if frame is None:
            return
        with self._lock:
            self._backlog.append(frame)
            del self._backlog[:-_BACKLOG_SIZE]
        self._broadcast(frame)

    def clear_backlog(self) -> None:
        with self._lock:
            self._backlog.clear()

    @staticmethod
    def _encode(event: dict[str, Any]) -> Optional[bytes]:
        try:
            payload = json.dumps(event, ensure_ascii=False)
        except (TypeError, ValueError):
            logger.warning("Dropping non-serializable conversation window event", exc_info=True)
            return None
        return f"data: {payload}\n\n".encode("utf-8")

    def _broadcast(self, frame: bytes) -> None:
        with self._lock:
            subscribers = list(self._subscribers)
        for subscriber in subscribers:
            try:
                subscriber.put_nowait(frame)
            except queue.Full:
                # A slow/stalled tab must not slow the pipeline; drop the frame.
                pass

    # -- request handling -----------------------------------------------------------

    def _page_bytes(self) -> bytes:
        return self._page

    def _subscribe(self) -> queue.Queue[bytes]:
        subscriber: queue.Queue[bytes] = queue.Queue(maxsize=_SUBSCRIBER_QUEUE_SIZE)
        with self._lock:
            self._subscribers.add(subscriber)
            backlog = list(self._backlog)
        for frame in backlog:
            try:
                subscriber.put_nowait(frame)
            except queue.Full:
                break
        return subscriber

    def _unsubscribe(self, subscriber: queue.Queue[bytes]) -> None:
        with self._lock:
            self._subscribers.discard(subscriber)


class _WindowHTTPServer(ThreadingHTTPServer):
    """ThreadingHTTPServer that carries a reference to its owning window."""

    allow_reuse_address = True

    def __init__(self, address: tuple[str, int], handler: type[BaseHTTPRequestHandler], window: ConversationWindow):
        self.window = window
        super().__init__(address, handler)


class _PageHandler(BaseHTTPRequestHandler):
    """Serve the page, the SSE stream, and a health probe."""

    protocol_version = "HTTP/1.1"
    server_version = "ConversationWindow"

    @property
    def _window(self) -> ConversationWindow:
        server = self.server
        assert isinstance(server, _WindowHTTPServer)
        return server.window

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002 - stdlib signature
        # The stdlib default writes to stderr; the window is chrome, not pipeline output.
        logger.debug("conversation window: " + format, *args)

    def do_GET(self) -> None:  # noqa: N802 - stdlib naming
        path = self.path.split("?", 1)[0]
        if path in {"/", "/index.html"}:
            self._send_page()
        elif path == "/events":
            self._stream_events()
        elif path == "/health":
            self._send_bytes(HTTPStatus.OK, b"ok", "text/plain; charset=utf-8")
        else:
            self._send_bytes(HTTPStatus.NOT_FOUND, b"not found", "text/plain; charset=utf-8")

    def _send_page(self) -> None:
        self._send_bytes(HTTPStatus.OK, self._window._page_bytes(), "text/html; charset=utf-8")

    def _send_bytes(self, status: HTTPStatus, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _stream_events(self) -> None:
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "keep-alive")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()

        subscriber = self._window._subscribe()
        try:
            # Tell the tab the stream is live so it can drop its "connecting" state.
            self.wfile.write(b"event: open\ndata: {}\n\n")
            self.wfile.flush()
            while True:
                try:
                    frame = subscriber.get(timeout=15.0)
                except queue.Empty:
                    frame = _HEARTBEAT
                self.wfile.write(frame)
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, OSError):
            # The tab closed or navigated away; that is the normal way this ends.
            pass
        finally:
            self._window._unsubscribe(subscriber)
