"""Native Python microphone/speaker client used by the ``talk`` and ``local`` commands.

The browser demo has separate browser-native WebSocket and WebRTC clients under
``demo/``. Those clients share the Realtime protocol with this module, but not
its OpenAI Python SDK, ``sounddevice``, or process-signal implementation.
"""

from __future__ import annotations

import asyncio
import base64
import inspect
import json
import logging
import signal
import time
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from importlib import import_module
from importlib.util import module_from_spec, spec_from_file_location
from ipaddress import ip_address
from pathlib import Path
from queue import Empty, Full, Queue
from threading import Event, Lock
from typing import Any, Optional
from urllib.parse import urlsplit, urlunsplit

from jsonschema import SchemaError, ValidationError
from jsonschema.validators import validator_for
from openai import AsyncOpenAI

from speech_to_speech.pipeline.transcript_logging import log_exception

logger = logging.getLogger(__name__)

# 类型别名：流式助手转写需要用响应、项目和内容索引共同去重。
_AssistantTranscriptStream = tuple[str | None, str | None, int | None, int | None]
# 工具执行器必须是异步函数，避免阻塞 Realtime 事件接收循环。
ToolExecutor = Callable[[str, dict[str, Any]], Awaitable[Any]]
# 用于关联 response.create 与其 response.created/error 事件的内部元数据键。
_TOOL_CREATE_ID_METADATA_KEY = "s2s_local_tool_create_id"
# 默认不等待额外音频，收到音频块即可开始播放。
_PLAYBACK_START_BUFFER_MS = 0.0


@dataclass(frozen=True)
class ToolResult:
    """A local tool output with a per-call follow-up response policy."""

    output: Any
    create_response: bool = True


@dataclass
class RealtimeAudioClientConfig:
    """Configuration for the packaged microphone/speaker Realtime client.

    该配置同时覆盖网络连接、音频设备、终端显示、唤醒词和本地工具。
    16 kHz 是本地服务默认采样率；只有显式使用 24 kHz 时才发送 PCM 格式字段。
    """

    url: str = "ws://127.0.0.1:7869/v1/realtime"
    model: str = "local"
    api_key: Optional[str] = None
    send_rate: int = 16000
    recv_rate: int = 16000
    playback_buffer_ms: float = _PLAYBACK_START_BUFFER_MS
    chunk_size: int = 1024
    input_device: Optional[int] = None
    output_device: Optional[int] = None
    instructions: Optional[str] = None
    voice: Optional[str] = None
    print_json: bool = False
    block_mic_during_playback: bool = False
    log_transcripts: bool = False
    connection_retry_timeout_s: float = 30.0
    tools: list[dict[str, Any]] = field(default_factory=list)
    tool_executor: ToolExecutor | None = None
    tool_response_create: bool = True
    wake_word: str | None = None
    wake_word_timeout_s: float = 300.0
    wake_ack: str = "嗯哼，您说"

    def __post_init__(self) -> None:
        if not 0 <= self.playback_buffer_ms < float("inf"):
            raise ValueError("playback_buffer_ms must be a finite non-negative number")


def load_realtime_tool_module(module_name: str) -> tuple[list[dict[str, Any]], ToolExecutor, bool]:
    """加载 CLI 指定的工具模块，并返回工具定义、执行器和响应策略。

    首先按 Python 模块导入；控制台脚本找不到当前目录时，再回退到项目目录
    下的同名 ``.py`` 文件。加载后立即校验，避免连接建立后才发现配置错误。
    """

    if not module_name or not module_name.strip():
        raise ValueError("Tool module name must not be empty")
    try:
        module = import_module(module_name)
    except ModuleNotFoundError as exc:
        # Console-script entry points do not consistently include the working
        # directory on sys.path, so allow local tool modules from the project.
        if exc.name != module_name and not module_name.startswith("examples."):
            raise
        module_path = Path.cwd() / Path(*module_name.split(".")).with_suffix(".py")
        if not module_path.is_file():
            raise
        spec = spec_from_file_location(module_name, module_path)
        if spec is None or spec.loader is None:
            raise
        module = module_from_spec(spec)
        spec.loader.exec_module(module)
    try:
        tools = list(module.TOOLS)
    except AttributeError as exc:
        raise ValueError(f"Tool module {module_name!r} must define TOOLS") from exc
    executor = getattr(module, "execute_tool", None)
    if not callable(executor):
        raise ValueError(f"Tool module {module_name!r} must define callable execute_tool(name, arguments)")
    create_response = getattr(module, "CREATE_RESPONSE", True)
    if not isinstance(create_response, bool):
        raise ValueError(f"Tool module {module_name!r} CREATE_RESPONSE must be a boolean")
    _validate_tool_config(tools, executor)
    return tools, executor, create_response


def _validate_tool_config(tools: list[dict[str, Any]], executor: ToolExecutor | None) -> dict[str, Any]:
    """校验工具声明，并为每个工具构造可复用的 JSON Schema 校验器。"""

    # 工具定义和执行器必须成对出现，否则模型生成调用后无法完成闭环。
    if tools and executor is None:
        raise ValueError("A tool_executor is required when tools are configured")
    if tools and not callable(executor):
        raise ValueError("tool_executor must be callable")
    validators: dict[str, Any] = {}
    for tool in tools:
        if not isinstance(tool, dict) or tool.get("type") != "function":
            raise ValueError("Each local client tool must be a function tool definition")
        name = tool.get("name")
        if not isinstance(name, str) or not name:
            raise ValueError("Each local client tool must have a non-empty name")
        if name in validators:
            raise ValueError(f"Duplicate local client tool name: {name}")
        parameters = tool.get("parameters", {})
        if not isinstance(parameters, dict):
            raise ValueError(f"Tool {name!r} parameters must be a JSON Schema object")
        validator_class = validator_for(parameters)
        try:
            validator_class.check_schema(parameters)
        except SchemaError as exc:
            raise ValueError(f"Tool {name!r} parameters are not a valid JSON Schema: {exc}") from exc
        validators[name] = validator_class(parameters)
    return validators


def normalize_realtime_url(url: str) -> tuple[str, str]:
    """将完整 Realtime 端点转换为 SDK 所需的 HTTP 和 WebSocket 基础地址。"""

    # SDK 的 HTTP 与 WebSocket 地址共享主机和路径，但协议不同。
    parsed = urlsplit(url.strip())
    if parsed.scheme not in {"ws", "wss", "http", "https"} or not parsed.netloc:
        raise ValueError("--url must be an absolute ws://, wss://, http://, or https:// URL")
    if parsed.query or parsed.fragment:
        raise ValueError("--url must not include a query string or fragment")

    path = parsed.path.rstrip("/")
    if not path.endswith("/realtime"):
        raise ValueError("--url must be the full Realtime endpoint ending in /realtime")

    sdk_path = path[: -len("/realtime")]
    websocket_scheme = "wss" if parsed.scheme in {"wss", "https"} else "ws"
    http_scheme = "https" if websocket_scheme == "wss" else "http"
    websocket_base_url = urlunsplit((websocket_scheme, parsed.netloc, sdk_path, "", ""))
    base_url = urlunsplit((http_scheme, parsed.netloc, sdk_path, "", ""))
    return base_url, websocket_base_url


def _make_client(config: RealtimeAudioClientConfig) -> AsyncOpenAI:
    base_url, websocket_base_url = normalize_realtime_url(config.url)
    client_kwargs: dict[str, Any] = {
        "base_url": base_url,
        "websocket_base_url": websocket_base_url,
    }
    api_key = config.api_key
    if api_key is None:
        hostname = urlsplit(config.url.strip()).hostname
        try:
            is_loopback = hostname == "localhost" or (hostname is not None and ip_address(hostname).is_loopback)
        except ValueError:
            is_loopback = False
        if is_loopback:
            api_key = "local"
    # Omitting the key for non-loopback endpoints lets the SDK read OPENAI_API_KEY.
    if api_key is not None:
        client_kwargs["api_key"] = api_key
    return AsyncOpenAI(**client_kwargs)


def build_session_update(config: RealtimeAudioClientConfig) -> dict[str, Any]:
    """Build the Realtime session update used by local and standalone clients."""

    _validate_tool_config(config.tools, config.tool_executor)

    def maybe_pcm_format(rate: int) -> Optional[dict[str, Any]]:
        # The OpenAI SDK models only validate explicit PCM formats at 24 kHz.
        # Omitting the format selects this server's native 16 kHz pipeline rate.
        if rate == 16000:
            return None
        if rate == 24000:
            return {"type": "audio/pcm", "rate": 24000}
        raise ValueError(
            f"Unsupported rate {rate}. Use 16000 for the local pipeline default "
            "or 24000 for the OpenAI Realtime PCM schema."
        )

    input_config: dict[str, Any] = {
        "turn_detection": {"type": "server_vad", "interrupt_response": True},
    }
    output_config: dict[str, Any] = {}

    input_format = maybe_pcm_format(config.send_rate)
    output_format = maybe_pcm_format(config.recv_rate)
    if input_format is not None:
        input_config["format"] = input_format
    if output_format is not None:
        output_config["format"] = output_format
    if config.voice:
        output_config["voice"] = config.voice

    session: dict[str, Any] = {
        "type": "realtime",
        "audio": {
            "input": input_config,
            "output": output_config,
        },
    }
    if config.instructions:
        session["instructions"] = config.instructions
    if config.tools:
        session["tools"] = config.tools
        session["tool_choice"] = "auto"
    return {"type": "session.update", "session": session}


class PlaybackBuffer:
    """线程安全的播放缓冲区，连接协程写入，sounddevice 回调读取。

    ``_active_until`` 用于判断播放期间是否暂时屏蔽麦克风；
    ``_input_complete`` 区分“仍在接收音频”和“已接收完但尚未播放完”。
    """

    def __init__(self, recv_rate: int, startup_buffer_ms: float = _PLAYBACK_START_BUFFER_MS) -> None:
        if recv_rate <= 0:
            raise ValueError("recv_rate must be positive")
        if not 0 <= startup_buffer_ms < float("inf"):
            raise ValueError("startup_buffer_ms must be a finite non-negative number")
        self.recv_rate = recv_rate
        # int16 单声道每个采样占 2 字节；强制偶数避免截断半个采样。
        self._startup_buffer_bytes = int(recv_rate * 2 * startup_buffer_ms / 1000)
        self._startup_buffer_bytes -= self._startup_buffer_bytes % 2
        self._audio = bytearray()
        self._lock = Lock()
        self._active_until = 0.0
        self._playing = False
        self._input_complete = False
        self._response_id: str | None = None
        self._cancelled_response_ids: set[str] = set()

    def clear(self) -> None:
        with self._lock:
            self._active_until = 0.0
            self._audio.clear()
            self._playing = False
            self._input_complete = False

    def activate_response(self, response_id: str | None) -> None:
        with self._lock:
            self._response_id = response_id

    def cancel_active_response(self) -> None:
        with self._lock:
            if self._response_id is not None:
                self._cancelled_response_ids.add(self._response_id)
            self._active_until = 0.0
            self._audio.clear()
            self._playing = False
            self._input_complete = False

    def finish_response(self, response_id: str | None, *, cancelled: bool) -> None:
        with self._lock:
            if response_id is not None and cancelled:
                self._cancelled_response_ids.add(response_id)
            if response_id is None or response_id == self._response_id:
                self._active_until = 0.0 if cancelled else self._active_until
                if cancelled:
                    self._audio.clear()
                    self._playing = False
                    self._input_complete = False
                if response_id == self._response_id:
                    self._response_id = None

    def append(self, audio: bytes, response_id: str | None = None) -> None:
        if not audio:
            return
        with self._lock:
            if response_id is not None and response_id in self._cancelled_response_ids:
                return
            if response_id is not None:
                self._response_id = response_id
            if self._input_complete and not self._audio:
                self._playing = False
            self._input_complete = False
            self._audio.extend(audio)
            self._active_until = time.monotonic() + max(0.15, len(audio) / (2 * self.recv_rate))

    def finish(self) -> None:
        """Allow a completed short response to play before reaching the startup target."""

        with self._lock:
            self._input_complete = True

    def is_active(self) -> bool:
        with self._lock:
            return bool(self._audio) or time.monotonic() < self._active_until

    def write(self, outdata: Any) -> None:
        """向声卡填充一个块；数据不足时补静音，绝不阻塞音频回调。"""

        needed = len(outdata)
        with self._lock:
            if not self._playing:
                ready = len(self._audio) >= self._startup_buffer_bytes
                if not ready and not (self._input_complete and self._audio):
                    outdata[:] = b"\x00" * needed
                    return
                self._playing = True
            available = min(needed, len(self._audio))
            if available:
                outdata[:available] = self._audio[:available]
                del self._audio[:available]
            if available < needed:
                outdata[available:] = b"\x00" * (needed - available)
                self._playing = False
            if not self._audio and self._input_complete:
                self._playing = False

    @property
    def buffered_bytes(self) -> int:
        with self._lock:
            return len(self._audio)


class _FriendlyEventRenderer:
    """将 Realtime 事件渲染为可读终端输出，并处理增量文本的行覆盖。"""

    def __init__(self) -> None:
        # Input transcription events can arrive out of order across items.
        self.user_transcript_by_item: dict[str | None, str] = {}
        self.live_user_width = 0
        self.saw_user_speech = False
        self.live_assistant_stream: _AssistantTranscriptStream | None = None
        self.streamed_assistant_transcripts: set[_AssistantTranscriptStream] = set()

    def render_live_user_text(self, text: str, *, final: bool = False) -> None:
        """实时覆盖显示用户转写；完成事件到达后提交该行并清零宽度。"""

        line = f"USER: {text}"
        padded = line + (" " * max(0, self.live_user_width - len(line)))
        if final:
            print(f"\r{padded}", flush=True)
            self.live_user_width = 0
            return
        print(f"\r{padded}", end="", flush=True)
        self.live_user_width = len(line)

    def clear_live_user_text(self) -> None:
        if self.live_user_width == 0:
            return
        print("\r" + (" " * self.live_user_width) + "\r", end="", flush=True)
        self.live_user_width = 0

    @staticmethod
    def _assistant_stream_key(event: Any) -> _AssistantTranscriptStream:
        return (
            getattr(event, "response_id", None),
            getattr(event, "item_id", None),
            getattr(event, "output_index", None),
            getattr(event, "content_index", None),
        )

    def render_assistant_text_delta(self, event: Any) -> None:
        delta = event.delta or ""
        if not delta:
            return
        self.clear_live_user_text()
        stream_key = self._assistant_stream_key(event)
        if self.live_assistant_stream != stream_key:
            self.finish_live_assistant_text()
            delta = delta.lstrip()
            if not delta:
                return
            print("ASSISTANT: ", end="", flush=True)
            self.live_assistant_stream = stream_key
        self.streamed_assistant_transcripts.add(stream_key)
        print(delta, end="", flush=True)

    def render_assistant_text_done(self, event: Any) -> None:
        stream_key = self._assistant_stream_key(event)
        if self.live_assistant_stream == stream_key:
            self.finish_live_assistant_text()
        if stream_key in self.streamed_assistant_transcripts:
            self.streamed_assistant_transcripts.remove(stream_key)
            return
        self.clear_live_user_text()
        self.finish_live_assistant_text()
        print(f"ASSISTANT: {event.transcript or ''}", flush=True)

    def finish_live_assistant_text(self) -> None:
        if self.live_assistant_stream is not None:
            print("", flush=True)
            self.live_assistant_stream = None

    def reset_assistant_text(self) -> None:
        self.finish_live_assistant_text()
        self.streamed_assistant_transcripts.clear()

    def finish_assistant_response(self, response_id: str | None) -> None:
        self.finish_live_assistant_text()
        self.streamed_assistant_transcripts = {
            stream_key for stream_key in self.streamed_assistant_transcripts if stream_key[0] != response_id
        }


def handle_server_event(
    event: Any,
    *,
    playback: PlaybackBuffer,
    renderer: _FriendlyEventRenderer,
    print_json: bool,
) -> None:
    """处理一个 Realtime 生命周期事件，更新播放缓冲区和终端状态。

    工具协调器在调用方先处理同一事件；本函数只负责音频播放、转写和显示，
    两者共享事件但职责分离。
    """

    if print_json:
        # 调试模式仍保留友好输出状态，先换行避免两种格式互相覆盖。
        renderer.finish_live_assistant_text()
        try:
            print(f"EVENT: {event.model_dump_json()}", flush=True)
        except Exception:
            print(f"EVENT: {event}", flush=True)

    if event.type == "session.created":
        renderer.finish_live_assistant_text()
        print("Connected.", flush=True)
    elif event.type == "input_audio_buffer.speech_started":
        renderer.finish_live_assistant_text()
        playback.cancel_active_response()
        if renderer.saw_user_speech:
            print("", flush=True)
        renderer.saw_user_speech = True
    elif event.type == "input_audio_buffer.speech_stopped":
        return
    elif event.type == "conversation.item.input_audio_transcription.delta":
        renderer.finish_live_assistant_text()
        item_id = getattr(event, "item_id", None)
        transcript = renderer.user_transcript_by_item.get(item_id, "") + (event.delta or "")
        renderer.user_transcript_by_item[item_id] = transcript
        display_text = transcript.strip()
        if display_text:
            renderer.render_live_user_text(display_text)
    elif event.type == "conversation.item.input_audio_transcription.completed":
        renderer.finish_live_assistant_text()
        item_id = getattr(event, "item_id", None)
        transcript = event.transcript or ""
        renderer.render_live_user_text(transcript.strip(), final=True)
        renderer.user_transcript_by_item.pop(item_id, None)
    elif event.type == "response.created":
        playback.activate_response(getattr(event.response, "id", None))
        renderer.clear_live_user_text()
        renderer.finish_live_assistant_text()
        print("ASSISTANT: <response started>", flush=True)
    elif event.type in {"response.output_item.added", "response.output_item.done"}:
        return
    elif event.type == "response.output_audio.delta":
        # 音频增量是 base64 编码的 int16 PCM，直接追加到播放缓冲区。
        playback.append(base64.b64decode(event.delta), getattr(event, "response_id", None))
    elif event.type == "response.output_audio.done":
        playback.finish()
        renderer.finish_live_assistant_text()
        print("ASSISTANT: <audio done>", flush=True)
    elif event.type == "response.output_audio_transcript.delta":
        renderer.render_assistant_text_delta(event)
    elif event.type == "response.output_audio_transcript.done":
        renderer.render_assistant_text_done(event)
    elif event.type == "response.function_call_arguments.done":
        renderer.finish_live_assistant_text()
        print(
            f"TOOL: {event.name} call_id={event.call_id} arguments={event.arguments}",
            flush=True,
        )
    elif event.type == "response.done":
        response_id = getattr(event.response, "id", None)
        renderer.finish_assistant_response(response_id)
        cancelled = event.response.status == "cancelled"
        playback.finish_response(response_id, cancelled=cancelled)
        if not cancelled:
            playback.finish()
        print(f"ASSISTANT: <response {event.response.status}>", flush=True)
    elif event.type == "output_audio_buffer.cleared":
        playback.clear()
    elif event.type == "error":
        renderer.clear_live_user_text()
        renderer.finish_live_assistant_text()
        print(f"ERROR: {event.error.type}: {event.error.message}", flush=True)
    else:
        renderer.clear_live_user_text()
        renderer.finish_live_assistant_text()
        print(f"EVENT: {event.type}", flush=True)


@dataclass(frozen=True)
class _ToolExecutionResult:
    call_id: str
    output: str
    create_response: bool


@dataclass
class _ToolResponseBatch:
    """Execution and delivery state scoped to one Realtime response.

    一个 response 可能包含多个 function call。只有在调用结果按 output index
    顺序发送完、且 response.done 确认成功后，才允许创建后续 response。
    """

    call_ids: set[str] = field(default_factory=set)
    execution_tasks: set[asyncio.Task[None]] = field(default_factory=set)
    output_indices: dict[str, int] = field(default_factory=dict)
    authoritative_call_ids: set[str] = field(default_factory=set)
    ordered_call_ids: list[str] = field(default_factory=list)
    delivered_call_ids: set[str] = field(default_factory=set)
    results: dict[str, _ToolExecutionResult] = field(default_factory=dict)
    pending_deliveries: int = 0
    completed: bool = False
    successful: bool = False
    cancelled: bool = False
    create_response: bool = False


class _ToolCoordinatorError(RuntimeError):
    """Raised when the server rejects a tool follow-up response permanently."""


class _ToolCallCoordinator:
    """协调工具调用，避免阻塞事件接收并避免 response 竞态。

    工具执行在独立 task 中进行；发送 function_call_output 则由单独的串行
    delivery 队列负责，从而保持同一响应内的 output 顺序。
    """

    def __init__(self, conn: Any, config: RealtimeAudioClientConfig) -> None:
        self._conn = conn
        self._executor = config.tool_executor
        self._tool_validators = _validate_tool_config(config.tools, self._executor)
        self._default_create_response = config.tool_response_create
        self._active_response_id: str | None = None
        self._pending_tool_flushes = 0
        self._queued_follow_ups = 0
        self._pending_create_id: str | None = None
        self._pending_create_follow_ups = 0
        self._pending_create_saw_response = False
        self._waiting_for_response_after_collision = False
        self._next_create_sequence = 0
        self._tool_batches: dict[str, _ToolResponseBatch] = {}
        self._tool_batch_order: list[str] = []
        self._background_tasks: set[asyncio.Task[Any]] = set()
        self._delivery_lock = asyncio.Lock()
        self._follow_up_lock = asyncio.Lock()
        self._failure: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        self._closing = False

    def handle_event(self, event: Any) -> None:
        """消费与工具相关的事件；其余音频/转写事件交给渲染器处理。"""

        if self._closing:
            return
        if event.type == "response.created":
            response = event.response
            self._active_response_id = getattr(response, "id", None)
            metadata = getattr(response, "metadata", None)
            create_id = metadata.get(_TOOL_CREATE_ID_METADATA_KEY) if isinstance(metadata, Mapping) else None
            if create_id and create_id == self._pending_create_id:
                self._pending_create_id = None
                self._pending_create_saw_response = False
                self._queued_follow_ups -= self._pending_create_follow_ups
                self._pending_create_follow_ups = 0
            elif self._pending_create_id is not None:
                self._pending_create_saw_response = True
        elif event.type == "response.output_item.added":
            item = getattr(event, "item", None)
            if getattr(item, "type", None) == "function_call":
                response_id = getattr(event, "response_id", None) or self._active_response_id
                output_index = getattr(event, "output_index", None)
                call_id = getattr(item, "call_id", None)
                # response.created 可能先于工具参数完成事件到达，因此这里只登记调用，
                # 真正执行和回传由后续的 arguments.done 驱动。
                if (
                    isinstance(response_id, str)
                    and response_id
                    and isinstance(output_index, int)
                    and isinstance(call_id, str)
                    and call_id
                ):
                    self._register_call_order(response_id, call_id, output_index)
        elif event.type == "response.function_call_arguments.done":
            response_id = getattr(event, "response_id", None) or self._active_response_id
            if isinstance(response_id, str) and response_id:
                self._schedule_call(response_id, event)
        elif event.type == "response.done":
            self._handle_response_done(event.response)
        elif event.type == "error":
            self._handle_error(event.error)
        elif event.type == "conversation.item.input_audio_transcription.completed":
            if self._waiting_for_response_after_collision:
                self._waiting_for_response_after_collision = False
                self._kick_follow_up()

    async def _execute_tool(self, call_id: str, name: Any, raw_arguments: Any) -> _ToolExecutionResult:
        """解析、校验并执行一次工具调用，将结果统一序列化为字符串。"""

        display_name = name if isinstance(name, str) and name else "<unnamed>"
        create_response = self._default_create_response
        try:
            if not isinstance(name, str) or name not in self._tool_validators:
                raise ValueError(f"unknown tool {display_name!r}")
            if not isinstance(raw_arguments, str):
                raise ValueError("arguments must be a JSON string")
            try:
                arguments = json.loads(raw_arguments)
            except json.JSONDecodeError as exc:
                raise ValueError(f"arguments are not valid JSON: {exc.msg}") from exc
            if not isinstance(arguments, dict):
                raise ValueError("arguments must decode to a JSON object")
            try:
                self._tool_validators[name].validate(arguments)
            except ValidationError as exc:
                raise ValueError(f"arguments do not match the declared schema: {exc.message}") from exc

            assert self._executor is not None
            pending_result = self._executor(name, arguments)
            if not inspect.isawaitable(pending_result):
                raise TypeError("tool_executor must return an awaitable")
            result = await pending_result
            if isinstance(result, ToolResult):
                if not isinstance(result.create_response, bool):
                    raise TypeError("ToolResult.create_response must be a boolean")
                create_response = result.create_response
                result = result.output
            if isinstance(result, str):
                output = result
            else:
                try:
                    output = json.dumps(result)
                except (TypeError, ValueError) as exc:
                    raise ValueError("tool result is not JSON serializable") from exc
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            # 工具失败也要返回可消费的 JSON 错误，避免服务器一直等待 output。
            message = f"{type(exc).__name__}: {exc}"
            log_exception(logger, f"Local tool {display_name} failed", exc, level=logging.DEBUG)
            print(f"TOOL ERROR: {display_name} call_id={call_id}: {message}", flush=True)
            output = json.dumps({"error": message})
            create_response = True
        return _ToolExecutionResult(call_id=call_id, output=output, create_response=create_response)

    def _handle_response_done(self, response: Any) -> None:
        """在响应结束时确认/取消批次，并触发必要的后续响应。"""

        response_id = getattr(response, "id", None)
        if response_id == self._active_response_id:
            self._active_response_id = None
        self._waiting_for_response_after_collision = False
        status = getattr(response, "status", None)
        if isinstance(response_id, str) and response_id and status == "completed":
            output = getattr(response, "output", None) or []
            calls = [
                (output_index, item)
                for output_index, item in enumerate(output)
                if getattr(item, "type", None) == "function_call"
            ]
            batch = self._tool_batches.get(response_id)
            if calls and batch is None:
                batch = self._batch(response_id)
            for output_index, call in calls:
                call_id = getattr(call, "call_id", None)
                if isinstance(call_id, str) and call_id:
                    self._register_call_order(response_id, call_id, output_index, kick=False)
                self._schedule_call(response_id, call)
            if batch is not None:
                batch.completed = True
                batch.successful = True
                self._kick_delivery()
        elif isinstance(response_id, str) and response_id:
            batch = self._tool_batches.get(response_id)
            if batch is not None:
                self._cancel_batch(response_id, batch)
        self._kick_follow_up()

    def _batch(self, response_id: str) -> _ToolResponseBatch:
        batch = self._tool_batches.get(response_id)
        if batch is None:
            batch = _ToolResponseBatch()
            self._tool_batches[response_id] = batch
            self._tool_batch_order.append(response_id)
        return batch

    def _register_call_order(
        self,
        response_id: str,
        call_id: str,
        output_index: int,
        *,
        kick: bool = True,
    ) -> None:
        batch = self._batch(response_id)
        if call_id in batch.delivered_call_ids:
            return
        batch.output_indices[call_id] = output_index
        batch.authoritative_call_ids.add(call_id)
        batch.ordered_call_ids = sorted(
            batch.authoritative_call_ids,
            key=batch.output_indices.__getitem__,
        )
        if kick:
            self._kick_delivery()

    def _schedule_call(self, response_id: str, call: Any) -> None:
        """去重并异步执行工具调用，执行完成后唤醒发送协程。"""

        call_id = getattr(call, "call_id", None)
        if not isinstance(call_id, str) or not call_id:
            print("TOOL ERROR: received a call without call_id; cannot return an output", flush=True)
            return
        batch = self._batch(response_id)
        if call_id in batch.call_ids:
            return
        batch.call_ids.add(call_id)
        output_index = getattr(call, "output_index", None)
        if isinstance(output_index, int):
            batch.output_indices.setdefault(call_id, output_index)
        batch.pending_deliveries += 1
        self._pending_tool_flushes += 1

        async def execute() -> None:
            result = await self._execute_tool(
                call_id,
                getattr(call, "name", None),
                getattr(call, "arguments", None),
            )
            if not batch.cancelled:
                batch.results[call_id] = result
                self._kick_delivery()

        execution = asyncio.create_task(execute())
        batch.execution_tasks.add(execution)
        execution.add_done_callback(batch.execution_tasks.discard)
        self._track(execution, report_errors=True)

    def _kick_delivery(self) -> None:
        task = asyncio.create_task(self._deliver_ready_results())
        self._track(task, report_errors=True)

    async def _deliver_ready_results(self) -> None:
        """按服务器声明的 output index 顺序发送已完成的工具结果。"""

        async with self._delivery_lock:
            while self._tool_batch_order:
                response_id = self._tool_batch_order[0]
                batch = self._tool_batches.get(response_id)
                if batch is None or batch.cancelled:
                    self._tool_batch_order.pop(0)
                    continue
                if batch.ordered_call_ids:
                    call_id = batch.ordered_call_ids[0]
                    result = batch.results.get(call_id)
                    if result is None:
                        return
                    await self._conn.send(
                        {
                            "type": "conversation.item.create",
                            "item": {
                                "type": "function_call_output",
                                "call_id": result.call_id,
                                "output": result.output,
                            },
                        }
                    )
                    # response.done can cancel the batch while send() is
                    # suspended on transport backpressure. _cancel_batch()
                    # already releases its counters, so do not account for the
                    # same delivery a second time when the send resumes.
                    if batch.cancelled:
                        return
                    # 只有真正发送成功后才减少计数；计数归零才可能触发 response.create。
                    batch.ordered_call_ids.pop(0)
                    batch.authoritative_call_ids.discard(call_id)
                    batch.delivered_call_ids.add(call_id)
                    batch.results.pop(call_id, None)
                    batch.pending_deliveries -= 1
                    self._pending_tool_flushes -= 1
                    batch.create_response = batch.create_response or result.create_response
                    continue
                if batch.completed and batch.pending_deliveries == 0:
                    self._finalize_batch_if_ready(response_id, batch)
                    continue
                return

    def _finalize_batch_if_ready(self, response_id: str, batch: _ToolResponseBatch) -> None:
        if not batch.completed or batch.pending_deliveries > 0:
            return
        if self._tool_batches.get(response_id) is batch:
            self._tool_batches.pop(response_id, None)
        if response_id in self._tool_batch_order:
            self._tool_batch_order.remove(response_id)
        if batch.successful and not batch.cancelled and batch.create_response:
            self._queued_follow_ups += 1
        self._kick_follow_up()

    def _cancel_batch(self, response_id: str, batch: _ToolResponseBatch) -> None:
        """取消未完成批次，并回收计数和后台执行任务。"""

        batch.cancelled = True
        self._pending_tool_flushes -= batch.pending_deliveries
        batch.pending_deliveries = 0
        self._tool_batches.pop(response_id, None)
        if response_id in self._tool_batch_order:
            self._tool_batch_order.remove(response_id)
        for task in tuple(batch.execution_tasks):
            task.cancel()
        self._kick_delivery()

    def _handle_error(self, error: Any) -> None:
        if getattr(error, "event_id", None) != self._pending_create_id:
            return
        is_collision = "conversation_already_has_active_response" in {
            getattr(error, "type", None),
            getattr(error, "code", None),
        }
        if self._pending_create_id is None:
            return
        rejected_create_id = self._pending_create_id
        self._pending_create_id = None
        self._pending_create_follow_ups = 0

        if not is_collision:
            self._pending_create_saw_response = False
            self._closing = True
            message = getattr(error, "message", None) or "unknown error"
            error_type = getattr(error, "type", None) or "unknown_error"
            code = getattr(error, "code", None)
            detail = f"{error_type}/{code}" if code else error_type
            if not self._failure.done():
                self._failure.set_exception(
                    _ToolCoordinatorError(
                        f"Tool follow-up response.create {rejected_create_id!r} was rejected ({detail}): {message}"
                    )
                )
            return

        response_already_finished = self._pending_create_saw_response and self._active_response_id is None
        self._waiting_for_response_after_collision = not self._pending_create_saw_response
        self._pending_create_saw_response = False
        if response_already_finished:
            self._kick_follow_up()

    def _kick_follow_up(self) -> None:
        task = asyncio.create_task(self._maybe_send_follow_up())
        self._track(task, report_errors=True)

    async def _maybe_send_follow_up(self) -> None:
        async with self._follow_up_lock:
            if (
                self._closing
                or self._pending_tool_flushes > 0
                or self._queued_follow_ups == 0
                or self._active_response_id is not None
                or self._pending_create_id is not None
                or self._waiting_for_response_after_collision
            ):
                return
            self._next_create_sequence += 1
            create_id = f"tool_{self._next_create_sequence}"
            self._pending_create_id = create_id
            self._pending_create_follow_ups = self._queued_follow_ups
            self._pending_create_saw_response = False
            try:
                await self._conn.send(
                    {
                        "event_id": create_id,
                        "type": "response.create",
                        "response": {"metadata": {_TOOL_CREATE_ID_METADATA_KEY: create_id}},
                    }
                )
            except BaseException:
                self._pending_create_id = None
                self._pending_create_follow_ups = 0
                self._pending_create_saw_response = False
                raise

    def _track(self, task: asyncio.Task[Any], *, report_errors: bool = False) -> None:
        self._background_tasks.add(task)

        def done(completed: asyncio.Task[Any]) -> None:
            self._background_tasks.discard(completed)
            if report_errors and not completed.cancelled():
                exception = completed.exception()
                if exception is not None and not self._failure.done():
                    self._failure.set_exception(exception)

        task.add_done_callback(done)

    async def wait_for_failure(self) -> None:
        await asyncio.shield(self._failure)

    async def close(self) -> None:
        self._closing = True
        tasks = list(self._background_tasks)
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        if not self._failure.done():
            self._failure.cancel()


async def _wait_for_stop(stop_event: Event) -> None:
    while not stop_event.is_set():
        await asyncio.to_thread(stop_event.wait, 0.1)


async def _run_audio_session(
    conn: Any,
    config: RealtimeAudioClientConfig,
    stop_event: Event,
) -> None:
    import sounddevice as sd
    from speech_to_speech.api.openai_realtime.aec3 import AEC3Processor

    aec3 = AEC3Processor(config.send_rate)

    wake_gate = None
    if config.wake_word:
        from speech_to_speech.api.openai_realtime.wake_word import OpenWakeWordDetector, WakeWordGate

        wake_gate = WakeWordGate(
            OpenWakeWordDetector(config.wake_word, config.send_rate),
            timeout_s=config.wake_word_timeout_s,
            sample_rate=config.send_rate,
        )

    mic_queue: Queue[bytes] = Queue(maxsize=128)
    playback = PlaybackBuffer(config.recv_rate, startup_buffer_ms=config.playback_buffer_ms)
    renderer = _FriendlyEventRenderer()
    tool_calls = _ToolCallCoordinator(conn, config)

    def callback_recv(outdata: Any, _frames: int, _time_info: Any, status: Any) -> None:
        if status:
            logger.warning("Speaker status: %s", status)
        playback.write(outdata)
        aec3.process_render(bytes(outdata))

    def callback_send(indata: Any, _frames: int, _time_info: Any, status: Any) -> None:
        if status:
            logger.warning("Microphone status: %s", status)
        try:
            mic_queue.put_nowait(aec3.process_capture(bytes(indata)))
        except Full:
            logger.debug("Dropping local microphone chunk because the send queue is full")

    async def send_audio() -> None:
        """从线程安全麦克风队列取 PCM，必要时经过唤醒词门控后上传。"""

        while not stop_event.is_set():
            try:
                chunk = await asyncio.to_thread(mic_queue.get, True, 0.1)
            except Empty:
                continue
            chunks = wake_gate.process(chunk) if wake_gate is not None else [chunk]
            if wake_gate is not None and wake_gate.consume_wake():
                await conn.send(
                    {
                        "type": "wake_word.detected",
                        "text": config.wake_ack,
                    }
                )
                print(f"WAKE ACK: {config.wake_ack}", flush=True)
            for gated_chunk in chunks:
                await conn.send(
                    {
                        "type": "input_audio_buffer.append",
                        "audio": base64.b64encode(gated_chunk).decode("ascii"),
                    }
                )

    async def receive_events() -> None:
        """持续读取服务器事件，并同时分发给工具协调器和 UI 渲染器。"""

        while not stop_event.is_set():
            event = await conn.recv()
            tool_calls.handle_event(event)
            handle_server_event(
                event,
                playback=playback,
                renderer=renderer,
                print_json=config.print_json,
            )

    opened_streams: list[Any] = []
    started_streams: list[Any] = []
    try:
        input_stream = sd.RawInputStream(
            samplerate=config.send_rate,
            channels=1,
            dtype="int16",
            blocksize=config.chunk_size,
            callback=callback_send,
            device=config.input_device,
        )
        opened_streams.append(input_stream)
        output_stream = sd.RawOutputStream(
            samplerate=config.recv_rate,
            channels=1,
            dtype="int16",
            blocksize=config.chunk_size,
            callback=callback_recv,
            device=config.output_device,
        )
        opened_streams.append(output_stream)

        for stream in opened_streams:
            stream.start()
            started_streams.append(stream)

        tasks = {
            asyncio.create_task(send_audio()),
            asyncio.create_task(receive_events()),
            asyncio.create_task(_wait_for_stop(stop_event)),
            asyncio.create_task(tool_calls.wait_for_failure()),
        }

        # 任一任务结束都意味着会话需要收尾：断开、停止信号或工具致命错误。
        done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        stop_event.set()
        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
        for task in done:
            if not task.cancelled() and task.exception() is not None:
                raise task.exception()  # type: ignore[misc]
    finally:
        stop_event.set()
        await tool_calls.close()
        if wake_gate is not None:
            wake_gate.close()
        aec3.close()
        renderer.clear_live_user_text()
        renderer.reset_assistant_text()
        for stream in reversed(started_streams):
            try:
                stream.stop()
            except Exception:
                logger.exception("Failed to stop local audio stream")
        for stream in reversed(opened_streams):
            try:
                stream.close()
            except Exception:
                logger.exception("Failed to close local audio stream")


async def listen_and_play_realtime(
    config: RealtimeAudioClientConfig,
    *,
    stop_event: Event | None = None,
) -> None:
    """通过 WebSocket 连接 Realtime 服务，失败时仅在初次连接阶段短暂重试。"""

    owned_stop_event = stop_event is None
    stop_event = stop_event or Event()
    client = _make_client(config)
    connected = False
    retry_started = time.monotonic()

    try:
        while not stop_event.is_set():
            try:
                async with client.realtime.connect(model=config.model) as conn:
                    connected = True
                    await conn.send(build_session_update(config))  # type: ignore[arg-type]
                    await _run_audio_session(
                        conn,
                        config,
                        stop_event,
                    )
                    return
            except asyncio.CancelledError:
                raise
            except _ToolCoordinatorError:
                raise
            except Exception as exc:
                if stop_event.is_set():
                    return
                if connected or time.monotonic() - retry_started >= config.connection_retry_timeout_s:
                    raise
                logger.debug("Realtime loopback server is not ready yet: %s", exc)
                await asyncio.sleep(0.1)
    finally:
        if owned_stop_event:
            stop_event.set()
        await client.close()


def run_realtime_audio_client(config: RealtimeAudioClientConfig) -> None:
    """Run the audio client until SIGINT, SIGTERM, disconnect, or an error."""

    stop_event = Event()
    previous_handlers: dict[signal.Signals, Any] = {}

    def request_shutdown(_sig: int, _frame: Any) -> None:
        stop_event.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        previous_handlers[sig] = signal.getsignal(sig)
        signal.signal(sig, request_shutdown)

    try:
        asyncio.run(listen_and_play_realtime(config, stop_event=stop_event))
    except KeyboardInterrupt:
        stop_event.set()
    finally:
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)


class RealtimeAudioClient:
    """ThreadManager handler that embeds the packaged client for ``local``."""

    def __init__(self, stop_event: Event, config: RealtimeAudioClientConfig) -> None:
        self.stop_event = stop_event
        self.config = config

    def run(self) -> None:
        try:
            asyncio.run(
                listen_and_play_realtime(
                    self.config,
                    stop_event=self.stop_event,
                )
            )
        except Exception:
            logger.exception("Local Realtime audio client stopped unexpectedly")
            self.stop_event.set()
