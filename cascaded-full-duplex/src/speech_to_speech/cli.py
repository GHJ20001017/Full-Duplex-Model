from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from typing import Any, Literal

from speech_to_speech.api.openai_realtime.audio_client import (
    RealtimeAudioClientConfig,
    load_realtime_tool_module,
    run_realtime_audio_client,
)
from speech_to_speech.pipeline.transcript_logging import (
    set_log_transcripts,
    warn_if_log_transcripts_enabled,
)

Command = Literal["serve", "talk", "local"]

# 兼容旧版入口：旧的 realtime 模式已映射到 serve，local 名称保持不变。
_LEGACY_MODE_COMMANDS: dict[str, Command] = {
    "realtime": "serve",
    "local": "local",
}


def _command_parser() -> argparse.ArgumentParser:
    """创建顶层命令解析器；子命令参数由各自的处理流程继续解析。"""

    parser = argparse.ArgumentParser(
        prog="speech-to-speech",
        description="Run or connect to the Realtime speech-to-speech pipeline.",
    )
    subparsers = parser.add_subparsers(dest="command", metavar="COMMAND")
    # 子解析器关闭自身 help，避免顶层解析器抢先消费子命令参数。
    subparsers.add_parser("serve", add_help=False, help="Run the Realtime pipeline server.")
    subparsers.add_parser("talk", add_help=False, help="Connect microphone and speakers to a Realtime URL.")
    subparsers.add_parser("local", add_help=False, help="Run the server and audio client together over loopback.")
    return parser


def _extract_legacy_mode(command_args: list[str], parser: argparse.ArgumentParser) -> tuple[str | None, list[str]]:
    """提取一个旧版 ``--mode``，其余参数原样交给对应命令处理。"""

    # 不能直接用 argparse 解析，否则会误读 talk/local 自己的选项。
    mode: str | None = None
    remaining: list[str] = []
    index = 0
    while index < len(command_args):
        argument = command_args[index]
        if argument == "--mode":
            if mode is not None:
                parser.error("--mode may only be specified once")
            if index + 1 == len(command_args) or command_args[index + 1].startswith("-"):
                parser.error("--mode requires a value: realtime or local")
            mode = command_args[index + 1]
            index += 2
            continue
        if argument.startswith("--mode="):
            if mode is not None:
                parser.error("--mode may only be specified once")
            mode = argument.partition("=")[2]
            if not mode:
                parser.error("--mode requires a value: realtime or local")
            index += 1
            continue
        remaining.append(argument)
        index += 1
    return mode, remaining


def parse_command(argv: Sequence[str] | None = None) -> tuple[Command, list[str]]:
    """拆分顶层命令和该命令拥有的参数，不提前解释子命令选项。"""

    command_args = list(sys.argv[1:] if argv is None else argv)
    parser = _command_parser()
    if not command_args:
        parser.error("a command is required: serve, talk, or local")
    if command_args[0] in {"-h", "--help"}:
        parser.print_help()
        raise SystemExit(0)
    if command_args[0] not in {"serve", "talk", "local"}:
        # 仅当首项不是新命令时尝试旧版参数，保持新 CLI 的错误信息清晰。
        legacy_mode, remaining = _extract_legacy_mode(command_args, parser)
        if legacy_mode is not None:
            legacy_command = _LEGACY_MODE_COMMANDS.get(legacy_mode)
            if legacy_command is None:
                parser.error(
                    f"--mode {legacy_mode!r} is no longer supported; only 'realtime' and 'local' remain "
                    "temporarily. Use 'speech-to-speech serve' or 'speech-to-speech local' instead."
                )
            print(
                f"Warning: '--mode {legacy_mode}' is deprecated and will stop working soon; "
                f"use 'speech-to-speech {legacy_command}' instead.",
                file=sys.stderr,
            )
            return legacy_command, remaining
    command = command_args[0]
    if command not in {"serve", "talk", "local"}:
        parser.error(f"unknown command {command!r}; choose serve, talk, or local")
    return command, command_args[1:]  # type: ignore[return-value]


def parse_talk_arguments(argv: Sequence[str]) -> RealtimeAudioClientConfig:
    """解析 talk 命令，并组装音频客户端所需的完整配置。"""

    defaults = RealtimeAudioClientConfig()
    parser = argparse.ArgumentParser(
        prog="speech-to-speech talk",
        description="Connect microphone and speakers to an OpenAI-compatible Realtime endpoint.",
    )
    # 参数默认值统一从客户端配置读取，避免 CLI 与底层默认值漂移。
    parser.add_argument(
        "--url",
        default=defaults.url,
        help="Full Realtime WebSocket endpoint, including /realtime.",
    )
    parser.add_argument("--model", default=defaults.model)
    parser.add_argument(
        "--api-key",
        default=defaults.api_key,
        help=(
            "Realtime API key. Defaults to OPENAI_API_KEY, or a harmless placeholder for an unauthenticated "
            "loopback endpoint."
        ),
    )
    parser.add_argument("--send-rate", type=int, default=defaults.send_rate)
    parser.add_argument("--recv-rate", type=int, default=defaults.recv_rate)
    parser.add_argument(
        "--playback-buffer-ms",
        type=float,
        default=defaults.playback_buffer_ms,
        help="Audio to buffer before playback starts, in milliseconds.",
    )
    parser.add_argument("--chunk-size", type=int, default=defaults.chunk_size)
    parser.add_argument("--input-device", type=int, default=defaults.input_device)
    parser.add_argument("--output-device", type=int, default=defaults.output_device)
    parser.add_argument("--instructions", default=defaults.instructions)
    parser.add_argument(
        "--wake-word",
        default=defaults.wake_word or "hey_jarvis",
        help="Enable local openWakeWord gating. Default: hey jarvis.",
    )
    parser.add_argument(
        "--wake-word-timeout",
        type=float,
        default=defaults.wake_word_timeout_s,
        help="Seconds allowed for speech after the wake word. Default: 300 (5 minutes).",
    )
    parser.add_argument(
        "--wake-ack",
        default=defaults.wake_ack,
        help="Fixed acknowledgement spoken after wake-word detection.",
    )
    parser.add_argument(
        "--tool-module",
        help="Importable module defining TOOLS and async execute_tool(name, arguments).",
    )
    parser.add_argument(
        "--no-ui",
        dest="ui",
        action="store_false",
        default=defaults.ui,
        help="Do not open the local conversation window; print to the terminal only.",
    )
    parser.add_argument(
        "--open-browser",
        dest="ui_open_browser",
        action="store_true",
        default=defaults.ui_open_browser,
        help="Open the conversation window in the default browser automatically (default).",
    )
    parser.add_argument(
        "--no-open-browser",
        dest="ui_open_browser",
        action="store_false",
        help="Serve the conversation window without opening a browser; open its URL manually.",
    )
    parser.add_argument(
        "--voice",
        default=defaults.voice,
        help="session.audio.output.voice (for example bm_fable, marin, or alloy).",
    )
    parser.add_argument("--print-json", action="store_true", default=defaults.print_json)
    parser.add_argument(
        "--block-mic-during-playback",
        action="store_true",
        default=defaults.block_mic_during_playback,
        help="Deprecated compatibility option; microphone capture is always processed by AEC3.",
    )
    parser.add_argument(
        "--log-transcripts",
        "--log_transcripts",
        dest="log_transcripts",
        action="store_true",
        default=defaults.log_transcripts,
        help="Write full Realtime transcript and tool error details to application logs.",
    )
    parser.add_argument(
        "--connection-retry-timeout",
        type=float,
        default=defaults.connection_retry_timeout_s,
        help="Seconds to wait for the Realtime endpoint to become available.",
    )
    namespace = parser.parse_args(list(argv))
    # 工具模块是可选的；加载函数会同时校验 TOOLS 和 execute_tool 契约。
    tools: list[dict[str, Any]] = []
    tool_executor = None
    tool_response_create = defaults.tool_response_create
    if namespace.tool_module:
        tools, tool_executor, tool_response_create = load_realtime_tool_module(namespace.tool_module)
    # 显式逐项映射，确保 argparse 命名与客户端配置命名的边界可见。
    return RealtimeAudioClientConfig(
        url=namespace.url,
        model=namespace.model,
        api_key=namespace.api_key,
        send_rate=namespace.send_rate,
        recv_rate=namespace.recv_rate,
        playback_buffer_ms=namespace.playback_buffer_ms,
        chunk_size=namespace.chunk_size,
        input_device=namespace.input_device,
        output_device=namespace.output_device,
        instructions=namespace.instructions,
        wake_word=namespace.wake_word,
        wake_word_timeout_s=namespace.wake_word_timeout,
        wake_ack=namespace.wake_ack,
        voice=namespace.voice,
        print_json=namespace.print_json,
        block_mic_during_playback=namespace.block_mic_during_playback,
        log_transcripts=namespace.log_transcripts,
        connection_retry_timeout_s=namespace.connection_retry_timeout,
        tools=tools,
        tool_executor=tool_executor,
        tool_response_create=tool_response_create,
        ui=namespace.ui,
        ui_open_browser=namespace.ui_open_browser,
    )


def main() -> None:
    """执行 CLI：talk 直接运行客户端，其余命令交给 pipeline。"""

    command, command_args = parse_command()
    if command == "talk":
        # 日志开关必须在启动客户端前设置，保证整个会话都使用同一策略。
        config = parse_talk_arguments(command_args)
        set_log_transcripts(config.log_transcripts)
        warn_if_log_transcripts_enabled()
        run_realtime_audio_client(config)
        return

    from speech_to_speech.s2s_pipeline import run_pipeline_command

    run_pipeline_command(command, command_args)


if __name__ == "__main__":
    main()
