from dataclasses import dataclass, field
from typing import Optional


@dataclass
class LocalAudioArguments:
    local_audio_tool_module: Optional[str] = field(
        default=None,
        metadata={
            "help": "Importable module defining TOOLS and async execute_tool(name, arguments).",
            "aliases": ["--tool-module"],
        },
    )
    local_audio_input_device: Optional[int] = field(
        default=None,
        metadata={"help": "Optional sounddevice input device index used by the local command."},
    )
    local_audio_output_device: Optional[int] = field(
        default=None,
        metadata={"help": "Optional sounddevice output device index used by the local command."},
    )
    local_audio_chunk_size: int = field(
        default=1024,
        metadata={"help": "Microphone and speaker callback block size in samples. Default is 1024."},
    )
    local_audio_playback_buffer_ms: Optional[float] = field(
        default=None,
        metadata={
            "help": (
                "Audio to buffer before local playback starts, in milliseconds. "
                "Defaults to 196 for OpenAI-compatible TTS and 0 otherwise."
            ),
            "aliases": ["--playback-buffer-ms"],
        },
    )
    local_audio_block_mic_during_playback: bool = field(
        default=False,
        metadata={
            "help": "Deprecated compatibility option; microphone capture is always processed by AEC3."
        },
    )
    local_audio_print_json: bool = field(
        default=False,
        metadata={"help": "Print raw Realtime events received by the packaged local audio client."},
    )
    local_audio_wake_word: Optional[str] = field(
        default="hey_jarvis",
        metadata={"help": "Enable local openWakeWord gating. Default: hey jarvis.", "aliases": ["--wake-word"]},
    )
    local_audio_wake_word_timeout_s: float = field(
        default=300.0,
        metadata={"help": "Seconds allowed for speech after the wake word. Default: 300 (5 minutes)."},
    )
    local_audio_ui: bool = field(
        default=True,
        metadata={
            "help": "Open a local browser window showing the user/assistant conversation. On by default.",
            "aliases": ["--ui"],
        },
    )
