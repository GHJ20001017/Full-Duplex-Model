from dataclasses import dataclass, field


@dataclass
class FunASRNanoSTTHandlerArguments:
    fun_asr_nano_stt_model_name: str = field(
        default="FunAudioLLM/Fun-ASR-Nano-2512",
        metadata={"help": "Fun-ASR-Nano model ID or local model directory (loaded through FunASR/PyTorch)."},
    )
    fun_asr_nano_stt_device: str = field(default="cuda", metadata={"help": "Fun-ASR-Nano inference device."})
    fun_asr_nano_stt_hotwords_file: str | None = field(
        default=None,
        metadata={"help": "Required for Nano: UTF-8 hotwords file, one phrase per line. Loaded once at startup."},
    )
    fun_asr_nano_stt_hub: str = field(
        default="hf",
        metadata={"help": "Model hub: hf (Hugging Face) or ms (ModelScope).", "choices": ["hf", "ms"]},
    )
    fun_asr_nano_stt_language: str = field(
        default="中文",
        metadata={"help": "Recognition language.", "choices": ["中文", "英文", "日文"]},
    )
    fun_asr_nano_stt_itn: bool = field(default=True, metadata={"help": "Enable inverse text normalization."})
