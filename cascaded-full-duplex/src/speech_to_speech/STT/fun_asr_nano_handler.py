from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Iterator

import torch

from speech_to_speech.pipeline.handler_types import STTIn, STTOut
from speech_to_speech.pipeline.messages import PartialTranscription, Transcription
from speech_to_speech.STT.base_stt_handler import BaseSTTHandler

logger = logging.getLogger(__name__)


class FunASRNanoSTTHandler(BaseSTTHandler):
    """PyTorch/FunASR inference on cumulative VAD snapshots, without vLLM."""

    def setup(
        self,
        model_name: str = "FunAudioLLM/Fun-ASR-Nano-2512",
        device: str = "cuda",
        hotwords_file: str | None = None,
        hub: str = "hf",
        language: str = "中文",
        itn: bool = True,
        gen_kwargs: dict[str, Any] | None = None,
    ) -> None:
        if gen_kwargs:
            raise ValueError("Fun-ASR-Nano uses explicit language, itn and hotwords_file options, not gen_kwargs.")
        if not hotwords_file:
            raise ValueError("Fun-ASR-Nano requires --fun_asr_nano_stt_hotwords_file (UTF-8, one phrase per line).")
        path = Path(hotwords_file).expanduser()
        # utf-8-sig accepts both ordinary UTF-8 and files with a BOM.
        lines = path.read_text(encoding="utf-8-sig").splitlines()
        self.hotwords = list(
            dict.fromkeys(line.strip() for line in lines if line.strip() and not line.lstrip().startswith("#"))
        )
        if not self.hotwords:
            raise ValueError(f"Fun-ASR-Nano hotwords file contains no phrases: {path}")
        languages = {"中文": "zh", "英文": "en", "日文": "ja"}
        if language not in languages:
            raise ValueError("Fun-ASR-Nano language must be 中文, 英文, or 日文.")
        self.language = language
        self.language_code = languages[language]
        self.itn = itn
        try:
            from funasr import AutoModel
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "Fun-ASR-Nano STT requires the optional 'fun-asr-nano' extra. "
                "Install it with `pip install speech-to-speech[fun-asr-nano]`."
            ) from exc
        logger.info("Loading Fun-ASR-Nano STT model: %s (%d hotwords)", model_name, len(self.hotwords))
        self.model = AutoModel(model=model_name, device=device, hub=hub, trust_remote_code=True)

    def process(self, vad_audio: STTIn) -> Iterator[STTOut]:
        # VAD supplies the entire utterance so far, NOT Paraformer delta chunks.
        # Re-decode each snapshot independently: partials may be dropped/revised
        # and a reopened turn must never inherit a stale text prefix or cache.
        text = ""
        if vad_audio.audio.size:
            # Nano's ChatML adapter accepts Tensor audio, not NumPy arrays.
            # Copy the snapshot so inference cannot mutate the VAD-owned buffer.
            audio = torch.tensor(vad_audio.audio, dtype=torch.float32, device="cpu")
            result = self.model.generate(
                input=[audio],
                cache={},
                batch_size=1,
                hotwords=list(self.hotwords),
                language=self.language,
                itn=self.itn,
            )
            if not result or not isinstance(result[0], dict) or not isinstance(result[0].get("text"), str):
                raise RuntimeError("Fun-ASR-Nano returned no valid transcription text.")
            text = result[0]["text"].strip()
        if vad_audio.mode == "progressive":
            yield PartialTranscription(
                text=text,
                turn_id=vad_audio.turn_id,
                turn_revision=vad_audio.turn_revision,
            )
        else:
            yield Transcription(
                text=text,
                language_code=self.language_code,
                turn_id=vad_audio.turn_id,
                turn_revision=vad_audio.turn_revision,
                speech_stopped_at_s=vad_audio.created_at_s,
            )
