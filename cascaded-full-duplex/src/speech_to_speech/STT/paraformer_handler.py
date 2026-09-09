from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Iterator

import numpy as np
import torch
from rich.console import Console

from speech_to_speech.pipeline.handler_types import STTIn, STTOut
from speech_to_speech.pipeline.messages import PartialTranscription, Transcription
from speech_to_speech.STT.base_stt_handler import BaseSTTHandler

logger = logging.getLogger(__name__)

console = Console()


@dataclass
class _StreamingSession:
    cache: dict[str, Any] = field(default_factory=dict)
    text: str = ""


class ParaformerSTTHandler(BaseSTTHandler):
    """
    Handles the Speech To Text generation using a Paraformer model.
    The default for this model is set to Chinese.
    This model was contributed by @wuhongsheng.
    """

    def setup(
        self,
        model_name: str = "paraformer-zh-streaming",
        device: str = "cuda",
        gen_kwargs: dict[str, Any] | None = None,
    ) -> None:
        logger.info("Loading Paraformer STT model: %s", model_name)
        if len(model_name.split("/")) > 1:
            model_name = model_name.split("/")[-1]
        self.language = model_name.split("-")[1] if "-" in model_name else "zh"
        self.device = device
        self.streaming = model_name.endswith("-streaming")
        self._streaming_sessions: dict[tuple[str | None, int | None], _StreamingSession] = {}
        self.gen_kwargs = dict(gen_kwargs or {})
        try:
            from funasr import AutoModel
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "Paraformer STT requires the optional 'paraformer' extra. "
                "Install it with `pip install speech-to-speech[paraformer]`."
            ) from exc
        self.model = AutoModel(model=model_name, device=device)
        self.warmup()

    def warmup(self) -> None:
        logger.info(f"Warming up {self.__class__.__name__}")

        # 2 warmup steps for no compile or compile mode with CUDA graphs capture
        n_steps = 1
        dummy_input = np.array([0] * 512, dtype=np.float32)
        for _ in range(n_steps):
            if getattr(self, "streaming", False):
                _ = self.model.generate(
                    input=dummy_input,
                    cache={},
                    is_final=True,
                    chunk_size=[0, 10, 5],
                    encoder_chunk_look_back=4,
                    decoder_chunk_look_back=1,
                )
            else:
                _ = self.model.generate(dummy_input)[0]["text"].strip().replace(" ", "")

    def process(self, vad_audio: STTIn) -> Iterator[STTOut]:
        logger.debug("infering paraformer...")

        turn_key = (vad_audio.turn_id, vad_audio.turn_revision)
        if getattr(self, "streaming", False):
            sessions = getattr(self, "_streaming_sessions", {})
            session = sessions.setdefault(turn_key, _StreamingSession())
            is_final = vad_audio.mode == "final"
            result = self.model.generate(
                input=vad_audio.audio,
                cache=session.cache,
                is_final=is_final,
                chunk_size=[0, 10, 5],
                encoder_chunk_look_back=4,
                decoder_chunk_look_back=1,
                **self.gen_kwargs,
            )[0]
            chunk_text = str(result.get("text", "")).strip().replace(" ", "")
            if chunk_text.startswith(session.text):
                session.text = chunk_text
            elif not session.text:
                session.text = chunk_text
            elif chunk_text and not session.text.endswith(chunk_text):
                session.text += chunk_text
            pred_text = session.text
            if is_final:
                sessions.pop(turn_key, None)
        else:
            pred_text = self.model.generate(vad_audio.audio)[0]["text"].strip().replace(" ", "")
        # Same idea as ChatTTSHandler: MPS cache clear only on Apple Silicon.
        if self.device == "mps":
            torch.mps.empty_cache()

        logger.debug("finished paraformer inference")
        console.print(f"[yellow]USER: {pred_text}")

        if vad_audio.mode == "progressive":
            yield PartialTranscription(
                text=pred_text,
                turn_id=vad_audio.turn_id,
                turn_revision=vad_audio.turn_revision,
            )
        else:
            yield Transcription(
                text=pred_text,
                language_code=self.language,
                turn_id=vad_audio.turn_id,
                turn_revision=vad_audio.turn_revision,
                speech_stopped_at_s=vad_audio.created_at_s,
            )
