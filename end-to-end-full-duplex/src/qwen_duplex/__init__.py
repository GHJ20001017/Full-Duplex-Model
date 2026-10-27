"""Qwen2.5-Omni native action vocabulary and interleaved Thinker sessions.

Action selection is produced by the Thinker LM head. Supervised Talker training
is available for paired text/speech codes; no pretrained action policy,
waveform target encoder, or real-time playback runtime is provided here.
"""

from .audio import PrefixAudioEncoder
from .sequence import TrainingWindow, prepare_interleaved_inputs
from .session import ActionEvent, ThinkerSession
from .speech import SpeechTarget, SpeechTrainingOutput, forward_with_speech_loss
from .tokens import (
    ACTION_TOKENS,
    extend_action_tokens,
    load_duplex_checkpoint,
    save_duplex_checkpoint,
    validate_action_tokens,
)

__all__ = [
    "ACTION_TOKENS", "ActionEvent", "PrefixAudioEncoder", "ThinkerSession", "TrainingWindow",
    "extend_action_tokens", "load_duplex_checkpoint", "prepare_interleaved_inputs",
    "SpeechTarget", "SpeechTrainingOutput", "forward_with_speech_loss",
    "save_duplex_checkpoint", "validate_action_tokens",
]
