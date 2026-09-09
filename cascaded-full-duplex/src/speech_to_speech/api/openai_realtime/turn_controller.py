from __future__ import annotations

import re
from enum import Enum


class TurnDecision(str, Enum):
    """Decision for a possible user interruption while the assistant speaks."""

    NORMAL_TURN = "normal_turn"
    WAIT = "wait"
    BACKCHANNEL = "backchannel"
    CANCEL = "cancel"


class TurnController:
    """Conservative semantic gate for barge-in decisions.

    VAD only says that speech-like audio exists. This controller waits for a
    stable-enough transcript before turning that candidate into a hard cancel.
    """

    min_speech_ms: int = 280
    min_meaningful_chars: int = 3

    _backchannels = frozenset(
        {
            "嗯",
            "嗯嗯",
            "啊",
            "哦",
            "噢",
            "对",
            "好的",
            "好",
            "知道了",
            "继续",
            "yes",
            "yeah",
            "ok",
            "okay",
        }
    )
    _explicit_interruptions = (
        "等一下",
        "等一等",
        "停一下",
        "停",
        "暂停",
        "不是这个意思",
        "我换个问题",
        "打断一下",
        "先别说",
    )

    @staticmethod
    def normalize(transcript: str) -> str:
        return re.sub(r"[\s，。！？、,.!?；;：:‘’“”\"'（）()\[\]【】]+", "", transcript.casefold())

    def decide(
        self,
        transcript: str,
        *,
        speech_duration_ms: float = 0.0,
        assistant_is_speaking: bool = True,
        final: bool = False,
    ) -> TurnDecision:
        """Return a decision without mutating connection or response state."""
        if not assistant_is_speaking:
            return TurnDecision.NORMAL_TURN

        text = self.normalize(transcript)
        if not text:
            return TurnDecision.WAIT

        if any(text.startswith(phrase) for phrase in self._explicit_interruptions):
            return TurnDecision.CANCEL

        if text in self._backchannels:
            return TurnDecision.BACKCHANNEL

        # A final transcript is authoritative. If it is not a backchannel,
        # treat it as a real user turn even when partial STT was too short.
        if final:
            return TurnDecision.CANCEL

        if speech_duration_ms < self.min_speech_ms:
            return TurnDecision.WAIT

        # A question/action phrase is meaningful even when the first partial
        # hypothesis is shorter than the general character threshold.
        question_or_action = any(
            marker in text
            for marker in ("吗", "什么", "怎么", "哪里", "哪个", "天气", "帮我", "请", "我想", "我要")
        )
        if len(text) >= self.min_meaningful_chars or question_or_action:
            return TurnDecision.CANCEL
        return TurnDecision.WAIT
