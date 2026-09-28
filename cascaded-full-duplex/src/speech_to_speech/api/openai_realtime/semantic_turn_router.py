from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass
from enum import Enum
from urllib.request import Request, urlopen
from uuid import uuid4

logger = logging.getLogger(__name__)


class SemanticTurnDecision(str, Enum):
    CONTINUE = "continue"
    YIELD = "yield"
    WAIT = "wait"


@dataclass(frozen=True)
class SemanticTurnRouter:
    """Synchronous client for the trained RLCD intent service.

    The Realtime pipeline calls this from its worker thread, not the asyncio
    WebSocket receive loop. The model's trained ``choice`` is authoritative;
    probabilities and entropy confidence are retained only for diagnostics.
    """

    url: str
    timeout_s: float = 0.12

    def decide(self, *, assistant_text: str, user_asr: str) -> SemanticTurnDecision:
        if not user_asr.strip():
            return SemanticTurnDecision.WAIT
        payload = {
            "state": {
                "assistant_said": assistant_text,
                "user_said": user_asr,
            },
            "questions": {
                "action": {
                    "type": "choice",
                    "instructions": "助手接下来应该做什么？",
                    "criteria": {
                        "continue": "继续说",
                        "yield": "让出话轮",
                        "wait": "等待",
                    },
                }
            },
        }
        request = Request(
            self.url,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        request_id = uuid4().hex
        logger.info(
            "Semantic request id=%s timeout_s=%s input=%s",
            request_id,
            self.timeout_s,
            json.dumps(payload, ensure_ascii=False),
        )
        started = time.perf_counter()
        try:
            with urlopen(request, timeout=self.timeout_s) as response:
                body = json.load(response)
            logger.info(
                "Semantic response id=%s elapsed_ms=%.2f output=%s",
                request_id,
                (time.perf_counter() - started) * 1000,
                json.dumps(body, ensure_ascii=False),
            )
            choice = body["answers"]["action"]["choice"]
            try:
                return SemanticTurnDecision(choice)
            except ValueError as exc:
                raise ValueError(f"Unexpected semantic turn choice: {choice!r}") from exc
        except Exception as exc:
            logger.warning(
                "Semantic failure id=%s elapsed_ms=%.2f error_type=%s error=%s",
                request_id,
                (time.perf_counter() - started) * 1000,
                type(exc).__name__,
                json.dumps(str(exc), ensure_ascii=False),
            )
            raise


def router_from_environment() -> SemanticTurnRouter | None:
    url = os.environ.get("S2S_SEMANTIC_TURN_URL", "").strip()
    if not url:
        return None
    timeout = float(os.environ.get("S2S_SEMANTIC_TURN_TIMEOUT_S", "0.12"))
    return SemanticTurnRouter(url=url, timeout_s=timeout)
