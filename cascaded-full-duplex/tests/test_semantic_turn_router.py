import json
import unittest
from unittest.mock import patch

from speech_to_speech.api.openai_realtime.audio_client import RealtimeAudioClientConfig
from speech_to_speech.api.openai_realtime.semantic_turn_router import (
    SemanticTurnDecision,
    SemanticTurnRouter,
)


class _Response:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None

    def read(self):
        return json.dumps(self.payload).encode()


class SemanticTurnRouterTests(unittest.TestCase):
    def test_empty_asr_waits_without_http(self):
        with patch("speech_to_speech.api.openai_realtime.semantic_turn_router.urlopen") as open_url:
            decision = SemanticTurnRouter("http://intent.test").decide(
                assistant_text="正在回答",
                user_asr=" ",
            )
        self.assertIs(decision, SemanticTurnDecision.WAIT)
        open_url.assert_not_called()

    def test_uses_trained_choice_and_expected_state_fields(self):
        response = _Response({"answers": {"action": {"choice": "yield"}}})
        with patch(
            "speech_to_speech.api.openai_realtime.semantic_turn_router.urlopen",
            return_value=response,
        ) as open_url:
            decision = SemanticTurnRouter("http://intent.test").decide(
                assistant_text="正在回答",
                user_asr="请停一下",
            )
        self.assertIs(decision, SemanticTurnDecision.YIELD)
        request = open_url.call_args.args[0]
        payload = json.loads(request.data)
        self.assertEqual(payload["state"], {"assistant_said": "正在回答", "user_said": "请停一下"})
        self.assertEqual(payload["questions"]["action"]["criteria"]["yield"], "让出话轮")

    def test_logs_input_and_full_output_with_matching_id(self):
        body = {"answers": {"action": {"choice": "yield", "confidence": 0.99}}}
        module = "speech_to_speech.api.openai_realtime.semantic_turn_router"
        with self.assertLogs(module, level="INFO") as logs, patch(
            f"{module}.urlopen", return_value=_Response(body)
        ) as open_url:
            SemanticTurnRouter("http://intent.test").decide(assistant_text="正在回答", user_asr="停\n一下")
        self.assertEqual(len(logs.records), 2)
        request_log, response_log = logs.records
        self.assertEqual(request_log.args[0], response_log.args[0])
        self.assertEqual(json.loads(request_log.args[2]), json.loads(open_url.call_args.args[0].data))
        self.assertEqual(json.loads(response_log.args[2]), body)
        self.assertIn("正在回答", request_log.getMessage())
        self.assertNotIn("\n", request_log.getMessage())
        self.assertGreaterEqual(response_log.args[1], 0)

    def test_logs_timeout_and_preserves_exception(self):
        module = "speech_to_speech.api.openai_realtime.semantic_turn_router"
        error = TimeoutError("timed out")
        with self.assertLogs(module, level="INFO") as logs, patch(
            f"{module}.urlopen", side_effect=error
        ), self.assertRaises(TimeoutError) as raised:
            SemanticTurnRouter("http://intent.test").decide(assistant_text="回答", user_asr="停")
        self.assertIs(raised.exception, error)
        self.assertEqual(len(logs.records), 2)
        self.assertEqual(logs.records[0].args[0], logs.records[1].args[0])
        self.assertIn("error_type=TimeoutError", logs.records[1].getMessage())

    def test_logs_invalid_choice_output_before_failure(self):
        module = "speech_to_speech.api.openai_realtime.semantic_turn_router"
        body = {"answers": {"action": {"choice": "invalid"}}}
        with self.assertLogs(module, level="INFO") as logs, patch(
            f"{module}.urlopen", return_value=_Response(body)
        ), self.assertRaisesRegex(ValueError, "Unexpected semantic turn choice"):
            SemanticTurnRouter("http://intent.test").decide(assistant_text="回答", user_asr="停")
        self.assertEqual(len(logs.records), 3)
        self.assertEqual(json.loads(logs.records[1].args[2]), body)
        self.assertEqual(len({record.args[0] for record in logs.records}), 1)

    def test_client_config_has_no_interruption_route(self):
        self.assertFalse(hasattr(RealtimeAudioClientConfig(), "interruption_route"))


if __name__ == "__main__":
    unittest.main()
