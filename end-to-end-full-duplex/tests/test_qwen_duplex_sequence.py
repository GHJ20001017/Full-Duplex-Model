import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from transformers import WhisperFeatureExtractor

from qwen_duplex.audio import PrefixAudioEncoder
from qwen_duplex.sequence import TrainingWindow, prepare_interleaved_inputs
from qwen_duplex.session import ThinkerSession
from qwen_duplex.tokens import extend_action_tokens
from qwen_duplex_helpers import tiny_omni


class DuplexSequenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def setUp(self):
        torch.manual_seed(19)
        self.model, self.tokenizer = tiny_omni()
        self.actions = extend_action_tokens(self.model, self.tokenizer)
        self.session = ThinkerSession(self.model, self.tokenizer, record_sequence=True)
        self.prefix = [1, 3, 9]
        self.session.prefill(self.prefix)

    def choose(self, action):
        # Deterministic logits fixture tests protocol routing, NOT learned action quality.
        self.session.logits = torch.zeros_like(self.session.logits)
        self.session.logits[0, self.actions[action]] = 100
        return self.session.decide()

    def test_interleaved_cache_matches_full_replay_and_training_layout(self):
        first, second, third = (torch.randn(n, 32) for n in (5, 7, 4))
        self.session.append_audio(first, end_ms=1000)
        event = self.choose("<|wait|>")
        self.assertEqual((event.token, event.audio_end_ms), ("<|wait|>", 1000))
        cache = self.session.cache
        self.session.append_audio(second, end_ms=2000)
        self.choose("<|speak|>")
        # Force an ordinary text token so one bounded burst can be compared exactly.
        self.session.logits = torch.zeros_like(self.session.logits)
        self.session.logits[0, 3] = 100
        self.assertEqual(self.session.generate_text(max_new_tokens=1), [3])
        self.session.append_audio(third, end_ms=3000)
        self.choose("<|interrupt|>")
        self.assertIs(self.session.cache, cache)
        self.assertEqual(cache.get_seq_length(), self.session.length)
        for left, right in zip(self.session.events, self.session.events[1:]):
            self.assertEqual(left["end"], right["start"])
        batch = prepare_interleaved_inputs(self.model, self.tokenizer, self.prefix, [
            TrainingWindow(first, "<|wait|>"), TrainingWindow(second, "<|speak|>", (3,)),
            TrainingWindow(third, "<|interrupt|>"),
        ])
        torch.testing.assert_close(batch["inputs_embeds"], self.session.replay_inputs()["inputs_embeds"])
        with torch.no_grad():
            replay = self.model(**self.session.replay_inputs())
        torch.testing.assert_close(replay.logits[:, -1, :], self.session.logits, atol=1e-6, rtol=1e-5)
        supervised = batch["labels"][batch["labels"] != -100].tolist()
        self.assertEqual(supervised, [self.actions["<|wait|>"], self.actions["<|speak|>"], 3,
                                      self.actions["<|interrupt|>"]])
        output = self.model(**batch)
        self.assertTrue(torch.isfinite(output.loss))
        output.loss.backward()
        for action in ("<|wait|>", "<|speak|>", "<|interrupt|>"):
            self.assertGreater(self.model.lm_head.weight.grad[self.actions[action]].abs().sum().item(), 0)

    def test_decide_uses_model_head_and_is_once_per_window(self):
        self.session.append_audio(torch.randn(3, 32), end_ms=1000)
        expected = max(self.actions, key=lambda token: self.session.logits[0, self.actions[token]].item())
        self.assertEqual(self.session.decide().token, expected)
        length = self.session.length
        with self.assertRaises(ValueError):
            self.session.decide()
        self.assertEqual(self.session.length, length)

    def test_end_speech_and_eos_do_not_end_session(self):
        for stop in (self.actions["<|end_speech|>"], self.tokenizer.eos_token_id):
            self.session.reset()
            self.session.prefill(self.prefix)
            self.session.append_audio(torch.randn(3, 32), end_ms=1000)
            self.choose("<|speak|>")
            self.session.logits = torch.zeros_like(self.session.logits)
            self.session.logits[0, stop] = 100
            self.assertEqual(self.session.generate_text(max_new_tokens=3), [stop])
            self.assertEqual(self.session.generate_text(), [])
            cache = self.session.cache
            self.session.append_audio(torch.randn(2, 32), end_ms=2000)
            self.choose("<|wait|>")
            self.assertIs(self.session.cache, cache)

    def test_new_audio_actually_changes_logits_with_same_text_history(self):
        sessions = [ThinkerSession(self.model, self.tokenizer) for _ in range(2)]
        first = torch.randn(4, 32)
        for session in sessions:
            session.prefill(self.prefix)
            session.append_audio(first, end_ms=1000)
            session.logits = torch.zeros_like(session.logits)
            session.logits[0, self.actions["<|wait|>"]] = 100
            session.decide()
        torch.testing.assert_close(sessions[0].logits, sessions[1].logits)
        for index, session in enumerate(sessions):
            session.append_audio(torch.randn(4, 32) * (index + 1), end_ms=2000)
        self.assertEqual(sessions[0].length, sessions[1].length)
        self.assertGreater((sessions[0].logits - sessions[1].logits).abs().max().item(), 1e-5)

    def test_validation_does_not_mutate_cache(self):
        baseline = self.session.length
        for end_ms in (0, 2000, 1000.0, True):
            with self.assertRaises(ValueError):
                self.session.append_audio(torch.randn(2, 32), end_ms=end_ms)
        for hidden in (torch.empty(0, 32), torch.randn(2, 31), torch.full((2, 32), float("nan"))):
            with self.assertRaises(ValueError):
                self.session.append_audio(hidden, end_ms=1000)
        with self.assertRaises(ValueError):
            self.session.generate_text()
        self.assertEqual(self.session.length, baseline)
        self.session.append_audio(torch.randn(3, 32), end_ms=1000)
        with self.assertRaises(ValueError):
            self.session.append_audio(torch.randn(3, 32), end_ms=2000)
        self.choose("<|wait|>")
        with self.assertRaises(ValueError):
            self.session.generate_text()
        with self.assertRaises(ValueError):
            self.session.prefill(self.prefix)

    def test_budget_failure_is_non_mutating_and_reset_restarts_clock(self):
        session = ThinkerSession(self.model, self.tokenizer, max_context_tokens=8)
        session.prefill(self.prefix)
        with self.assertRaises(ValueError):
            session.append_audio(torch.randn(3, 32), end_ms=1000)
        self.assertEqual(session.cache.get_seq_length(), len(self.prefix))
        self.assertFalse(session.pending_action)
        self.assertFalse(session.broken)
        session.reset()
        self.assertIsNone(session.cache)
        self.assertEqual(session.audio_end_ms, 0)
        session.prefill(self.prefix)
        session.append_audio(torch.randn(2, 32), end_ms=1000)
        session.decide()

    def test_native_failure_invalidates_cache_until_reset(self):
        with patch.object(self.model.lm_head, "forward", side_effect=RuntimeError("failed after partial write")):
            with self.assertRaises(RuntimeError):
                self.session.append_audio(torch.randn(3, 32), end_ms=1000)
        self.assertGreater(self.session.cache.get_seq_length(), self.session.length)
        self.assertTrue(self.session.broken)
        with self.assertRaises(RuntimeError):
            self.session.append_audio(torch.randn(3, 32), end_ms=1000)
        self.session.reset()
        self.session.prefill(self.prefix)
        self.assertFalse(self.session.broken)

    def test_text_phase_masks_actions_and_audio_specials(self):
        self.session.append_audio(torch.randn(3, 32), end_ms=1000)
        self.choose("<|speak|>")
        self.session.logits = torch.zeros_like(self.session.logits)
        self.session.logits[0, self.actions["<|wait|>"]] = 100
        self.session.logits[0, self.model.config.audio_start_token_id] = 90
        self.session.logits[0, 3] = 80
        self.assertEqual(self.session.generate_text(), [3])

    def test_training_rejects_inconsistent_targets(self):
        for window in (TrainingWindow(torch.randn(3, 32), "unknown"),
                       TrainingWindow(torch.randn(3, 32), "<|wait|>", (3,)),
                       TrainingWindow(torch.randn(3, 32), "<|speak|>", (self.actions["<|interrupt|>"],))):
            with self.assertRaises(ValueError):
                prepare_interleaved_inputs(self.model, self.tokenizer, self.prefix, [window])


class NativeAudioTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def setUp(self):
        self.model, self.tokenizer = tiny_omni()
        extend_action_tokens(self.model, self.tokenizer)
        self.extractor = WhisperFeatureExtractor(feature_size=8)
        self.processor = SimpleNamespace(feature_extractor=self.extractor)

    def test_actual_native_waveform_prefix_encoding_and_cache_append(self):
        session = ThinkerSession(self.model, self.tokenizer, processor=self.processor)
        session.prefill([1, 3])
        with patch.object(self.model, "get_audio_features", wraps=self.model.get_audio_features) as encode:
            for step in (1, 2):
                session.push_audio(np.random.default_rng(step).normal(0, .1, 16000).astype(np.float32), sampling_rate=16000)
                # Only the arrived prefix is exposed as valid Mel positions, not future samples.
                mask = encode.call_args.kwargs["feature_attention_mask"]
                self.assertEqual(mask.sum().item(), step * 100)
                self.assertEqual(len(session.encoder.waveform), step * 16000)
                self.assertEqual(session.encoder.committed, step * 25)
                event = session.decide()
                self.assertEqual(event.audio_end_ms, step * 1000)
        cache = session.cache
        self.assertEqual(session.length, 2 + 2 * (25 + 2 + 1))
        self.assertEqual(cache.get_seq_length(), session.length)
        session.reset()
        self.assertEqual(len(session.encoder.waveform), 0)
        self.assertEqual(session.encoder.committed, 0)

    def test_bad_waveform_and_budget_preserve_encoder_state(self):
        encoder = PrefixAudioEncoder(self.model, self.processor, max_audio_seconds=1)
        for pcm, rate in [(np.zeros(16000, np.float32), 8000), (np.zeros((16000, 2), np.float32), 16000),
                          (np.zeros(16000, np.int16), 16000), (np.array([np.nan]), 16000),
                          (np.zeros(16001, np.float32), 16000)]:
            with self.assertRaises(ValueError):
                encoder.push(pcm, sampling_rate=rate)
            self.assertEqual(len(encoder.waveform), 0)
        session = ThinkerSession(self.model, self.tokenizer, processor=self.processor)
        session.prefill([1, 3])
        with self.assertRaises(ValueError):
            session.push_audio(np.zeros(8000, np.float32), sampling_rate=16000)
        self.assertEqual(session.encoder.committed, 0)


if __name__ == "__main__":
    unittest.main()
