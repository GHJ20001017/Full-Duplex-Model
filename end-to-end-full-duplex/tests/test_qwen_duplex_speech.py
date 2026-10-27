"""Native, random-weight Thinker/Talker speech training contract tests (offline)."""

import math
import unittest
from contextlib import contextmanager
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from transformers import Qwen2_5OmniTalkerConfig, Qwen2_5OmniTalkerForConditionalGeneration

from qwen_duplex.sequence import TrainingWindow
from qwen_duplex.speech import SpeechTarget, forward_with_speech_loss
from qwen_duplex.tokens import extend_action_tokens
from qwen_duplex_helpers import tiny_omni


class DuplexSpeechTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def setUp(self):
        torch.manual_seed(73)
        thinker, self.tokenizer = tiny_omni()
        self.actions = extend_action_tokens(thinker, self.tokenizer)
        config = Qwen2_5OmniTalkerConfig(
            vocab_size=32, embedding_size=32, hidden_size=32, intermediate_size=64,
            num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2,
            head_dim=8, max_position_embeddings=512,
            rope_scaling={"rope_type": "default", "mrope_section": [1, 1, 2]},
            tts_text_start_token_id=28, tts_text_end_token_id=29,
            tts_text_pad_token_id=30, tts_codec_start_token_id=28,
            tts_codec_end_token_id=29, tts_codec_pad_token_id=30,
            tts_codec_mask_token_id=31, audio_token_index=6,
            audio_start_token_id=4, audio_end_token_id=5,
            image_token_index=7, video_token_index=8,
        )
        config._attn_implementation = "eager"
        talker = Qwen2_5OmniTalkerForConditionalGeneration(config).eval()
        # Never instantiate the production composite (which also builds Token2Wav).
        self.model = SimpleNamespace(
            thinker=thinker, talker=talker,
            config=SimpleNamespace(thinker_config=thinker.config),
            speaker_map={"Chelsie": {"bos_token": 27}, "Ethan": {"bos_token": 26}},
        )
        self.prefix = [1, 3, 9]

    def window(self, action="<|speak|>", text=None, audio_length=2):
        if text is None:
            text = (3, 10, self.actions["<|end_speech|>"])
        return TrainingWindow(torch.randn(audio_length, 32), action, text)

    def run_speech(self, windows, targets, **kwargs):
        return forward_with_speech_loss(
            self.model, self.tokenizer, self.prefix, windows, targets, **kwargs,
        )

    @contextmanager
    def capture_talker(self):
        calls = []

        def capture(module, args, kwargs):
            self.assertEqual(args, ())
            # Snapshot before native forward; in-place special insertion must not
            # be able to hide a faulty bridge by mutating the captured input.
            calls.append({key: value.detach().clone() if torch.is_tensor(value) else value
                          for key, value in kwargs.items()})

        hook = self.model.talker.register_forward_pre_hook(capture, with_kwargs=True)
        try:
            yield calls
        finally:
            hook.remove()

    def independent_thinker_layout(self, windows):
        """Build an oracle without calling the production serialization helper."""
        thinker = self.model.thinker
        embed = thinker.get_input_embeddings()

        def tokens(ids):
            return embed(torch.tensor([ids], dtype=torch.long))

        pieces = [tokens(self.prefix)]
        audio_positions = []
        body_starts = []
        length = len(self.prefix)
        for window in windows:
            pieces.append(tokens([thinker.config.audio_start_token_id]))
            pieces.append(window.audio_embeddings.unsqueeze(0))
            audio_positions.extend(range(length + 1, length + 1 + len(window.audio_embeddings)))
            pieces.append(tokens([thinker.config.audio_end_token_id, self.actions[window.action]]))
            length += len(window.audio_embeddings) + 3
            body_starts.append(length)
            if window.text_ids:
                pieces.append(tokens(list(window.text_ids)))
                length += len(window.text_ids)
        return torch.cat(pieces, dim=1), audio_positions, body_starts

    def expected_talker_input(self, windows, window_index, codes, hidden, speaker="Chelsie"):
        raw, audio_positions, starts = self.independent_thinker_layout(windows)
        prompt_length = starts[window_index]
        # Audio start/end markers remain text embeddings; only actual audio
        # positions drop their raw acoustic embeddings on the Talker side.
        raw = raw.clone()
        raw[:, audio_positions, :] = 0
        condition = raw + hidden
        thinker_embed = self.model.thinker.get_input_embeddings()
        codec_embed = self.model.talker.get_input_embeddings()

        def text_embedding(token):
            return thinker_embed(torch.tensor([[token]]))

        def codec_embedding(token):
            return codec_embed(torch.tensor([[token]]))

        body = windows[window_index].text_ids[:-1]
        parts = [condition[:, :prompt_length],
                 text_embedding(self.model.speaker_map[speaker]["bos_token"])
                 + codec_embedding(self.model.talker.codec_pad_token),
                 condition[:, prompt_length:prompt_length + 1]
                 + codec_embedding(self.model.talker.codec_bos_token)]
        for j, code in enumerate(codes):
            next_text = j + 1
            if next_text < len(body):
                text_condition = condition[:, prompt_length + next_text:prompt_length + next_text + 1]
            elif next_text == len(body):
                text_condition = text_embedding(self.model.talker.text_eos_token)
            else:
                text_condition = text_embedding(self.model.talker.text_pad_token)
            parts.append(text_condition + codec_embedding(code))
        return torch.cat(parts, dim=1), prompt_length

    def test_native_losses_match_unshifted_manual_cross_entropy(self):
        windows = [self.window()]
        codes = (1, 7, 4, 8, 2)
        with self.capture_talker() as calls:
            output = self.run_speech(windows, [SpeechTarget(0, codes)],
                                     thinker_loss_weight=0.7, speech_loss_weight=1.3)
        self.assertIsInstance(output.talker_outputs, tuple)
        self.assertEqual(len(output.talker_outputs), 1)
        self.assertEqual(len(calls), 1)
        for loss in (output.loss, output.thinker_loss, output.speech_loss):
            self.assertEqual(loss.ndim, 0)
            self.assertTrue(loss.is_floating_point())
            self.assertTrue(math.isfinite(loss.item()))
            self.assertGreater(loss.item(), 0)
        torch.testing.assert_close(output.thinker_loss, output.thinker_output.loss)
        _, _, starts = self.independent_thinker_layout(windows)
        p = starts[0]
        logits = output.talker_outputs[0].logits
        self.assertEqual(logits.shape, (1, p + 2 + len(codes), 32))
        target = torch.tensor([*codes, self.model.talker.codec_eos_token])
        expected = F.cross_entropy(logits[0, p + 1:p + len(codes) + 2], target)
        torch.testing.assert_close(output.speech_loss, expected)
        torch.testing.assert_close(output.loss, 0.7 * output.thinker_loss + 1.3 * expected)
        self.assertIsNone(output.talker_outputs[0].loss)

    def test_exact_prompt_bos_first_text_eos_and_repeated_pad_alignment(self):
        windows = [self.window("<|wait|>", (), 3), self.window(audio_length=4)]
        codes = (1, 2, 3, 4, 5, 6)
        with self.capture_talker() as calls:
            output = self.run_speech(windows, [SpeechTarget(1, codes)], speaker="Ethan")
        self.assertEqual(len(calls), 1)
        raw, _, _ = self.independent_thinker_layout(windows)
        length = raw.shape[1]
        with torch.no_grad():
            reference = self.model.thinker(
                inputs_embeds=raw,
                attention_mask=torch.ones(1, length, dtype=torch.long),
                position_ids=torch.arange(length).reshape(1, 1, -1).expand(3, 1, -1),
                output_hidden_states=True, use_cache=False, return_dict=True,
            )
            expected, p = self.expected_talker_input(
                windows, 1, codes, reference.hidden_states[-1], speaker="Ethan",
            )
        captured = calls[0]
        torch.testing.assert_close(captured["inputs_embeds"], expected, atol=1e-6, rtol=1e-5)
        self.assertEqual(expected.shape[1], p + 2 + len(codes))
        self.assertIs(captured.get("use_cache"), False)
        torch.testing.assert_close(
            captured["position_ids"],
            torch.arange(expected.shape[1]).reshape(1, 1, -1).expand(3, 1, -1),
        )
        torch.testing.assert_close(captured["attention_mask"], torch.ones(1, expected.shape[1], dtype=torch.long))
        torch.testing.assert_close(output.thinker_output.hidden_states[-1], reference.hidden_states[-1])
        self.assertIsNone(output.talker_outputs[0].past_key_values)

    def test_body_longer_than_codes_keeps_next_real_text_condition(self):
        windows = [self.window(text=(3, 9, 10, 11, self.tokenizer.eos_token_id))]
        codes = (4,)
        with self.capture_talker() as calls:
            output = self.run_speech(windows, [SpeechTarget(0, codes)])
        expected, _ = self.expected_talker_input(windows, 0, codes, output.thinker_output.hidden_states[-1])
        torch.testing.assert_close(calls[0]["inputs_embeds"], expected)

    def test_multiple_utterances_skip_non_speech_and_pool_all_codec_tokens(self):
        windows = [self.window("<|wait|>", ()), self.window(),
                   self.window("<|interrupt|>", ()), self.window("<|backchannel|>", (3, 2)),
                   self.window("<|yield|>", ())]
        targets = [SpeechTarget(1, (4,)), SpeechTarget(3, torch.tensor([1, 7, 3, 8, 5], dtype=torch.int32))]
        with self.capture_talker() as calls:
            output = self.run_speech(windows, targets)
        self.assertEqual(len(calls), 2)
        self.assertEqual(len(output.talker_outputs), 2)
        total = torch.zeros(())
        count = 0
        for target, native, call in zip(targets, output.talker_outputs, calls):
            codes = torch.as_tensor(target.codec_ids, dtype=torch.long)
            expected, p = self.expected_talker_input(
                windows, target.window_index, codes.tolist(), output.thinker_output.hidden_states[-1],
            )
            torch.testing.assert_close(call["inputs_embeds"], expected)
            labels = torch.cat((codes, torch.tensor([self.model.talker.codec_eos_token])))
            total = total + F.cross_entropy(native.logits[0, p + 1:p + 1 + len(labels)], labels, reduction="sum")
            count += len(labels)
        self.assertEqual(count, 8)
        torch.testing.assert_close(output.speech_loss, total / count)

    def test_empty_targets_return_zero_without_calling_talker(self):
        windows = [self.window("<|wait|>", ()), self.window()]
        with self.capture_talker() as calls:
            output = self.run_speech(windows, [], thinker_loss_weight=2.0)
        self.assertEqual(calls, [])
        self.assertEqual(output.talker_outputs, ())
        self.assertEqual(output.speech_loss.item(), 0.0)
        self.assertTrue(torch.isfinite(output.thinker_loss))
        torch.testing.assert_close(output.loss, 2 * output.thinker_loss)

    def test_future_windows_do_not_change_earlier_speech_condition_or_logits(self):
        first = self.window()
        targets = [SpeechTarget(0, (1, 2, 3, 4))]
        with self.capture_talker() as initial:
            before = self.run_speech([first], targets)
        future = [self.window("<|wait|>", (), 5), self.window("<|backchannel|>", (9, 10, 2))]
        with self.capture_talker() as extended:
            after = self.run_speech([first, *future], targets)
        torch.testing.assert_close(initial[0]["inputs_embeds"], extended[0]["inputs_embeds"], atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(before.talker_outputs[0].logits, after.talker_outputs[0].logits, atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(before.speech_loss, after.speech_loss)

    @staticmethod
    def gradient_mass(module):
        return sum(parameter.grad.abs().sum().item() for parameter in module.parameters()
                   if parameter.grad is not None)

    def test_speech_loss_alone_reaches_both_native_models_and_audio(self):
        window = self.window()
        window.audio_embeddings.requires_grad_()
        output = self.run_speech([window], [SpeechTarget(0, (1, 2, 3, 4))], thinker_loss_weight=0)
        output.speech_loss.backward()
        self.assertGreater(self.gradient_mass(self.model.thinker), 0)
        self.assertGreater(self.gradient_mass(self.model.talker), 0)
        self.assertGreater(self.model.thinker.get_input_embeddings().weight.grad.abs().sum().item(), 0)
        self.assertGreater(self.model.talker.codec_head.weight.grad.abs().sum().item(), 0)
        self.assertIsNotNone(window.audio_embeddings.grad)
        self.assertGreater(window.audio_embeddings.grad.abs().sum().item(), 0)

    def test_detach_blocks_every_thinker_speech_path_but_not_talker(self):
        window = self.window()
        window.audio_embeddings.requires_grad_()
        output = self.run_speech([window], [SpeechTarget(0, (1, 2, 3, 4))], detach_thinker=True)
        output.speech_loss.backward()
        self.assertEqual(self.gradient_mass(self.model.thinker), 0)
        self.assertTrue(window.audio_embeddings.grad is None or window.audio_embeddings.grad.count_nonzero().item() == 0)
        self.assertGreater(self.gradient_mass(self.model.talker), 0)
        self.assertGreater(self.model.talker.thinker_to_talker_proj.weight.grad.abs().sum().item(), 0)

    def test_detach_does_not_disable_independent_thinker_supervision(self):
        output = self.run_speech([self.window()], [SpeechTarget(0, (1, 2))], detach_thinker=True)
        output.loss.backward()
        self.assertGreater(self.gradient_mass(self.model.thinker), 0)
        self.assertGreater(self.gradient_mass(self.model.talker), 0)

    def test_train_mode_joint_optimizer_step_updates_both_heads(self):
        self.model.thinker.train()
        self.model.talker.train()
        parameters = [*self.model.thinker.parameters(), *self.model.talker.parameters()]
        optimizer = torch.optim.SGD(parameters, lr=0.01)
        thinker_before = self.model.thinker.lm_head.weight.detach().clone()
        talker_before = self.model.talker.codec_head.weight.detach().clone()
        optimizer.zero_grad()
        output = self.run_speech([self.window()], [SpeechTarget(0, (1, 2, 3))])
        output.loss.backward()
        optimizer.step()
        self.assertGreater((self.model.thinker.lm_head.weight - thinker_before).abs().sum().item(), 0)
        self.assertGreater((self.model.talker.codec_head.weight - talker_before).abs().sum().item(), 0)

    def test_split_hidden_and_codec_logits_devices_preserve_gradients(self):
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            self.skipTest("A second device is needed to test dispatched hidden states and logits")

        def move_final_hidden(module, args, output):
            output.hidden_states = (*output.hidden_states[:-1], output.hidden_states[-1].to(device))
            return output

        def move_codec_logits(module, args, output):
            output.logits = output.logits.to(device)
            return output

        hidden_hook = self.model.thinker.register_forward_hook(move_final_hidden)
        logits_hook = self.model.talker.register_forward_hook(move_codec_logits)
        try:
            output = self.run_speech([self.window()], [SpeechTarget(0, (1, 2, 3))])
            self.assertEqual(output.thinker_output.hidden_states[0].device.type, "cpu")
            self.assertEqual(output.thinker_output.hidden_states[-1].device.type, device.type)
            self.assertEqual(output.talker_outputs[0].logits.device.type, device.type)
            self.assertEqual(output.loss.device.type, "cpu")
            self.assertTrue(torch.isfinite(output.loss))
            output.loss.backward()
            self.assertGreater(self.gradient_mass(self.model.thinker), 0)
            self.assertGreater(self.gradient_mass(self.model.talker), 0)
        finally:
            hidden_hook.remove()
            logits_hook.remove()

    def test_both_utterance_terminators_are_valid(self):
        for stop in (self.tokenizer.eos_token_id, self.actions["<|end_speech|>"]):
            with self.subTest(stop=stop):
                output = self.run_speech([self.window(text=(3, stop))], [SpeechTarget(0, (0, 27))])
                self.assertTrue(torch.isfinite(output.speech_loss))

    def test_rejects_duplicate_out_of_range_and_non_speaking_targets(self):
        windows = [self.window(), self.window("<|wait|>", ()),
                   self.window("<|interrupt|>", ()), self.window("<|yield|>", ())]
        invalid = [[SpeechTarget(0, (1,)), SpeechTarget(0, (2,))],
                   [SpeechTarget(-1, (1,))], [SpeechTarget(len(windows), (1,))],
                   [SpeechTarget(1, (1,))], [SpeechTarget(2, (1,))], [SpeechTarget(3, (1,))]]
        for targets in invalid:
            with self.subTest(targets=targets), self.assertRaises(ValueError):
                self.run_speech(windows, targets)

    def test_rejects_incomplete_or_nonordinary_utterance_text(self):
        invalid = [(), (3,), (self.tokenizer.eos_token_id,), (self.actions["<|end_speech|>"],),
                   (3, self.tokenizer.eos_token_id, 9),
                   (self.actions["<|wait|>"], self.tokenizer.eos_token_id),
                   (self.model.thinker.config.audio_start_token_id, self.tokenizer.eos_token_id)]
        for text in invalid:
            with self.subTest(text=text), self.assertRaises(ValueError):
                self.run_speech([self.window(text=text)], [SpeechTarget(0, (1, 2))])

    def test_rejects_empty_noninteger_nonvector_reserved_and_out_of_vocab_codes(self):
        talker = self.model.talker
        invalid = [(), torch.empty(0, dtype=torch.long), torch.tensor(1), torch.tensor([[1, 2]]),
                   torch.tensor([1.0, 2.0]), torch.tensor([True, False]), (1.5,), (True,),
                   (-1,), (talker.config.vocab_size,),
                   (talker.codec_bos_token,), (talker.codec_eos_token,),
                   (talker.codec_pad_token,), (talker.codec_mask_token,)]
        for codes in invalid:
            with self.subTest(codes=codes), self.assertRaises((ValueError, TypeError)):
                self.run_speech([self.window()], [SpeechTarget(0, codes)])

    def test_rejects_unknown_speaker_and_invalid_loss_weights(self):
        windows = [self.window()]
        targets = [SpeechTarget(0, (1, 2))]
        with self.assertRaises(ValueError):
            self.run_speech(windows, targets, speaker="unknown")
        invalid = [(-1, 1), (1, -1), (float("nan"), 1), (1, float("nan")),
                   (float("inf"), 1), (1, float("inf")), (-float("inf"), 1), (0, 0)]
        for thinker_weight, speech_weight in invalid:
            with self.subTest(weights=(thinker_weight, speech_weight)), self.assertRaises(ValueError):
                self.run_speech(windows, targets, thinker_loss_weight=thinker_weight,
                                speech_loss_weight=speech_weight)
        # Zeroing either individual objective is explicitly legal.
        for weights in ((0, 1), (1, 0)):
            with self.subTest(valid_weights=weights):
                output = self.run_speech(windows, targets, thinker_loss_weight=weights[0], speech_loss_weight=weights[1])
                self.assertTrue(torch.isfinite(output.loss))

    def test_rejects_thinker_and_talker_context_overflow(self):
        # 3 prefix + (2 audio + 2 delimiters + action) + 3 text = 11.
        self.model.thinker.config.text_config.max_position_embeddings = 10
        with self.assertRaises(ValueError):
            self.run_speech([self.window()], [SpeechTarget(0, (1,))])
        self.model.thinker.config.text_config.max_position_embeddings = 512
        # P=8, so C=4 requires 14 Talker input positions.
        self.model.talker.config.max_position_embeddings = 13
        with self.assertRaises(ValueError):
            self.run_speech([self.window()], [SpeechTarget(0, (1, 2, 3, 4))])
        self.model.talker.config.max_position_embeddings = 14
        self.assertTrue(torch.isfinite(self.run_speech([self.window()], [SpeechTarget(0, (1, 2, 3, 4))]).loss))

    def test_native_cached_steps_match_full_teacher_forced_talker_logits(self):
        windows = [self.window()]
        codes = (1, 2, 3, 4, 5)
        output = self.run_speech(windows, [SpeechTarget(0, codes)])
        raw, audio_positions, starts = self.independent_thinker_layout(windows)
        p = starts[0]
        embed = self.model.thinker.get_input_embeddings()
        hidden = output.thinker_output.hidden_states[-1]
        masked = raw.clone()
        masked[:, audio_positions] = 0
        prompt = hidden[:, :p] + masked[:, :p]
        reply = raw[:, p:p + 2] + hidden[:, p:p + 2]
        bos = embed(torch.tensor([[self.model.speaker_map["Chelsie"]["bos_token"]]]))
        eos = embed(torch.tensor([[self.model.talker.text_eos_token]]))
        pad = embed(torch.tensor([[self.model.talker.text_pad_token]]))
        queue = torch.cat((reply[:, 1:], eos, pad), dim=1)
        # Exercise native automatic prefill PAD/BOS insertion and its actual
        # input_ids + reply-queue incremental branch, not precombined embeddings.
        first_length = p + 2
        with torch.no_grad():
            step = self.model.talker(
                inputs_embeds=torch.cat((prompt, bos, reply[:, :1]), dim=1),
                input_text_ids=torch.ones(1, first_length, dtype=torch.long),
                thinker_reply_part=queue,
                attention_mask=torch.ones(1, first_length, dtype=torch.long),
                cache_position=torch.arange(first_length),
                use_cache=True, return_dict=True,
            )
            chunks = [step.logits]
            for j, code in enumerate(codes):
                step = self.model.talker(
                    input_ids=torch.tensor([[code]]),
                    thinker_reply_part=step.thinker_reply_part,
                    attention_mask=torch.ones(1, first_length + j + 1, dtype=torch.long),
                    cache_position=torch.tensor([first_length + j]),
                    past_key_values=step.past_key_values,
                    use_cache=True, return_dict=True,
                )
                chunks.append(step.logits)
        self.assertEqual(step.past_key_values.get_seq_length(), first_length + len(codes))
        torch.testing.assert_close(torch.cat(chunks, dim=1), output.talker_outputs[0].logits, atol=1e-6, rtol=1e-5)


if __name__ == "__main__":
    unittest.main()
