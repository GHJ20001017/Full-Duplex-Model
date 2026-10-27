"""Offline input-model contracts using a real, randomly initialized Qwen3."""

import unittest
from contextlib import ExitStack

import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from voxbase_omni import PreparedInputs, VoxBaseOmni
from voxbase_omni.audio import align_continuous


class VoxBaseOmniTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        torch.manual_seed(17)
        config = Qwen3Config(
            vocab_size=41,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
            max_position_embeddings=64,
            attention_dropout=0.0,
            pad_token_id=0,
            bos_token_id=1,
            eos_token_id=2,
            tie_word_embeddings=False,
        )
        config._attn_implementation = "eager"
        self.llm = Qwen3ForCausalLM(config)
        self.model = VoxBaseOmni(
            self.llm, audio_hidden_size=12, codebook_size=16, adapter_dim=8
        ).eval()
        self.ids = torch.tensor([[3, -999, -999, 7, 8, 9]])
        self.positions = torch.tensor([[1, 2]])
        self.codes = torch.randint(0, 16, (1, 8, 2))

    def audio_arguments(self):
        return {"audio_codes": self.codes, "audio_positions": self.positions}

    def assert_live_gradients(self, module):
        for name, parameter in module.named_parameters():
            with self.subTest(parameter=name):
                self.assertIsNotNone(parameter.grad)
                self.assertTrue(torch.isfinite(parameter.grad).all().item())
                self.assertGreater(parameter.grad.abs().sum().item(), 0)

    @staticmethod
    def forbidden_call(module, args):
        raise AssertionError("Unavailable continuous audio must bypass this module")

    def test_discrete_forward_backward_and_independent_adapters(self):
        with ExitStack() as stack:
            for module in (self.model.projector, self.model.fusion):
                stack.enter_context(module.register_forward_pre_hook(self.forbidden_call))
            output = self.model(
                self.ids, labels=self.ids.clone(), **self.audio_arguments()
            )
            self.assertEqual(output.logits.shape, (1, 6, 41))
            self.assertTrue(torch.isfinite(output.loss).item())
            self.assertGreater(output.loss.item(), 0)
            output.loss.backward()
        encoder = self.model.discrete_encoder
        self.assertEqual(len(encoder.acoustic_adapters), 7)
        adapter_parameters = [
            parameter
            for adapter in encoder.acoustic_adapters
            for parameter in adapter.parameters()
        ]
        self.assertEqual(
            len({parameter.data_ptr() for parameter in adapter_parameters}),
            len(adapter_parameters),
        )
        self.assertNotEqual(
            encoder.semantic_embedding.weight.data_ptr(),
            encoder.acoustic_embedding.weight.data_ptr(),
        )
        self.assert_live_gradients(encoder.semantic_embedding)
        self.assert_live_gradients(encoder.acoustic_embedding)
        for adapter in encoder.acoustic_adapters:
            self.assert_live_gradients(adapter)
        self.assertTrue(all(p.grad is None for p in self.model.projector.parameters()))
        self.assertTrue(all(p.grad is None for p in self.model.fusion.parameters()))

    def test_joint_alignment_projection_fusion_and_backward(self):
        features = torch.randn(1, 4, 12, requires_grad=True)
        indices = torch.tensor([[3, 1]])
        seen = []
        with self.model.projector.register_forward_pre_hook(
            lambda module, args: seen.append(args[0].detach().clone())
        ):
            output = self.model(
                self.ids,
                labels=self.ids.clone(),
                continuous_features=features,
                continuous_frame_indices=indices,
                **self.audio_arguments(),
            )
        torch.testing.assert_close(seen[0], features.detach()[0, [3, 1]])
        self.assertTrue(torch.isfinite(output.loss).item())
        output.loss.backward()
        self.assert_live_gradients(self.model.projector)
        self.assert_live_gradients(self.model.fusion)
        self.assertTrue(torch.isfinite(features.grad).all().item())
        for index in (1, 3):
            self.assertGreater(features.grad[0, index].abs().sum().item(), 0)
        torch.testing.assert_close(features.grad[0, [0, 2]], torch.zeros(2, 12))

    def test_partial_missing_and_nan_padding_bypass(self):
        codes = torch.randint(0, 16, (1, 8, 4))
        codes[:, :, -1] = -1
        mask = torch.tensor([[True, True, True, False]])
        features = torch.randn(1, 3, 12)
        features[:, 0] = float("nan")
        features.requires_grad_()
        indices = torch.tensor([[2, -1, 1, 999]])
        feature_mask = torch.tensor([[False, True, True]])
        discrete = self.model.discrete_encoder(codes, mask)
        expected = self.model.fusion(
            discrete[:, [0, 2]], self.model.projector(features[:, [2, 1]])
        )
        fused = self.model.encode_audio(
            codes,
            mask,
            continuous_features=features,
            continuous_frame_indices=indices,
            continuous_feature_mask=feature_mask,
        )
        torch.testing.assert_close(fused[:, [0, 2]], expected)
        torch.testing.assert_close(fused[:, 1], discrete[:, 1], rtol=0, atol=0)
        torch.testing.assert_close(fused[:, 3], torch.zeros(1, 32))
        self.assertTrue(torch.isfinite(fused).all().item())
        fused.square().sum().backward()
        self.assertTrue(torch.isfinite(features.grad).all().item())
        torch.testing.assert_close(features.grad[:, 0], torch.zeros(1, 12))

    def test_all_missing_or_empty_continuous_audio_never_calls_modules(self):
        mask = torch.ones(1, 2, dtype=torch.bool)
        discrete = self.model.discrete_encoder(self.codes, mask)
        with ExitStack() as stack:
            for module in (self.model.projector, self.model.fusion):
                stack.enter_context(module.register_forward_pre_hook(self.forbidden_call))
            for time in (0, 3):
                with self.subTest(continuous_frames=time):
                    actual = self.model.encode_audio(
                        self.codes,
                        mask,
                        continuous_features=torch.full((1, time, 12), float("nan")),
                        continuous_frame_indices=torch.full((1, 2), -1),
                        continuous_feature_mask=torch.zeros(1, time, dtype=torch.bool),
                    )
                    torch.testing.assert_close(actual, discrete, rtol=0, atol=0)

    def test_mixed_batch_missing_continuous_and_text_only_sample(self):
        ids = self.ids.expand(3, -1).clone()
        ids[2] = torch.tensor([3, 4, 5, 7, 8, 9])
        codes = self.codes.expand(3, -1, -1).clone()
        codes[2] = -1
        positions = torch.tensor([[1, 2], [1, 2], [-1, -1]])
        baseline = self.model.prepare_inputs(ids, audio_codes=codes, audio_positions=positions)
        features = torch.randn(3, 3, 12)
        features[1:] = float("nan")
        features.requires_grad_()
        kwargs = dict(
            audio_codes=codes, audio_positions=positions,
            continuous_features=features,
            continuous_frame_indices=torch.tensor([[2, 0], [-1, -1], [-1, -1]]),
            continuous_feature_mask=torch.tensor([[True, True, True], [False] * 3, [False] * 3]),
        )
        seen = []
        with self.model.projector.register_forward_pre_hook(
            lambda module, args: seen.append(args[0].shape)
        ):
            prepared = self.model.prepare_inputs(ids, **kwargs)
        self.assertEqual(seen, [torch.Size([2, 12])])
        torch.testing.assert_close(prepared.inputs_embeds[1:], baseline.inputs_embeds[1:], rtol=0, atol=0)
        output = self.model(ids, labels=ids.clone(), **kwargs)
        self.assertTrue(torch.isfinite(output.loss))
        output.loss.backward()
        torch.testing.assert_close(features.grad[1:], torch.zeros_like(features.grad[1:]))

    def test_explicit_alignment_and_rejected_continuous_indices(self):
        features = torch.arange(48, dtype=torch.float32).reshape(1, 4, 12)
        mask = torch.ones(1, 2, dtype=torch.bool)
        aligned, available = align_continuous(features, torch.tensor([[3, 0]]), mask)
        torch.testing.assert_close(aligned, features[:, [3, 0]])
        self.assertTrue(available.all().item())
        cases = [
            {"continuous_features": features},
            {"continuous_features": torch.tensor(0.0), "continuous_frame_indices": torch.tensor([[0, 1]])},
            {"continuous_features": features[0], "continuous_frame_indices": torch.tensor([[0, 1]])},
            {"continuous_features": features, "continuous_frame_indices": torch.tensor([[4, 0]])},
            {"continuous_features": features, "continuous_frame_indices": torch.tensor([[-2, 0]])},
            {"continuous_features": features, "continuous_frame_indices": torch.tensor([[0.0, 1.0]])},
            {"continuous_features": features, "continuous_frame_indices": torch.tensor([[0]])},
            {"continuous_features": features, "continuous_frame_indices": torch.tensor([[0, 2]]),
             "continuous_feature_mask": torch.tensor([[True, True, False, True]])},
            {"continuous_frame_indices": torch.tensor([[0, 1]])},
            {"continuous_feature_mask": torch.ones(1, 4, dtype=torch.bool)},
            {"continuous_features": features.long(), "continuous_frame_indices": torch.tensor([[0, 1]])},
            {"continuous_features": features[:, :, :11], "continuous_frame_indices": torch.tensor([[0, 1]])},
            {"continuous_features": features, "continuous_frame_indices": torch.tensor([[0, 1]]),
             "continuous_feature_mask": torch.ones(1, 4)},
        ]
        for case in cases:
            with self.subTest(arguments=tuple(case), indices=case.get("continuous_frame_indices")):
                with self.assertRaises(ValueError):
                    self.model.encode_audio(self.codes, mask, **case)

    def test_interleaved_audio_blocks_preserve_text_slots_and_caller_labels(self):
        ids = torch.tensor([[3, -999, -999, 4, 5, -999, -999, 6, 7]])
        positions = torch.tensor([[1, 2, 5, 6]])
        codes = torch.randint(0, 16, (1, 8, 4))
        labels = ids.clone()
        prepared = self.model.prepare_inputs(
            ids, labels=labels, audio_codes=codes, audio_positions=positions
        )
        self.assertIsInstance(prepared, PreparedInputs)
        self.assertEqual(prepared.inputs_embeds.shape, (1, 9, 32))
        text_positions = [0, 3, 4, 7, 8]
        torch.testing.assert_close(
            prepared.inputs_embeds[:, text_positions],
            self.llm.get_input_embeddings()(ids[:, text_positions]),
        )
        torch.testing.assert_close(
            prepared.inputs_embeds[:, positions[0]], self.model.discrete_encoder(codes)
        )
        expected_labels = ids.clone()
        expected_labels[:, positions[0]] = -100
        torch.testing.assert_close(prepared.labels, expected_labels)
        torch.testing.assert_close(labels, ids)
        torch.testing.assert_close(prepared.position_ids, torch.arange(9).unsqueeze(0))

    def test_left_and_right_padding_labels_positions_and_loss(self):
        ids = torch.tensor([[3, -999, 7, 8, -999, -999], [-999, -999, 3, -999, 7, 8]])
        mask = torch.tensor([[1, 1, 1, 1, 0, 0], [0, 0, 1, 1, 1, 1]])
        codes = self.codes[:, :, :1].expand(2, -1, -1).clone()
        positions = torch.tensor([[1], [3]])
        prepared = self.model.prepare_inputs(
            ids, attention_mask=mask, labels=ids.clone(),
            audio_codes=codes, audio_positions=positions,
        )
        torch.testing.assert_close(
            prepared.position_ids, torch.tensor([[0, 1, 2, 3, 3, 3], [0, 0, 0, 1, 2, 3]])
        )
        torch.testing.assert_close(
            prepared.labels, torch.tensor([[3, -100, 7, 8, -100, -100], [-100, -100, 3, -100, 7, 8]])
        )
        self.assertEqual(prepared.attention_mask.dtype, torch.bool)
        batched = self.model(
            ids, attention_mask=mask, labels=ids.clone(),
            audio_codes=codes, audio_positions=positions,
        )
        unpadded = self.model(
            ids[:1, :4], labels=ids[:1, :4].clone(),
            audio_codes=codes[:1], audio_positions=positions[:1],
        )
        torch.testing.assert_close(batched.loss, unpadded.loss)
        torch.testing.assert_close(batched.logits[0, :4], unpadded.logits[0])
        torch.testing.assert_close(batched.logits[1, 2:], unpadded.logits[0])

    def test_padded_mimi_frames_are_not_embedded(self):
        codes = torch.cat((self.codes, torch.full((1, 8, 1), -1)), dim=2)
        positions = torch.tensor([[1, 2, -1]])
        padded = self.model.prepare_inputs(self.ids, audio_codes=codes, audio_positions=positions)
        baseline = self.model.prepare_inputs(self.ids, **self.audio_arguments())
        torch.testing.assert_close(padded.inputs_embeds, baseline.inputs_embeds)

    def test_pure_text_matches_stock_qwen_loss_and_logits(self):
        ids = torch.tensor([[3, 4, 5, 6, 7], [8, 9, 10, 11, 12]])
        labels = ids.clone()
        labels[:, :2] = -100
        stock = self.llm(input_ids=ids, labels=labels, use_cache=False)
        wrapped = self.model(ids, labels=labels)
        torch.testing.assert_close(wrapped.logits, stock.logits)
        torch.testing.assert_close(wrapped.loss, stock.loss)
        self.assertIsNone(self.model(ids).loss)
        self.assertIsNone(self.model.prepare_inputs(ids).labels)

    def test_all_ignored_and_one_token_have_finite_differentiable_zero_loss(self):
        for ids, labels in (
            (torch.tensor([[3, 4, 5]]), torch.full((1, 3), -100)),
            (torch.tensor([[3]]), torch.tensor([[3]])),
        ):
            with self.subTest(length=ids.shape[1]):
                self.model.zero_grad(set_to_none=True)
                output = self.model(ids, labels=labels)
                self.assertTrue(torch.isfinite(output.loss).item())
                self.assertEqual(output.loss.item(), 0)
                self.assertTrue(output.loss.requires_grad)
                output.loss.backward()
                gradient = self.llm.get_input_embeddings().weight.grad
                self.assertIsNotNone(gradient)
                self.assertTrue(torch.isfinite(gradient).all().item())
                self.assertEqual(gradient.abs().sum().item(), 0)

    def test_half_precision_ignored_loss_does_not_overflow(self):
        self.model.half()
        with torch.no_grad():
            self.llm.get_input_embeddings().weight.fill_(1)
            self.llm.get_output_embeddings().weight.fill_(1)
        ids = torch.full((1, 64), 3, dtype=torch.long)
        output = self.model(ids, labels=torch.full_like(ids, -100))
        self.assertTrue(torch.isfinite(output.logits).all())
        self.assertTrue(torch.isinf(output.logits.sum()))
        self.assertEqual(output.loss.item(), 0)
        self.assertEqual(output.loss.dtype, torch.float32)
        output.loss.backward()
        self.assertTrue(torch.isfinite(self.llm.get_input_embeddings().weight.grad).all())

    def test_invalid_codebooks_codes_and_frame_masks(self):
        for codes in (
            torch.zeros(1, 7, 2, dtype=torch.long),
            torch.zeros(1, 9, 2, dtype=torch.long),
            torch.zeros(1, 8, dtype=torch.long),
            self.codes.float(),
            torch.full_like(self.codes, -1),
            torch.full_like(self.codes, 16),
        ):
            with self.subTest(shape=tuple(codes.shape), dtype=codes.dtype):
                with self.assertRaises(ValueError):
                    self.model.discrete_encoder(codes)
        for mask in (torch.ones(1, 2), torch.ones(1, 3, dtype=torch.bool)):
            with self.subTest(mask_shape=tuple(mask.shape), mask_dtype=mask.dtype):
                with self.assertRaises(ValueError):
                    self.model.discrete_encoder(self.codes, mask)

    def test_invalid_audio_positions_and_input_contracts(self):
        for positions in (
            torch.tensor([[-2, 2]]), torch.tensor([[1, 6]]),
            torch.tensor([[1, 1]]), torch.tensor([[2, 1]]),
            torch.tensor([[1.0, 2.0]]), torch.tensor([[1]]),
        ):
            with self.subTest(positions=positions):
                with self.assertRaises(ValueError):
                    self.model.prepare_inputs(self.ids, audio_codes=self.codes, audio_positions=positions)
        with self.assertRaises(ValueError):
            self.model.prepare_inputs(
                self.ids, attention_mask=torch.tensor([[1, 1, 0, 1, 1, 1]]),
                **self.audio_arguments(),
            )
        text = torch.tensor([[3, 4, 5]])
        for arguments in (
            {"audio_codes": self.codes},
            {"audio_positions": self.positions},
            {"continuous_features": torch.randn(1, 2, 12)},
            {"attention_mask": torch.tensor([[1, 2, 1]])},
            {"attention_mask": torch.zeros(1, 3, dtype=torch.long)},
            {"attention_mask": torch.ones(1, 2)},
            {"labels": text.float()},
            {"labels": torch.tensor([[3, 4]])},
            {"labels": torch.tensor([[3, -1, 5]])},
            {"labels": torch.tensor([[3, 41, 5]])},
        ):
            with self.subTest(arguments=tuple(arguments)):
                with self.assertRaises(ValueError):
                    self.model.prepare_inputs(text, **arguments)
        for ids in (text.float(), torch.tensor([3, 4]), torch.empty(1, 0, dtype=torch.long)):
            with self.subTest(input_shape=tuple(ids.shape), input_dtype=ids.dtype):
                with self.assertRaises(ValueError):
                    self.model.prepare_inputs(ids)
        for invalid_id in (-1, 41):
            with self.subTest(invalid_text_id=invalid_id):
                with self.assertRaises((ValueError, IndexError)):
                    self.model.prepare_inputs(torch.tensor([[3, invalid_id, 5]]))

    def test_float64_features_are_cast_to_bfloat16_model_dtype(self):
        self.model.to(dtype=torch.bfloat16)
        features = torch.randn(1, 2, 12, dtype=torch.float64, requires_grad=True)
        seen_dtypes = []
        with self.model.projector.register_forward_pre_hook(
            lambda module, args: seen_dtypes.append(args[0].dtype)
        ):
            output = self.model(
                self.ids, labels=self.ids.clone(),
                continuous_features=features,
                continuous_frame_indices=torch.tensor([[0, 1]]),
                **self.audio_arguments(),
            )
        self.assertEqual(seen_dtypes, [torch.bfloat16])
        self.assertEqual(output.logits.dtype, torch.bfloat16)
        self.assertEqual(output.loss.dtype, torch.float32)
        self.assertTrue(torch.isfinite(output.loss).item())
        output.loss.backward()
        self.assertTrue(torch.isfinite(features.grad).all().item())
        self.assertGreater(features.grad.abs().sum().item(), 0)


if __name__ == "__main__":
    unittest.main()
