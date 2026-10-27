"""Exercise the real upstream AuT with random small weights; no downloads."""

import importlib.util
import unittest
from unittest.mock import patch

import torch
from torch import nn

from voxbase_omni import Qwen3ASRAudioEncoder, VoxBaseOmni


@unittest.skipUnless(importlib.util.find_spec("qwen_asr"), "Optional qwen-asr is not installed")
class AudioEncoderTests(unittest.TestCase):
    def setUp(self):
        from qwen_asr.core.transformers_backend.configuration_qwen3_asr import Qwen3ASRAudioEncoderConfig
        from qwen_asr.core.transformers_backend.modeling_qwen3_asr import Qwen3ASRAudioEncoder as AuT

        torch.manual_seed(9)
        config = Qwen3ASRAudioEncoderConfig(
            num_mel_bins=128, encoder_layers=1, encoder_attention_heads=2,
            encoder_ffn_dim=32, d_model=16, output_dim=12,
            downsample_hidden_size=4, n_window=50, n_window_infer=100,
            max_source_positions=128,
        )
        config._attn_implementation = "eager"
        self.encoder = Qwen3ASRAudioEncoder(AuT(config))
        self.features = torch.randn(2, 128, 110)
        self.mask = torch.arange(110)[None, :] < torch.tensor([110, 73])[:, None]

    def test_real_aut_ragged_output_and_backward(self):
        features = self.features.requires_grad_()
        encoded, mask = self.encoder(features, self.mask)
        # The one-second convolution reset produces 13 + ceil(10 / 8), not ceil(110 / 8).
        self.assertEqual(encoded.shape, (2, 15, 12))
        self.assertEqual(mask.sum(-1).tolist(), [15, 10])
        self.assertTrue(torch.equal(encoded[1, 10:], torch.zeros_like(encoded[1, 10:])))
        encoded.square().sum().backward()
        self.assertGreater(features.grad[0].abs().sum().item(), 0)
        self.assertEqual(features.grad[1, :, 73:].abs().sum().item(), 0)
        self.assertIsNotNone(self.encoder.audio_tower.proj2.weight.grad)

    def test_padding_does_not_change_valid_features(self):
        self.encoder.eval()
        original, _ = self.encoder(self.features, self.mask)
        changed = self.features.clone()
        changed[1, :, 73:] = float("nan")
        actual, _ = self.encoder(changed, self.mask)
        torch.testing.assert_close(actual, original)

    def test_invalid_feature_masks(self):
        for mask in (torch.zeros_like(self.mask), self.mask.roll(1, dims=-1), self.mask.float() * 0.5):
            with self.subTest(mask=mask), self.assertRaises(ValueError):
                self.encoder(self.features, mask)
        with self.assertRaises(ValueError):
            self.encoder(self.features[:, :64], self.mask)

    def test_full_continuous_input_to_qwen_backward(self):
        from transformers import Qwen3Config, Qwen3ForCausalLM

        llm = Qwen3ForCausalLM(Qwen3Config(
            vocab_size=32, hidden_size=32, intermediate_size=48,
            num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
            head_dim=16,
        ))
        model = VoxBaseOmni(llm, audio_encoder=self.encoder, audio_hidden_size=12,
                            codebook_size=16, adapter_dim=8)
        ids = torch.tensor([[1, -1, -1, 2, 3], [1, -1, -1, 4, 5]])
        labels = ids.clone()
        labels[:, :3] = -100
        output = model(
            ids, labels=labels, audio_codes=torch.randint(16, (2, 8, 2)),
            audio_positions=torch.tensor([[1, 2], [1, 2]]),
            continuous_frame_indices=torch.tensor([[0, 14], [0, 9]]),
            audio_inputs={"input_features": self.features, "feature_attention_mask": self.mask},
        )
        self.assertTrue(torch.isfinite(output.loss))
        output.loss.backward()
        for parameter in (self.encoder.audio_tower.proj2.weight,
                          model.projector.net[1].weight,
                          model.fusion.gate.weight,
                          model.discrete_encoder.semantic_embedding.weight):
            self.assertIsNotNone(parameter.grad)
            self.assertGreater(parameter.grad.abs().sum().item(), 0)

    def test_loading_retains_only_audio_tower(self):
        from types import SimpleNamespace
        from qwen_asr.core.transformers_backend import Qwen3ASRForConditionalGeneration

        tower = self.encoder.audio_tower
        fake_asr = SimpleNamespace(thinker=SimpleNamespace(audio_tower=tower, model=nn.Linear(3, 3)))
        with patch.object(Qwen3ASRForConditionalGeneration, "from_pretrained", return_value=fake_asr) as loader:
            encoder = Qwen3ASRAudioEncoder.from_pretrained("local-asr", local_files_only=True)
        loader.assert_called_once_with("local-asr", local_files_only=True)
        self.assertIs(encoder.audio_tower, tower)
        self.assertEqual(list(encoder._modules), ["audio_tower"])


if __name__ == "__main__":
    unittest.main()
