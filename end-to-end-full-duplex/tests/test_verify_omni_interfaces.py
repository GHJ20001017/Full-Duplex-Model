"""Offline tests for diagnostic bookkeeping, not model-capability tests."""

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / 'script' / 'verify_omni_interfaces.py'
SPEC = importlib.util.spec_from_file_location('verify_omni_interfaces', SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class InterfaceProbeHelpersTests(unittest.TestCase):
    def test_transcript_normalization_preserves_words_and_digits(self):
        self.assertEqual(MODULE.normalize_text('Hi, BRION! 123.'), 'hi brion 123')

    def test_unicode_and_ascii_apostrophes_normalize_equally(self):
        self.assertEqual(MODULE.normalize_text("It's fine."), MODULE.normalize_text('It\u2019s fine.'))

    def test_asset_audit_distinguishes_decoder_and_codec_encoder(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'config.json').write_text('{}')
            (root / 'weights.safetensors').write_bytes(b'not-a-real-checkpoint')
            weights = {
                'thinker.audio_tower.conv1.weight': 'weights.safetensors',
                'token2wav.code2wav_dit_model.weight': 'weights.safetensors',
            }
            (root / 'model.safetensors.index.json').write_text(json.dumps({'weight_map': weights}))
            audit = MODULE.asset_audit(root)
            self.assertEqual(audit['waveform_codec_encoder_name_matches'], [])
            self.assertEqual(audit['weight_count'], 2)
            self.assertEqual(len(audit['config_sha256']), 64)
            weights['speech_tokenizer.quantizer.weight'] = 'weights.safetensors'
            (root / 'model.safetensors.index.json').write_text(json.dumps({'weight_map': weights}))
            self.assertEqual(MODULE.asset_audit(root)['waveform_codec_encoder_name_matches'], ['speech_tokenizer.quantizer.weight'])

    def test_missing_weight_shard_is_not_silently_accepted(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'config.json').write_text('{}')
            (root / 'model.safetensors.index.json').write_text(json.dumps({'weight_map': {'talker.weight': 'missing.safetensors'}}))
            with self.assertRaises(FileNotFoundError):
                MODULE.asset_audit(root)


if __name__ == '__main__':
    unittest.main()
