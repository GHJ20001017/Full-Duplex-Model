import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dataset import EmiliaDataset, collate_emilia


class Tokenizer:
    cache_identity = 'mock-tokenizer-v1'

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        assert not add_special_tokens
        # One word splits into two tokens: character != token.
        result = {'input_ids': [10, 11, 12]}
        if return_offsets_mapping:
            result['offset_mapping'] = [(0, 2), (2, 5), (5, len(text))]
        return result


class Codec:
    cache_identity = 'mock-codec-v1'
    codebook_sizes = (32,) * 8

    def __init__(self, frames=20):
        self.frames, self.calls = frames, 0

    def __call__(self, waveform):
        self.calls += 1
        assert waveform.shape == (24000,)
        assert waveform.device.type == 'cpu'
        return torch.zeros(8, self.frames, dtype=torch.long)


class EmiliaTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)
        self.root = self.base / 'emilia-en-100h'
        (self.root / 'extracted').mkdir(parents=True)
        self.manifest = self.base / 'aligned.jsonl'
        self.codec = Codec()
        self.rows = [self.row()]
        self.save()
        self.audio = patch('dataset.emilia._load_audio', return_value=torch.zeros(24000))
        self.audio.start()
        self.addCleanup(self.audio.stop)

    def row(self, source='Y000001', segment=1):
        identifier = f'EN_{source}_S{segment:05d}_W000001'
        metadata = self.root / 'extracted' / (identifier + '.json')
        metadata.write_text(json.dumps({'id': identifier, 'text': 'hello world'}))
        audio = metadata.with_suffix('.wav')
        audio.write_bytes(b'mock audio')
        return {'config': {'schema': 2, 'datasets': [str(self.root)]},
                'metadata': str(metadata), 'metadata_sha256': hashlib.sha256(metadata.read_bytes()).hexdigest(),
                'id': identifier, 'audio': str(audio), 'text': 'hello world', 'duration': 1.0,
                'status': 'generated', 'quality_status': 'not_evaluated',
                'error_rate': 0.99, 'timestamp_unit': 'seconds',
                'timestamps': [{'text': 'hello', 'start': 0.0, 'end': 0.4},
                               {'text': 'world', 'start': 0.5, 'end': 1.04}]}

    def save(self):
        self.manifest.write_text(''.join(json.dumps(r) + '\n' for r in self.rows))

    def dataset(self, **kwargs):
        kwargs.setdefault('validation_fraction', 0)
        return EmiliaDataset(self.manifest, Tokenizer(), self.codec, 99, **kwargs)

    def test_shapes_full_tokenization_and_collate(self):
        dataset = self.dataset()
        sample = dataset[0]
        self.assertEqual(sample['audio_tokens'].shape, (8, 20))
        self.assertEqual(sample['text_tokens'][sample['text_tokens'] != 99].tolist(), [10, 11, 12])
        self.assertTrue(sample['alignment']['word_internal_start_estimated'])
        short = dict(sample, audio_tokens=sample['audio_tokens'][:, :15], text_tokens=sample['text_tokens'][:15])
        batch = collate_emilia([sample, short])
        self.assertEqual(batch['audio_tokens'].shape, (2, 8, 20))
        self.assertTrue((batch['audio_tokens'][1, :, 15:] == -1).all())
        self.assertTrue((batch['text_tokens'][1, 15:] == -1).all())
        self.assertFalse(batch['frame_mask'][1, 15:].any())
        with self.assertRaises(ValueError):
            collate_emilia([])

    def test_latest_error_excludes_previously_generated(self):
        other = self.row('Y000002')
        self.rows.extend([other, dict(self.rows[0], status='error', timestamps=None)])
        self.save()
        dataset = self.dataset()
        self.assertEqual(len(dataset), 1)
        self.assertEqual(dataset.records[0]['id'], other['id'])
        self.assertEqual(dataset.stats['errors'], 1)
        self.assertEqual(dataset.stats['superseded'], 1)
        self.rows.append(dict(self.rows[0]))
        self.save()
        self.assertEqual(len(self.dataset()), 2)

    def test_grouped_split_order_independent_and_max_samples(self):
        self.rows = [self.row(f'Y{i:06d}', j) for i in range(40) for j in (1, 2)]
        self.save()
        train = self.dataset(validation_fraction=0.3, seed=17)
        val = self.dataset(split='val', validation_fraction=0.3, seed=17)
        self.assertFalse({r['group'] for r in train.records} & {r['group'] for r in val.records})
        self.assertEqual(len(train) + len(val), 80)
        self.rows.reverse()
        self.save()
        repeated = self.dataset(validation_fraction=0.3, seed=17)
        self.assertEqual(train.records, repeated.records)
        self.assertEqual(train.fingerprint, repeated.fingerprint)
        self.assertEqual(len(self.dataset(max_samples=3)), 3)

    def test_empty_splits_and_filters_raise(self):
        with self.assertRaisesRegex(ValueError, 'Empty val'):
            self.dataset(split='val')
        with self.assertRaisesRegex(ValueError, 'Empty train'):
            self.dataset(min_duration=2)
        with self.assertRaises(ValueError):
            self.dataset(max_samples=0)

    def test_bad_generated_is_not_hidden_by_filter(self):
        for bad in (float('nan'), -0.1, float('inf')):
            self.rows[0]['timestamps'][0]['start'] = bad
            self.save()
            with self.assertRaisesRegex(ValueError, 'finite|nonnegative'):
                self.dataset(min_duration=2)
        self.rows[0]['timestamps'][0]['start'] = 0.8
        self.rows[0]['timestamps'][0]['end'] = 0.9
        self.save()
        with self.assertRaisesRegex(ValueError, 'ordered'):
            self.dataset()

    def test_zero_intervals_spill_and_overflow(self):
        for unit in self.rows[0]['timestamps']:
            unit['start'] = unit['end'] = 0
        self.save()
        sample = self.dataset()[0]
        self.assertEqual(sample['text_tokens'][:3].tolist(), [10, 11, 12])
        self.assertEqual(sample['alignment']['collision_spills'], 2)
        self.codec.frames = 2
        with self.assertRaisesRegex(ValueError, self.rows[0]['id'] + '.*overflow'):
            self.dataset()[0]

    def test_start_overrun_not_clipped(self):
        self.rows[0]['timestamps'][1].update(start=1.6, end=1.65)
        self.save()
        with self.assertRaisesRegex(ValueError, 'overflow'):
            self.dataset()[0]

    def test_alignment_coverage_and_id_validation(self):
        self.rows[0]['timestamps'][1]['text'] = 'missing'
        self.save()
        with self.assertRaisesRegex(ValueError, 'match'):
            self.dataset()
        self.rows[0] = self.row()
        self.rows[0]['id'] = 'unparseable'
        self.save()
        with self.assertRaisesRegex(ValueError, 'program/session'):
            self.dataset()

    def test_remap(self):
        old = '/unavailable/data/emilia-en-100h'
        for key in ('metadata', 'audio'):
            self.rows[0][key] = self.rows[0][key].replace(str(self.root), old)
        self.rows[0]['config']['datasets'] = [old]
        self.save()
        sample = self.dataset(data_root=self.base)[0]
        self.assertTrue(sample['metadata'].startswith(str(self.base.resolve())))

    def test_slow_tokenizer_preserves_ids(self):
        class Slow(Tokenizer):
            def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
                if return_offsets_mapping:
                    raise NotImplementedError
                return {'input_ids': [1, 2, 3, 4]}
        dataset = EmiliaDataset(self.manifest, Slow(), self.codec, 99, validation_fraction=0)
        sample = dataset[0]
        self.assertEqual(sample['text_tokens'][sample['text_tokens'] != 99].tolist(), [1, 2, 3, 4])
        self.assertEqual(sample['alignment']['token_alignment'], 'estimated_uniform_token_positions')

    def test_cache_hit_and_audio_identity_changes(self):
        cache = self.base / 'cache'
        dataset = self.dataset(cache_dir=cache)
        first = dataset[0]
        second = dataset[0]
        self.assertEqual(self.codec.calls, 1)
        self.assertTrue(torch.equal(first['text_tokens'], second['text_tokens']))
        Path(self.rows[0]['audio']).write_bytes(b'new audio')
        dataset[0]
        self.assertEqual(self.codec.calls, 2)
        self.codec.cache_identity = 'mock-codec-v2'
        self.dataset(cache_dir=cache)[0]
        self.assertEqual(self.codec.calls, 3)
        self.assertFalse(list(cache.glob('*.tmp')))

    def test_cache_corruption_and_metadata_staleness(self):
        cache = self.base / 'cache'
        dataset = self.dataset(cache_dir=cache)
        dataset[0]
        path = next(cache.glob('*.pt'))
        payload = torch.load(path, weights_only=True)
        payload['sample']['audio_tokens'][0, 0] = 2
        torch.save(payload, path)
        dataset[0]
        self.assertEqual(self.codec.calls, 2)
        Path(self.rows[0]['metadata']).write_text('changed')
        with self.assertRaisesRegex(ValueError, 'checksum'):
            dataset[0]

    def test_cache_requires_explicit_identities(self):
        class AnonymousCodec(Codec):
            cache_identity = None
        with self.assertRaisesRegex(ValueError, 'cache_identity'):
            EmiliaDataset(self.manifest, Tokenizer(), AnonymousCodec(), 99,
                          validation_fraction=0, cache_dir=self.base / 'cache')

    def test_bad_codes_fail_with_id(self):
        class BadCodec(Codec):
            def __call__(self, waveform):
                return torch.full((8, 20), 32)
        dataset = EmiliaDataset(self.manifest, Tokenizer(), BadCodec(), 99, validation_fraction=0)
        with self.assertRaisesRegex(ValueError, self.rows[0]['id'] + '.*cardinality'):
            dataset[0]

    def test_punctuation_and_non_character_tokenization(self):
        self.rows[0]['text'] = '你好世界。'
        self.rows[0]['timestamps'] = [
            {'text': char, 'start': i * 0.2, 'end': (i + 1) * 0.2}
            for i, char in enumerate('你好世界')]
        self.save()
        class Chinese(Tokenizer):
            def __call__(self, text, **kwargs):
                return {'input_ids': [20, 21, 22],
                        'offset_mapping': [(0, 2), (2, 4), (4, 5)]}
        dataset = EmiliaDataset(self.manifest, Chinese(), self.codec, 99, validation_fraction=0)
        sample = dataset[0]
        self.assertEqual(sample['text_tokens'][sample['text_tokens'] != 99].tolist(), [20, 21, 22])

    def test_schema_and_malformed_journal(self):
        self.rows[0]['config']['schema'] = 1
        self.save()
        with self.assertRaisesRegex(ValueError, 'schema 2'):
            self.dataset()
        self.manifest.write_text('{broken json')
        with self.assertRaisesRegex(ValueError, 'aligned.jsonl:1'):
            self.dataset()

    def test_cache_tokenizer_identity_invalidates(self):
        cache = self.base / 'cache'
        self.dataset(cache_dir=cache)[0]
        class Changed(Tokenizer):
            cache_identity = 'mock-tokenizer-v2'
        dataset = EmiliaDataset(self.manifest, Changed(), self.codec, 99,
                                validation_fraction=0, cache_dir=cache)
        dataset[0]
        self.assertEqual(self.codec.calls, 2)

    def test_fractional_duration_uses_fixed_codec_clock(self):
        self.rows[0]['duration'] = 1.03
        self.rows[0]['timestamps'][1].update(start=0.795, end=1.03)
        self.save()
        class RoundedCodec(Codec):
            def __call__(self, waveform):
                self.calls += 1
                self.asserted_waveform = waveform
                assert waveform.shape == (24720,)
                assert waveform.dtype == torch.float32 and waveform.device.type == 'cpu'
                return torch.zeros(8, 13, dtype=torch.long)
        codec = RoundedCodec()
        dataset = EmiliaDataset(self.manifest, Tokenizer(), codec, 99, validation_fraction=0)
        with patch('dataset.emilia._load_audio', return_value=torch.zeros(24720)):
            sample = dataset[0]
        # floor(.795 * 12.5) == 9, but floor(.795 * 13 / 1.03) == 10.
        self.assertEqual(sample['text_tokens'][9].item(), 12)
        self.assertEqual(sample['text_tokens'][10].item(), 99)
        self.assertEqual(sample['text_tokens'][sample['text_tokens'] != 99].tolist(), [10, 11, 12])
        self.assertEqual((sample['sample_rate'], sample['frame_rate'], sample['num_frames']),
                         (24000, 12.5, 13))

    def test_fast_tokenizer_identity_ignores_unsafe_constructor_kwargs(self):
        class AddedToken:
            def __str__(self):
                return '<special>'
        class Backend:
            def to_str(self):
                return '{"vocab":{"hello":10},"normalizer":null}'
        class Fast(Tokenizer):
            cache_identity = None
            backend_tokenizer = Backend()
            special_tokens_map = {'eos_token': AddedToken(),
                                  'additional_special_tokens': [AddedToken(), '<extra>']}
            init_kwargs = {'token': AddedToken(), 'path': Path('/local/model'),
                           'torch_dtype': torch.bfloat16}
        cache = self.base / 'cache'
        first = EmiliaDataset(self.manifest, Fast(), self.codec, 99,
                              validation_fraction=0, cache_dir=cache)
        first[0]
        second = EmiliaDataset(self.manifest, Fast(), self.codec, 99,
                               validation_fraction=0, cache_dir=cache)
        second[0]
        self.assertEqual(self.codec.calls, 1)
        self.assertEqual(first.fingerprint, second.fingerprint)
        identity = first.identities['tokenizer']
        self.assertNotIn('init_kwargs', identity)
        self.assertEqual(identity['special_tokens']['additional_special_tokens'],
                         ['<special>', '<extra>'])
        json.dumps(identity)

    def test_cache_rejects_wrong_frame_metadata_even_with_valid_checksum(self):
        from dataset.emilia import _sample_digest
        cache = self.base / 'cache'
        dataset = self.dataset(cache_dir=cache)
        dataset[0]
        path = next(cache.glob('*.pt'))
        for field, bad_value in [('sample_rate', 16000), ('frame_rate', 13.0), ('num_frames', 19)]:
            with self.subTest(field=field):
                payload = torch.load(path, weights_only=True)
                payload['sample'][field] = bad_value
                payload['sample_sha256'] = _sample_digest(payload['sample'])
                torch.save(payload, path)
                calls = self.codec.calls
                sample = dataset[0]
                self.assertEqual(self.codec.calls, calls + 1)
                self.assertEqual((sample['sample_rate'], sample['frame_rate'], sample['num_frames']),
                                 (24000, 12.5, 20))
        calls = self.codec.calls
        dataset[0]
        self.assertEqual(self.codec.calls, calls)

    def test_audio_mono_resample(self):
        self.audio.stop()
        try:
            import numpy as np
            import soundfile as sf
            from dataset.emilia import _load_audio
        except ImportError:
            self.skipTest('soundfile/scipy unavailable')
        path = self.base / 'stereo.wav'
        sf.write(path, np.column_stack([np.ones(16000) * 0.2, np.ones(16000) * 0.4]), 16000, subtype='FLOAT')
        waveform = _load_audio(path)
        self.assertEqual(waveform.shape, (24000,))
        self.assertEqual(waveform.dtype, torch.float32)
        self.assertAlmostEqual(waveform[100:-100].mean().item(), 0.3, places=4)


if __name__ == '__main__':
    unittest.main()
