import importlib.util
from pathlib import Path
import unittest


PATH = Path(__file__).resolve().parents[1] / 'script/probe_omni_streaming_encoder.py'
SPEC = importlib.util.spec_from_file_location('probe_omni_streaming_encoder', PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class StreamingProbeHelpersTests(unittest.TestCase):
    def test_arrivals_never_exceed_available_audio(self):
        ends = MODULE.arrival_ends(4567, 22050, 100)
        self.assertEqual(ends, [2205, 4410, 4567])
        self.assertTrue(all(0 < end <= 4567 for end in ends))

    def test_exact_multiple_has_no_duplicate_end(self):
        self.assertEqual(MODULE.arrival_ends(4410, 22050, 100), [2205, 4410])

    def test_invalid_audio_or_interval_rejected(self):
        for args in [(0, 16000, 100), (1600, 16000, 0)]:
            with self.assertRaises(ValueError):
                MODULE.arrival_ends(*args)

    def test_commit_count_obeys_holdback_and_observed_length(self):
        self.assertEqual(MODULE.commit_count(25, 1, 200), 20)
        self.assertEqual(MODULE.commit_count(2, .1, 0), 2)
        self.assertEqual(MODULE.commit_count(2, .1, 200), 0)
        self.assertEqual(MODULE.commit_count(10, .4, 0), 10)

    def test_word_error_rate_counts_insertions_and_deletions(self):
        self.assertEqual(MODULE.edit_counts('hello my friend', 'hello friend')['word_edits'], 1)
        self.assertEqual(MODULE.edit_counts('hello friend', 'hello my friend')['word_edits'], 1)
        self.assertEqual(MODULE.edit_counts('Hello, it\u2019s me.', "hello it's me")['wer'], 0)

    def test_latency_percentiles(self):
        self.assertEqual(MODULE.summary([3, 1, 2]), {'count': 3, 'mean': 2, 'median': 2, 'p95': 3, 'max': 3})
        self.assertEqual(MODULE.summary([]), {'count': 0})

    def test_endpoint_padding_preserves_normal_chunks(self):
        self.assertEqual(MODULE.endpoint_padding_samples(147, 16000), 493)
        self.assertEqual(MODULE.endpoint_padding_samples(352, 16000), 288)
        self.assertEqual(MODULE.endpoint_padding_samples(640, 16000), 0)
        self.assertEqual(MODULE.endpoint_padding_samples(1600, 16000), 0)
        with self.assertRaises(ValueError):
            MODULE.endpoint_padding_samples(0, 16000)


if __name__ == '__main__':
    unittest.main()
