import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'script'))
from probe_thinker_live_audio import append_range, positions


class LiveAudioHelpersTests(unittest.TestCase):
    def test_audio_append_ranges_have_no_repeated_positions(self):
        self.assertEqual(append_range(25, 27, 1.1), (25, 27))
        self.assertEqual(append_range(27, 30, 1.2), (27, 30))
        self.assertEqual(append_range(30, 30, 1.2), (30, 30))

    def test_positions_follow_interleaved_cache_length(self):
        self.assertEqual(positions(100, 4), [100, 101, 102, 103])
        self.assertEqual(positions(104, 1), [104])
        self.assertEqual(positions(105, 5), [105, 106, 107, 108, 109])

    def test_invalid_progress_is_rejected(self):
        with self.assertRaises(ValueError):
            append_range(30, 27, 1.1)
        for args in [(-1, 2), (0, 0)]:
            with self.assertRaises(ValueError):
                positions(*args)


if __name__ == '__main__':
    unittest.main()
