"""Offline CLI conversion with a tiny real native checkpoint, not pretrained downloads."""

import contextlib
import importlib.util
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch

from qwen_duplex import load_duplex_checkpoint, validate_action_tokens
from test_qwen_duplex_tokens import make_processor, make_thinker


SCRIPT = Path(__file__).resolve().parents[1] / "script" / "adapt_omni_actions.py"
spec = importlib.util.spec_from_file_location("adapt_omni_actions", SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class AdaptActionsCLITests(unittest.TestCase):
    def test_help(self):
        with patch.object(sys, "argv", [str(SCRIPT), "--help"]), contextlib.redirect_stdout(io.StringIO()) as out:
            with self.assertRaises(SystemExit) as exit_:
                module.main()
        self.assertEqual(exit_.exception.code, 0)
        self.assertIn("--thinker-only", out.getvalue())

    def test_tiny_native_checkpoint_conversion_and_overwrite_refusal(self):
        with tempfile.TemporaryDirectory(prefix="qwen-actions-cli-") as directory:
            root = Path(directory)
            source, target = root / "source", root / "adapted"
            model, processor = make_thinker(), make_processor()
            old_rows = model.get_input_embeddings().weight.detach().clone()
            model.save_pretrained(source)
            processor.save_pretrained(source, legacy_serialization=False)
            argv = [str(SCRIPT), "--model-dir", str(source), "--output-dir", str(target),
                    "--thinker-only", "--dtype", "float32"]
            with patch.object(sys, "argv", argv), contextlib.redirect_stdout(io.StringIO()) as out:
                module.main()
            self.assertIn('"trained_action_policy": false', out.getvalue())
            loaded, loaded_processor = load_duplex_checkpoint(target)
            self.assertEqual(len(validate_action_tokens(loaded, loaded_processor.tokenizer)), 6)
            self.assertTrue(torch.equal(loaded.get_input_embeddings().weight[:len(old_rows)], old_rows))
            original_metadata = (target / "duplex_tokens.json").read_bytes()
            with patch.object(sys, "argv", argv), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as exit_:
                    module.main()
            self.assertEqual(exit_.exception.code, 2)
            self.assertEqual((target / "duplex_tokens.json").read_bytes(), original_metadata)


if __name__ == "__main__":
    unittest.main()
