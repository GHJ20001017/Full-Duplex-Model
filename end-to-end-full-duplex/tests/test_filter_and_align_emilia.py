"""Dependency-free tests; mocked inference does not validate model quality."""
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

spec = importlib.util.spec_from_file_location("curate", Path(__file__).resolve().parents[1] / "scripts/filter_and_align_emilia.py")
curate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(curate)


class CurationTests(unittest.TestCase):
    def test_metrics(self):
        self.assertEqual(curate.compare("你好，世界！", "你好世界", "Chinese")["error_rate"], 0)
        self.assertEqual(curate.compare("Hello world", "hello there", "English")["error_rate"], .5)

    def test_alignment_preserved(self):
        items = [SimpleNamespace(text="hello", start_time=0, end_time=end)
                 for end in (0, 2, -1, float("nan"))]
        rows = curate.serialize_timestamps(items)
        self.assertEqual([row["end"] for row in rows], [0, 2, -1, "nan"])
        json.dumps(rows, allow_nan=False)
        self.assertEqual(curate.serialize_timestamps([]), [])

    def test_pipeline_and_resume(self):
        calls = []
        class ASR:
            @classmethod
            def from_pretrained(cls, path, **kwargs):
                self.assertTrue(kwargs["local_files_only"])
                return cls()

            def transcribe(self, audio, language):
                calls.append(language)
                return [SimpleNamespace(text="你好" if language == "Chinese" else "different words")]

        class Aligner(ASR):
            def align(self, audio, text, language):
                return [[SimpleNamespace(text=text, start_time=0, end_time=0),
                         SimpleNamespace(text="extra", start_time=-1, end_time=2)]]

        torch = ModuleType("torch")
        torch.bfloat16 = "bf16"
        api = ModuleType("qwen_asr")
        api.Qwen3ASRModel, api.Qwen3ForcedAligner = ASR, Aligner
        utils = ModuleType("qwen_asr.inference.utils")
        utils.normalize_audios = lambda audio: [[0] * 16000]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for language, name, text in [("zh", "emilia-zh-200h", "你好"), ("en", "emilia-en-100h", "hello world")]:
                extracted = root / name / "extracted"
                extracted.mkdir(parents=True)
                (extracted / "sample.json").write_text(json.dumps({"id": language, "text": text, "language": language, "wav": "wrong/original.mp3"}))
                (extracted / "sample.mp3").touch()
            model = root / "model"
            model.mkdir()
            (model / "config.json").write_text("{}")
            output = root / "out.jsonl"
            args = ["--data-root", str(root), "--output", str(output), "--asr-model", str(model), "--aligner-model", str(model), "--max-samples", "1"]
            with patch.dict(sys.modules, {"torch": torch, "qwen_asr": api, "qwen_asr.inference.utils": utils}):
                self.assertEqual(curate.main(args), 0)
                rows = [json.loads(line) for line in output.read_text().splitlines()]
                self.assertEqual([row["status"] for row in rows], ["generated", "generated"])
                self.assertGreater(rows[1]["error_rate"], .2)
                for row in rows:
                    self.assertEqual(row["quality_status"], "not_evaluated")
                    self.assertEqual(len(row["timestamps"]), 2)
                    self.assertEqual(row["timestamps"][0]["end"], 0)
                    self.assertEqual(row["timestamps"][1]["end"], 2)
                self.assertEqual(calls, ["Chinese", "English"])
                self.assertEqual(curate.main(args + ["--resume"]), 0)
                self.assertEqual(len(calls), 2)
                partial_args = args.copy()
                partial_args[partial_args.index(str(output))] = str(root / "partial.jsonl")
                with patch.object(ASR, "transcribe", side_effect=RuntimeError("ASR failed")):
                    self.assertEqual(curate.main(partial_args), 2)
                partial = [json.loads(line) for line in (root / "partial.jsonl").read_text().splitlines()]
                self.assertTrue(all("timestamps" in row and "asr" in row["errors"] for row in partial))
                with patch.object(Aligner, "align", side_effect=RuntimeError("Alignment failed")):
                    self.assertEqual(curate.main(partial_args + ["--resume", "--retry-errors"]), 2)
                partial = [json.loads(line) for line in (root / "partial.jsonl").read_text().splitlines()]
                self.assertTrue(all("asr_text" in row and "alignment" in row["errors"] for row in partial[-2:]))
                with self.assertRaises(SystemExit):
                    curate.main(args)
                with self.assertRaises(ValueError):
                    curate.main(args + ["--resume", "--max-cer", "0.1"])
                # A missing audio must not fall back to the JSON itself.
                (root / "emilia-en-100h/extracted/sample.mp3").unlink()
                self.assertEqual(curate.main(args + ["--dry-run"]), 2)
                with output.open("a") as handle:
                    handle.write('{"partial":')
                with self.assertRaises(ValueError):
                    curate.main(args + ["--resume"])


if __name__ == "__main__":
    unittest.main()
