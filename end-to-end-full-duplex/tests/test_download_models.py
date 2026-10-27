"""Offline download-entry tests; no model dependencies, files or network needed."""

import importlib.util
import io
from contextlib import ExitStack, redirect_stderr, redirect_stdout
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import Mock, patch


SCRIPT = Path(__file__).resolve().parents[1] / "script" / "download_models.py"
SPEC = importlib.util.spec_from_file_location("download_models", SCRIPT)
download_models = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(download_models)


class DownloadModelsTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stdout = io.StringIO()
        self.stderr = io.StringIO()
        self.stack.enter_context(redirect_stdout(self.stdout))
        self.stack.enter_context(redirect_stderr(self.stderr))
        self.mkdir = self.stack.enter_context(patch.object(Path, "mkdir", autospec=True))
        self.snapshot_download = Mock()
        module = types.ModuleType("modelscope")
        module.snapshot_download = self.snapshot_download
        self.stack.enter_context(patch.dict(sys.modules, {"modelscope": module}))

    def test_default_downloads_complete_7b_checkpoint(self):
        self.assertEqual(download_models.main([]), 0)
        target = SCRIPT.parents[1] / "model" / "Qwen2.5-Omni-7B"
        self.mkdir.assert_called_once_with(target, parents=True, exist_ok=True)
        self.snapshot_download.assert_called_once_with(
            "Qwen/Qwen2.5-Omni-7B", local_dir=str(target)
        )
        self.assertIn("模型下载完成", self.stdout.getvalue())

    def test_3b_has_separate_default_directory(self):
        self.assertEqual(
            download_models.main(["--model-id", "Qwen/Qwen2.5-Omni-3B"]), 0
        )
        self.snapshot_download.assert_called_once_with(
            "Qwen/Qwen2.5-Omni-3B",
            local_dir=str(SCRIPT.parents[1] / "model" / "Qwen2.5-Omni-3B"),
        )

    def test_explicit_directory(self):
        target = SCRIPT.parents[1] / "model" / "custom-omni"
        self.assertEqual(download_models.main(["--local-dir", str(target)]), 0)
        self.snapshot_download.assert_called_once_with(
            download_models.MODEL_ID, local_dir=str(target)
        )

    def test_relative_directory_is_resolved(self):
        self.assertEqual(download_models.main(["--local-dir", "custom-omni"]), 0)
        self.snapshot_download.assert_called_once_with(
            download_models.MODEL_ID, local_dir=str(Path("custom-omni").resolve())
        )

    def test_home_directory_is_expanded(self):
        self.assertEqual(download_models.main(["--local-dir", "~/custom-omni"]), 0)
        self.snapshot_download.assert_called_once_with(
            download_models.MODEL_ID,
            local_dir=str(Path.home() / "custom-omni"),
        )

    def test_help_does_not_need_modelscope_or_create_directories(self):
        with patch.dict(sys.modules, {"modelscope": None}):
            with self.assertRaises(SystemExit) as result:
                download_models.main(["--help"])
        self.assertEqual(result.exception.code, 0)
        self.assertIn("Qwen/Qwen2.5-Omni-7B", self.stdout.getvalue())
        self.assertIn("--local-dir", self.stdout.getvalue())
        self.mkdir.assert_not_called()
        self.snapshot_download.assert_not_called()

    def test_invalid_model_is_rejected_before_download(self):
        with self.assertRaises(SystemExit) as result:
            download_models.main(["--model-id", "Qwen/Qwen3-4B-Instruct-2507"])
        self.assertEqual(result.exception.code, 2)
        self.mkdir.assert_not_called()
        self.snapshot_download.assert_not_called()

    def test_missing_modelscope_reports_install_command(self):
        with patch.dict(sys.modules, {"modelscope": None}):
            self.assertEqual(download_models.main([]), 1)
        self.assertIn("-m pip install modelscope", self.stderr.getvalue())
        self.mkdir.assert_not_called()

    def test_missing_transitive_dependency_is_not_misreported(self):
        original_import = __import__

        def import_with_missing_dependency(name, *args, **kwargs):
            if name == "modelscope":
                raise ModuleNotFoundError("No module named 'requests'", name="requests")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=import_with_missing_dependency):
            with self.assertRaises(ModuleNotFoundError) as result:
                download_models.main([])
        self.assertEqual(result.exception.name, "requests")
        self.mkdir.assert_not_called()

    def test_download_failure_does_not_report_success(self):
        self.snapshot_download.side_effect = RuntimeError("offline")
        self.assertEqual(download_models.main([]), 1)
        self.assertIn("offline", self.stderr.getvalue())
        self.assertNotIn("模型下载完成", self.stdout.getvalue())

    def test_directory_failure_does_not_start_download(self):
        self.mkdir.side_effect = PermissionError("read-only directory")
        self.assertEqual(download_models.main([]), 1)
        self.snapshot_download.assert_not_called()
        self.assertIn("read-only directory", self.stderr.getvalue())
        self.assertNotIn("模型下载完成", self.stdout.getvalue())

    def test_interruption_can_be_retried(self):
        self.snapshot_download.side_effect = KeyboardInterrupt
        self.assertEqual(download_models.main([]), 130)
        self.assertIn("可重新运行", self.stderr.getvalue())
        self.assertNotIn("模型下载完成", self.stdout.getvalue())
        self.snapshot_download.side_effect = None
        self.assertEqual(download_models.main([]), 0)
        self.assertEqual(self.snapshot_download.call_count, 2)


if __name__ == "__main__":
    unittest.main()
