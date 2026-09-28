"""Offline tests for scenario-group dataset splitting."""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "split_turn_intent_dataset.py"
spec = importlib.util.spec_from_file_location("turn_intent_splitter", SCRIPT)
splitter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(splitter)


class SplitterTests(unittest.TestCase):
    def make_input(self, directory, group_count=12):
        data_path = directory / "data.jsonl"
        manifest_path = directory / "manifest.jsonl"
        families = ("出行交通", "消费服务", "工作协作", "学习解释")
        with data_path.open("w", encoding="utf-8") as data, manifest_path.open(
            "w", encoding="utf-8"
        ) as manifest:
            for group_number in range(group_count):
                group = f"seed42_scene{group_number}_family"
                family = families[group_number % len(families)]
                topic = f"topic-{group_number}"
                for label in splitter.LABELS:
                    data.write(json.dumps({
                        "assistant_text": "完整回答",
                        "user_asr": "用户文本",
                        "label": label,
                        "reason": "测试原因",
                    }, ensure_ascii=False) + "\n")
                    manifest.write(json.dumps({
                        "slot": {
                            "scenario_seed": group,
                            "topic_family": family,
                            "topic": topic,
                        }
                    }, ensure_ascii=False) + "\n")
        return data_path, manifest_path

    def test_ood_and_regular_splits_keep_groups_and_labels_together(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            data_path, manifest_path = self.make_input(directory)
            records = splitter.read_records(data_path, manifest_path)
            groups = splitter.group_records(records)
            assignments = splitter.assign_splits(
                groups, seed=7, train_ratio=0.75, validation_ratio=0.125,
                ood_families={"学习解释"},
            )
            summary = splitter.make_summary(groups, assignments)
            self.assertEqual(summary["ood_test"]["records"], 9)
            self.assertEqual(summary["ood_test"]["incomplete_groups"], 0)
            self.assertEqual(sum(summary[split]["records"] for split in summary), 36)
            for group in groups:
                labels = {record["label"] for record in group["records"]}
                self.assertEqual(labels, set(splitter.LABELS))
                self.assertEqual(len({assignments[group["group"]]}), 1)
            for group in groups:
                if group["family"] == "学习解释":
                    self.assertEqual(assignments[group["group"]], "ood_test")
                else:
                    self.assertNotEqual(assignments[group["group"]], "ood_test")

    def test_write_outputs_preserves_rows_and_emits_audit_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            data_path, manifest_path = self.make_input(directory, group_count=4)
            groups = splitter.group_records(splitter.read_records(data_path, manifest_path))
            assignments = {group["group"]: "train" for group in groups}
            summary = splitter.make_summary(groups, assignments)
            output_dir = directory / "splits"
            splitter.write_outputs(
                groups, assignments, output_dir, summary, seed=42,
                ood_families=set(),
            )
            self.assertEqual(len(output_dir.joinpath("train.jsonl").read_text().splitlines()), 12)
            self.assertEqual(output_dir.joinpath("validation.jsonl").read_text(), "")
            self.assertEqual(output_dir.joinpath("test.jsonl").read_text(), "")
            self.assertEqual(output_dir.joinpath("ood_test.jsonl").read_text(), "")
            split_manifest = [
                json.loads(line)
                for line in output_dir.joinpath("split_manifest.jsonl").read_text().splitlines()
            ]
            self.assertEqual(len(split_manifest), 4)
            self.assertTrue(all(item["split"] == "train" for item in split_manifest))
            self.assertEqual(json.loads(output_dir.joinpath("summary.json").read_text())["splits"]["train"]["groups"], 4)

    def test_existing_output_directory_is_rejected_by_main(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            data_path, manifest_path = self.make_input(directory, group_count=1)
            output_dir = directory / "splits"
            output_dir.mkdir()
            self.assertEqual(splitter.main([
                "--input", str(data_path), "--manifest", str(manifest_path),
                "--output-dir", str(output_dir), "--ood-families", "出行交通",
            ]), 1)


if __name__ == "__main__":
    unittest.main()
