"""Offline tests: no API credentials, model downloads, or network required."""

import importlib.util
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "generate_turn_intent_dataset.py"
spec = importlib.util.spec_from_file_location("turn_intent_generator", SCRIPT)
generator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(generator)


def row(slot, **updates):
    value = {"slot_id": slot["slot_id"], "assistant_text": f"助手正在讲方案{slot['slot_id']}",
             "user_asr": f"用户当前文本{slot['slot_id']}", "label": slot["label"],
             "reason": "这是测试结构用的判断依据。"}
    if "user_asr_chars" in slot:
        lower, upper = slot["user_asr_chars"]
        value["user_asr"] = value["user_asr"].ljust(lower, "字")[:upper]
    if "assistant_text_chars" in slot:
        lower, upper = slot["assistant_text_chars"]
        value["assistant_text"] = value["assistant_text"].ljust(lower, "字")[:upper]
    value.update(updates)
    return value


def response(samples=None, content=None, finish_reason="stop"):
    if content is None:
        content = json.dumps({"samples": samples}, ensure_ascii=False)
    return SimpleNamespace(choices=[SimpleNamespace(
        finish_reason=finish_reason, message=SimpleNamespace(content=content))])


class FakeClient:
    def __init__(self, handler):
        self.chat = SimpleNamespace(completions=self)
        self.calls = []
        self.handler = handler

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self.handler(kwargs)


class GeneratorTests(unittest.TestCase):
    def test_plan_is_balanced_and_reproducible(self):
        plan = generator.make_plan(101, 42)
        self.assertEqual(plan, generator.make_plan(101, 42))
        self.assertEqual(len({s["slot_id"] for s in plan}), 101)
        counts = generator.Counter(s["label"] for s in plan)
        self.assertLessEqual(max(counts.values()) - min(counts.values()), 1)
        self.assertNotEqual(plan, generator.make_plan(101, 43))

    def test_topic_contrasts_cover_all_labels_and_lengths(self):
        plan = generator.make_plan(27, 42, topic="航班改签")
        for start in range(0, 27, 3):
            group = plan[start:start + 3]
            self.assertEqual({s["label"] for s in group}, set(generator.LABELS))
            self.assertEqual({s["topic"] for s in group}, {"航班改签"})
            self.assertEqual(len({s["topic_group"] for s in group}), 1)
        self.assertEqual({s["topic_family"] for s in plan}, {"出行交通"})
        self.assertEqual(len({s["scenario_seed"] for s in plan}), 9)
        self.assertTrue(all("user_asr_chars" not in s and "assistant_text_chars" not in s for s in plan))
        self.assertGreaterEqual(len({s["length_hint"] for s in plan}), 2)
        self.assertLessEqual(max(len(s["assistant_context"]) for s in plan), 20)

    def test_unfixed_plan_spans_topic_families_and_labels(self):
        plan = generator.make_plan(72, 42)
        families = {s["topic_family"] for s in plan}
        self.assertEqual(families, set(generator.TOPIC_FAMILIES))
        for topic in generator.TOPICS:
            labels = {s["label"] for s in plan if s["topic"] == topic}
            self.assertEqual(labels, set(generator.LABELS))
        for family in families:
            labels = {s["label"] for s in plan if s["topic_family"] == family}
            self.assertEqual(labels, set(generator.LABELS))
        for group in range(24):
            group_slots = plan[group * 3:(group + 1) * 3]
            self.assertEqual(len({s["scenario_seed"] for s in group_slots}), 1)
            self.assertEqual({s["label"] for s in group_slots}, set(generator.LABELS))

    def test_random_length_preferences_are_reproducible(self):
        plan = generator.make_plan(300, 42)
        self.assertEqual(plan, generator.make_plan(300, 42))
        self.assertGreaterEqual(len({s["length_hint"] for s in plan}), 3)
        self.assertGreaterEqual(len({s["assistant_context"] for s in plan}), 3)
        self.assertTrue(all("user_asr_chars" not in s for s in plan))
        self.assertTrue(all("assistant_text_chars" not in s for s in plan))

    def test_assistant_surface_and_asr_punctuation_vary(self):
        plan = generator.make_plan(36, 42, topic="航班改签")
        self.assertEqual(set(s["assistant_surface"] for s in plan), set(generator.ASSISTANT_SURFACES))
        self.assertGreaterEqual(len({s["user_punctuation"] for s in plan}), 4)
        messages = generator.make_messages(plan[:3], [], 1200, 300)
        system_text = messages[0]["content"]
        request = json.loads(messages[1]["content"])
        self.assertIn("自然截断", system_text)
        self.assertIn("user_punctuation", request["instructions"])
        self.assertIn("assistant_surface", request["instructions"])

    def test_partial_assistant_text_and_punctuation_are_valid(self):
        slot = {"slot_id": 0, "label": "YIELD", "user_asr_chars": [1, 10]}
        sample = row(slot, assistant_text="改签费是八十元，支", user_asr="我想改签？")
        accepted, rejected = generator.validate_samples(json.dumps({"samples": [sample]}), [slot], set())
        self.assertEqual(len(accepted), 1)
        self.assertFalse(rejected)

    def test_ellipsis_and_length_band_rejected(self):
        slot = {"slot_id": 0, "label": "WAIT", "user_asr_chars": [11, 30]}
        for field in ("assistant_text", "user_asr", "reason"):
            for dots in ("……", "...", "．．．", "⋯", "︙"):
                payload = json.dumps({"samples": [row(slot, **{field: "这个具体事项" + dots})]})
                accepted, rejected = generator.validate_samples(payload, [slot], set())
                self.assertFalse(accepted)
                self.assertEqual(rejected["ellipsis"], 1)
        for text in ("那个", "字" * 31):
            accepted, rejected = generator.validate_samples(
                json.dumps({"samples": [row(slot, user_asr=text)]}), [slot], set())
            self.assertEqual(len(accepted), 1)
            self.assertFalse(rejected)

    def test_retry_keeps_siblings_beyond_recent_window(self):
        requests = []

        def handler(kwargs):
            request = json.loads(kwargs["messages"][1]["content"])
            requests.append(request)
            slots = request["slots"]
            if len(requests) == 1:
                slots = slots[1:]
            return response([row(slot) for slot in slots])

        args = generator.parse_args(["--model", "fake", "--count", "18", "--batch-size", "18"])
        manifest = io.StringIO()
        self.assertEqual(generator.generate(FakeClient(handler), args, io.StringIO(), manifest), 0)
        self.assertEqual(len(requests), 2)
        self.assertEqual([s["slot_id"] for s in requests[1]["slots"]], [0])
        siblings = requests[1]["accepted_topic_siblings"]
        self.assertEqual(len(siblings), 2)
        self.assertEqual({s["assistant_text"] for s in siblings}, {"助手正在讲方案1", "助手正在讲方案2"})
        records = [json.loads(line) for line in manifest.getvalue().splitlines()]
        group = [r for r in records if r["slot"]["slot_id"] < 3]
        self.assertEqual(len({r["generation_group"] for r in group}), 1)

    def test_preview_uses_execution_batch_selection(self):
        stdout = io.StringIO()
        with patch("sys.stdout", stdout):
            self.assertEqual(generator.main(["--dry-run", "--count", "6", "--batch-size", "4"]), 0)
        preview = json.loads(stdout.getvalue())
        slots = json.loads(preview["first_request"][1]["content"])["slots"]
        self.assertEqual(slots, generator.select_batch(generator.make_plan(6, 42), 4))
        self.assertEqual(len(slots), 3)

    def test_validation_rejects_conflicts_duplicates_and_non_four_field_payload(self):
        slots = [{"slot_id": i, "label": "YIELD"} for i in range(8)]
        samples = [row(slots[0]), row(slots[1], label="WAIT"), row(slots[2], reason=""),
                   row(slots[3], user_asr=""), row(slots[4], reason="<answer>YIELD</answer>"),
                   row(slots[5], assistant_text=samples_text(slots[0]), user_asr="用户当前文本0！"),
                   row(slots[6], extra="forbidden"), row(slots[7], user_asr=None)]
        accepted, rejected = generator.validate_samples(json.dumps({"samples": samples}), slots, set())
        self.assertEqual(len(accepted), 1)
        self.assertEqual(set(accepted[0][1]), generator.FIELDS)
        self.assertEqual(sum(rejected.values()), 7)
        self.assertEqual(rejected["duplicate_input"], 1)

    def test_echoed_plan_fields_accepted_but_unknown_extras_rejected(self):
        slot = {"slot_id": 0, "label": "YIELD", "special_case": "wake_start"}
        echoed = row(slot, assistant_text="", user_asr="我想改签",
                     special_case="wake_start")
        accepted, rejected = generator.validate_samples(
            json.dumps({"samples": [echoed]}), [slot], set())
        self.assertEqual(len(accepted), 1)
        self.assertFalse(rejected)
        # Output stays four-field even though the provider echoed plan fields.
        self.assertEqual(set(accepted[0][1]), generator.FIELDS)
        # A wrong echoed value must not be silently accepted.
        wrong = row(slot, assistant_text="", user_asr="我想改签",
                    special_case="assistant_echo")
        self.assertEqual(
            generator.validate_samples(json.dumps({"samples": [wrong]}), [slot], set())[1]["fields"], 1)
        # Unknown extras remain rejected.
        unknown = row(slot, assistant_text="", user_asr="我想改签", extra="x")
        self.assertEqual(
            generator.validate_samples(json.dumps({"samples": [unknown]}), [slot], set())[1]["fields"], 1)

    def test_empty_wait_allowed_and_seen_input_rejected(self):
        slots = [{"slot_id": 0, "label": "WAIT"}]
        sample = row(slots[0], user_asr="")
        content = json.dumps({"samples": [sample]})
        self.assertEqual(len(generator.validate_samples(content, slots, set())[0]), 1)
        seen = {generator.fingerprint(sample)}
        self.assertEqual(generator.validate_samples(content, slots, seen)[1]["duplicate_input"], 1)

    def test_exact_count_partial_acceptance_and_manifest(self):
        attempts = 0

        def handler(kwargs):
            nonlocal attempts
            attempts += 1
            slots = json.loads(kwargs["messages"][1]["content"])["slots"]
            samples = [row(s) for s in slots]
            if attempts == 1:
                samples = samples[:1]
            return response(samples)

        client = FakeClient(handler)
        args = generator.parse_args(["--model", "fake", "--count", "11", "--batch-size", "4"])
        data, manifest = io.StringIO(), io.StringIO()
        self.assertEqual(generator.generate(client, args, data, manifest), 0)
        records = [json.loads(line) for line in data.getvalue().splitlines()]
        metadata = [json.loads(line) for line in manifest.getvalue().splitlines()]
        self.assertEqual(len(records), 11)
        self.assertEqual(len({generator.fingerprint(r) for r in records}), 11)
        self.assertTrue(all(set(r) == generator.FIELDS for r in records))
        self.assertEqual([m["line"] for m in metadata], list(range(1, 12)))
        self.assertEqual({m["source"] for m in metadata}, {"synthetic_unreviewed"})
        self.assertLessEqual(len(client.calls), args.max_requests)

    def test_malformed_and_truncated_outputs_exhaust_budget(self):
        for bad in (response(content="not json"), response([], finish_reason="length")):
            client = FakeClient(lambda kwargs: bad)
            args = generator.parse_args(["--model", "fake", "--count", "2", "--max-requests", "2"])
            data, manifest = io.StringIO(), io.StringIO()
            self.assertEqual(generator.generate(client, args, data, manifest), 1)
            self.assertEqual(len(client.calls), 2)
            self.assertEqual(data.getvalue(), "")

    def test_transient_errors_retry_but_auth_errors_stop_without_secret(self):
        class ProviderError(Exception):
            status_code = 401

        for status, expected_calls in ((401, 1), (429, 2), (503, 2)):
            ProviderError.status_code = status

            def handler(kwargs):
                raise ProviderError("secret-test-key")

            client = FakeClient(handler)
            args = generator.parse_args(["--model", "fake", "--count", "1", "--max-requests", "2"])
            stderr = io.StringIO()
            with patch.object(generator.time, "sleep"), patch("sys.stderr", stderr):
                self.assertEqual(generator.generate(client, args, io.StringIO(), io.StringIO()), 1)
            self.assertEqual(len(client.calls), expected_calls)
            self.assertNotIn("secret-test-key", stderr.getvalue())

    def test_dry_run_has_no_credentials_network_or_files(self):
        with tempfile.TemporaryDirectory() as temp, patch.dict(os.environ, {}, clear=True):
            output = Path(temp) / "data.jsonl"
            stdout = io.StringIO()
            with patch("sys.stdout", stdout):
                code = generator.main(["--dry-run", "--count", "7", "--output", str(output)])
            self.assertEqual(code, 0)
            self.assertEqual(json.loads(stdout.getvalue())["count"], 7)
            self.assertEqual(list(Path(temp).iterdir()), [])

    def test_existing_files_not_overwritten(self):
        with tempfile.TemporaryDirectory() as temp, patch.dict(os.environ, {"OPENAI_API_KEY": "fake"}):
            output = Path(temp) / "data.jsonl"
            output.write_text("keep me", encoding="utf-8")
            code = generator.main(["--model", "fake", "--output", str(output)])
            self.assertEqual(code, 1)
            self.assertEqual(output.read_text(encoding="utf-8"), "keep me")

    def test_invalid_counts_rejected(self):
        for flags in (["--count", "0"], ["--batch-size", "-1"], ["--max-requests", "0"]):
            with self.assertRaises(SystemExit), patch("sys.stderr", io.StringIO()):
                generator.parse_args(["--dry-run", *flags])


def samples_text(slot):
    return row(slot)["assistant_text"]


if __name__ == "__main__":
    unittest.main()
