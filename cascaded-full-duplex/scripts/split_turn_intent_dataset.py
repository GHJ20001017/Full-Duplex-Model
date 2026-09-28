"""Split turn-intent JSONL by scenario groups without leakage.

The generator creates three-label contrast groups. This script keeps every
scenario_seed together, creates train/validation/test splits, and optionally
holds out complete topic families as an OOD test set.

Example:
  python scripts/split_turn_intent_dataset.py \
      --input turn_intent_large.jsonl \
      --manifest turn_intent_large.jsonl.manifest.jsonl \
      --output-dir splits \
      --ood-families 信息查询

The input files are never modified and output files must not already exist.
"""

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

LABELS = ("YIELD", "CONTINUE", "WAIT")
OUTPUT_NAMES = ("train.jsonl", "validation.jsonl", "test.jsonl", "ood_test.jsonl")


def read_records(data_path, manifest_path):
    data_lines = data_path.read_text(encoding="utf-8").splitlines()
    manifest_lines = manifest_path.read_text(encoding="utf-8").splitlines()
    if len(data_lines) != len(manifest_lines):
        raise ValueError(
            f"data/manifest line count mismatch: {len(data_lines)} != {len(manifest_lines)}"
        )
    records = []
    for line_number, (data_line, manifest_line) in enumerate(
        zip(data_lines, manifest_lines), start=1
    ):
        try:
            row = json.loads(data_line)
            metadata = json.loads(manifest_line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON at line {line_number}") from exc
        if not isinstance(row, dict) or set(row) != {
            "assistant_text", "user_asr", "label", "reason"
        }:
            raise ValueError(f"invalid data fields at line {line_number}")
        if not isinstance(metadata, dict) or not isinstance(metadata.get("slot"), dict):
            raise ValueError(f"invalid manifest slot at line {line_number}")
        slot = metadata["slot"]
        group = slot.get("scenario_seed") or slot.get("topic_group")
        if not group:
            raise ValueError(f"missing scenario_seed/topic_group at line {line_number}")
        family = slot.get("topic_family", "unknown")
        topic = slot.get("topic", "unknown")
        records.append({
            "line": line_number,
            "row": row,
            "group": group,
            "family": family,
            "topic": topic,
            "label": row["label"],
        })
    return records


def group_records(records):
    groups = {}
    for record in records:
        group = groups.setdefault(record["group"], {
            "group": record["group"],
            "family": record["family"],
            "topic": record["topic"],
            "records": [],
        })
        if (group["family"], group["topic"]) != (record["family"], record["topic"]):
            raise ValueError(f"group has mixed family/topic: {record['group']}")
        group["records"].append(record)
    return list(groups.values())


def parse_names(value):
    names = tuple(name.strip() for name in value.split(",") if name.strip())
    if not names:
        raise ValueError("at least one non-empty name is required")
    return set(names)


def assign_splits(groups, seed, train_ratio, validation_ratio, ood_families):
    if not 0 < train_ratio < 1 or not 0 < validation_ratio < 1:
        raise ValueError("train and validation ratios must be positive and below 1")
    if train_ratio + validation_ratio >= 1:
        raise ValueError("train and validation ratios must sum to less than 1")

    ood = [group for group in groups if group["family"] in ood_families]
    eligible = [group for group in groups if group["family"] not in ood_families]
    rng = random.Random(seed)
    rng.shuffle(eligible)
    train_count = round(len(eligible) * train_ratio)
    validation_count = round(len(eligible) * validation_ratio)
    train_end = train_count
    validation_end = train_count + validation_count
    assignments = {group["group"]: "ood_test" for group in ood}
    assignments.update({group["group"]: "train" for group in eligible[:train_end]})
    assignments.update({
        group["group"]: "validation"
        for group in eligible[train_end:validation_end]
    })
    assignments.update({
        group["group"]: "test"
        for group in eligible[validation_end:]
    })
    return assignments


def make_summary(groups, assignments):
    summary = {}
    for split in ("train", "validation", "test", "ood_test"):
        selected = [group for group in groups if assignments[group["group"]] == split]
        records = [record for group in selected for record in group["records"]]
        summary[split] = {
            "groups": len(selected),
            "records": len(records),
            "labels": dict(Counter(record["label"] for record in records)),
            "topic_families": dict(Counter(group["family"] for group in selected)),
            "topics": dict(Counter(group["topic"] for group in selected)),
            "incomplete_groups": sum(
                set(record["label"] for record in group["records"]) != set(LABELS)
                for group in selected
            ),
        }
    return summary


def write_outputs(groups, assignments, output_dir, summary, seed, ood_families):
    output_dir.mkdir()
    handles = {
        name.removesuffix(".jsonl"): (output_dir / name).open("x", encoding="utf-8")
        for name in OUTPUT_NAMES
    }
    try:
        for group in groups:
            split = assignments[group["group"]]
            handle = handles[split]
            for record in group["records"]:
                handle.write(json.dumps(record["row"], ensure_ascii=False) + "\n")
        manifest_path = output_dir / "split_manifest.jsonl"
        with manifest_path.open("x", encoding="utf-8") as handle:
            for group in groups:
                handle.write(json.dumps({
                    "scenario_seed": group["group"],
                    "split": assignments[group["group"]],
                    "topic_family": group["family"],
                    "topic": group["topic"],
                    "source_lines": [record["line"] for record in group["records"]],
                    "labels": [record["label"] for record in group["records"]],
                }, ensure_ascii=False) + "\n")
        (output_dir / "summary.json").write_text(json.dumps({
            "seed": seed,
            "ood_families": sorted(ood_families),
            "splits": summary,
        }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    finally:
        for handle in handles.values():
            handle.close()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ood-families", required=True,
                        help="Comma-separated topic families held out entirely for OOD testing")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--validation-ratio", type=float, default=0.1)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    try:
        if not args.input.is_file() or not args.manifest.is_file():
            raise ValueError("input and manifest must be existing files")
        if args.output_dir.exists():
            raise ValueError("output directory already exists; choose a new path")
        ood_families = parse_names(args.ood_families)
        records = read_records(args.input, args.manifest)
        groups = group_records(records)
        known_families = {group["family"] for group in groups}
        unknown = ood_families - known_families
        if unknown:
            raise ValueError(f"OOD family not present in input: {sorted(unknown)}")
        assignments = assign_splits(
            groups, args.seed, args.train_ratio, args.validation_ratio, ood_families
        )
        summary = make_summary(groups, assignments)
        write_outputs(groups, assignments, args.output_dir, summary, args.seed, ood_families)
    except (OSError, ValueError) as exc:
        print(f"Split failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
