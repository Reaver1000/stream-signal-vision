"""Validate an annotated JSONL manifest and export a session-safe YOLO dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stream_signal_vision.dataset import CLASS_IDS, DatasetError, split_by_session, validate_record, yolo_label_lines


def load_manifest(path: Path, dataset_root: Path) -> list[dict]:
    records = []
    errors = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            errors.append(f"line {line_no}: invalid JSON: {exc}")
            continue
        issues = validate_record(record, dataset_root)
        errors.extend(f"line {line_no}: {issue}" for issue in issues)
        if not issues:
            records.append(record)
    if errors:
        raise DatasetError("\n".join(errors))
    if not records:
        raise DatasetError("manifest contains no valid records")
    return records


def safe_stem(record: dict) -> str:
    session = "".join(c if c.isalnum() or c in "-_" else "_" for c in record["session_id"])
    image = Path(record["image_path"]).stem
    return f"{session}__{image}"


def export_dataset(manifest: Path, dataset_root: Path, output: Path, seed: int) -> None:
    records = load_manifest(manifest, dataset_root)
    splits = split_by_session(records, seed=seed)
    for split, split_records in splits.items():
        image_dir = output / split / "images"
        label_dir = output / split / "labels"
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        for record in split_records:
            source = (dataset_root / record["image_path"]).resolve()
            stem = safe_stem(record)
            shutil.copy2(source, image_dir / (stem + source.suffix.lower()))
            (label_dir / (stem + ".txt")).write_text(
                "\n".join(yolo_label_lines(record)) + "\n",
                encoding="utf-8",
            )

    names = "\n".join(f"  {idx}: {name}" for name, idx in CLASS_IDS.items())
    yaml = (
        f"path: {output.resolve().as_posix()}\n"
        "train: train/images\n"
        "val: val/images\n"
        "test: test/images\n"
        f"names:\n{names}\n"
    )
    output.mkdir(parents=True, exist_ok=True)
    (output / "dataset.yaml").write_text(yaml, encoding="utf-8")
    print("Exported session-separated dataset:")
    for split, split_records in splits.items():
        sessions = sorted({r["session_id"] for r in split_records})
        print(f"  {split}: {len(split_records)} frames from {len(sessions)} session(s)")
    print(f"  yaml: {output / 'dataset.yaml'}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path, help="JSONL annotations manifest")
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="Root used to resolve image_path")
    parser.add_argument("--output", type=Path, default=Path("data/derived/yolo"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    try:
        export_dataset(args.manifest, args.root, args.output, args.seed)
    except DatasetError as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
