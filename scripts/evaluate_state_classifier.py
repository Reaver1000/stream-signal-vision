"""Score a visual state classifier and list frame-level errors for review."""
import argparse
import json
from pathlib import Path

from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--split", default="val", choices=("train", "val", "external"))
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-errors", type=int, default=30)
    parser.add_argument("--errors-out", type=Path, default=None)
    args = parser.parse_args()

    split_dir = args.dataset / args.split
    files = []
    for true_label in ("open", "not_open"):
        files.extend((p, true_label) for p in sorted((split_dir / true_label).glob("*.jpg")))
    if not files:
        raise SystemExit(f"no images found in {split_dir}")

    model = YOLO(str(args.model))
    manifest_path = args.dataset / "manifest.jsonl"
    metadata = {}
    if manifest_path.exists():
        with manifest_path.open(encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                metadata[row["output_path"]] = row

    results = model.predict(source=[str(path) for path, _ in files],
                            imgsz=args.imgsz, device=args.device, verbose=False)
    matrix = {actual: {predicted: 0 for predicted in ("open", "not_open")}
              for actual in ("open", "not_open")}
    errors = []
    for (path, actual), result in zip(files, results):
        predicted = model.names[int(result.probs.top1)]
        confidence = float(result.probs.top1conf)
        matrix[actual][predicted] += 1
        if actual != predicted:
            details = metadata.get(path.relative_to(args.dataset).as_posix(), {})
            errors.append({"true": actual, "predicted": predicted,
                           "confidence": round(confidence, 4),
                           "path": path.relative_to(args.dataset).as_posix(),
                           "source_id": details.get("source_id"),
                           "video_timestamp": details.get("video_timestamp"),
                           "interval_reason": details.get("interval_reason")})

    total = len(files)
    correct = sum(matrix[label][label] for label in matrix)
    negative_total = sum(matrix["not_open"].values())
    false_open = matrix["not_open"]["open"]
    print(f"split={args.split} samples={total} accuracy={correct / total:.4f}")
    print(f"confusion={matrix}")
    print(f"false_open={false_open}/{negative_total} rate={false_open / max(1, negative_total):.4f}")
    for row in errors[:args.max_errors]:
        print(f"ERROR {row['video_timestamp']} true={row['true']} predicted={row['predicted']} "
              f"conf={row['confidence']:.3f} image={row['path']}")
    if args.errors_out:
        args.errors_out.parent.mkdir(parents=True, exist_ok=True)
        args.errors_out.write_text("\n".join(json.dumps(row) for row in errors) + "\n",
                                  encoding="utf-8")


if __name__ == "__main__":
    main()
