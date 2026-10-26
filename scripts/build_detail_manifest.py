"""Build a pooled YOLO manifest for trade-detail elements.

Unlike the open/not_open classifier, this targets what a live reader actually
needs to see: entry marker, stop line, target line, quantity badge, position
panel and order ticket.

Scans every private session's cue-visuals.jsonl, keeps valid boxes for those
classes, and adds a small share of empty frames as background negatives.
Images are referenced relative to the private data root so prepare_yolo.py can
resolve and copy them.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

VALID_KINDS = (
    "entry_marker",
    "stop_line",
    "target_line",
    "quantity_badge",
    "position_panel",
    "order_ticket",
)


def clean_elements(elements) -> list[dict]:
    clean = []
    for element in elements or []:
        kind = element.get("kind")
        if kind not in VALID_KINDS:
            continue
        bbox = element.get("bbox")
        if not (isinstance(bbox, list) and len(bbox) == 4):
            continue
        try:
            x, y, w, h = (float(v) for v in bbox)
        except (TypeError, ValueError):
            continue
        if not (0 <= x < 1 and 0 <= y < 1 and 0 < w <= 1 and 0 < h <= 1):
            continue
        if x + w > 1.0001 or y + h > 1.0001:
            w = min(w, 1.0 - x)
            h = min(h, 1.0 - y)
            if w <= 0 or h <= 0:
                continue
        clean.append({"kind": kind, "bbox": [round(x, 5), round(y, 5), round(w, 5), round(h, 5)]})
    return clean


def locate_image(root: Path, session_dir: Path, name: str) -> Path | None:
    for candidate in (session_dir / "frames" / name, session_dir / name):
        if candidate.is_file():
            return candidate
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", type=Path, required=True, help="private raw data root")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--negative-frac", type=float, default=0.12,
                    help="share of empty-element frames kept as background negatives")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    root = args.data_root.resolve()
    positives: list[dict] = []
    negatives: list[dict] = []
    counts = {kind: 0 for kind in VALID_KINDS}
    sessions_with_boxes: set[str] = set()

    for annotations in sorted(root.glob("*/cue-visuals.jsonl")):
        session_dir = annotations.parent
        session_id = session_dir.name
        for line in annotations.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            name = Path(str(record.get("image_path", ""))).name
            if not name:
                continue
            image = locate_image(root, session_dir, name)
            if image is None:
                continue
            rel = image.relative_to(root).as_posix()
            elements = clean_elements(record.get("elements"))
            base = {
                "image_path": rel,
                "platform_id": record.get("platform_id") or "other",
                "session_id": session_id,
            }
            if elements:
                base["elements"] = elements
                positives.append(base)
                sessions_with_boxes.add(session_id)
                for element in elements:
                    counts[element["kind"]] += 1
            else:
                base["elements"] = []
                negatives.append(base)

    rng = random.Random(args.seed)
    rng.shuffle(negatives)
    keep_negatives = negatives[: int(len(negatives) * args.negative_frac)]
    records = positives + keep_negatives
    rng.shuffle(records)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        "\n".join(json.dumps(record, ensure_ascii=False) for record in records) + "\n",
        encoding="utf-8",
    )

    print(f"positives={len(positives)} background={len(keep_negatives)} total={len(records)}")
    print(f"sessions_with_boxes={len(sessions_with_boxes)}")
    for kind in VALID_KINDS:
        print(f"  {kind:16s} {counts[kind]}")
    print(f"manifest -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
