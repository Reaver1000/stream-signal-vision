"""Render draft normalized bounding boxes for human annotation review.

Usage:
    python scripts/render_annotations.py annotations.jsonl frames_dir preview_dir \
        --names frame_00119.png,frame_01902.png
"""
import argparse
import json
from pathlib import Path

import cv2


COLORS = {
    "entry_marker": (255, 180, 0),
    "stop_line": (0, 0, 255),
    "target_line": (255, 0, 0),
    "quantity_badge": (0, 200, 255),
    "position_panel": (0, 255, 0),
    "order_ticket": (220, 0, 220),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("annotations", type=Path)
    parser.add_argument("frames", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--names", default=None, help="comma-separated image filenames")
    parser.add_argument("--limit", type=int, default=100)
    args = parser.parse_args()

    names = set(args.names.split(",")) if args.names else None
    args.output.mkdir(parents=True, exist_ok=True)
    rendered = 0
    with args.annotations.open(encoding="utf-8") as source:
        for line in source:
            record = json.loads(line)
            image_name = Path(record["image_path"]).name
            if names is not None and image_name not in names:
                continue
            image = cv2.imread(str(args.frames / image_name))
            if image is None:
                continue
            height, width = image.shape[:2]
            event = record.get("event") or {}
            for element in record.get("elements", []):
                x, y, box_w, box_h = [float(v) for v in element["bbox"]]
                x1, y1 = int(x * width), int(y * height)
                x2, y2 = int((x + box_w) * width), int((y + box_h) * height)
                kind = element.get("kind", "unknown")
                color = COLORS.get(kind, (255, 255, 255))
                cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
                label = kind + (f": {element['text']}" if element.get("text") else "")
                cv2.putText(image, label[:60], (x1, max(18, y1 - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
            status = f"state={event.get('trade_state', '?')} hint={event.get('action', '?')}"
            cv2.putText(image, status, (12, 24), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.imwrite(str(args.output / image_name), image)
            rendered += 1
            if names is None and rendered >= args.limit:
                break
    print(f"rendered {rendered} annotated frames to {args.output}")


if __name__ == "__main__":
    main()
