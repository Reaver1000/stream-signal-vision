"""Fine-tune the pooled UI-element detector on an exported dataset.

Ultralytics is an optional research backend; review its AGPL/commercial terms
before bundling or redistributing it in any closed-source paid product.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path, help="Path to prepared dataset.yaml")
    parser.add_argument("--model", default="yolo26n.pt", help="Pretrained detector checkpoint")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="", help="0, cpu, or empty for auto")
    parser.add_argument("--amp", action="store_true",
                        help="Enable mixed precision (off by default for Windows CUDA compatibility)")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--project", type=Path, default=Path("runs/vision"))
    parser.add_argument("--name", default="pooled-stream-ui")
    args = parser.parse_args()

    if not args.dataset.is_file():
        parser.error(f"dataset file not found: {args.dataset}")
    try:
        from ultralytics import YOLO
    except ImportError:
        parser.error("Install the optional detector backend first: pip install 'ultralytics>=8.4'")

    model = YOLO(args.model)
    model.train(
        data=str(args.dataset),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device or None,
        workers=args.workers,
        amp=args.amp,
        project=str(args.project),
        name=args.name,
        seed=42,
        deterministic=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
