"""Train an exploratory visual-only open/not-open state classifier."""
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--model", default="yolo26n-cls.pt")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="0")
    parser.add_argument("--amp", action="store_true",
                        help="Enable mixed precision (off by default for Windows CUDA compatibility)")
    parser.add_argument("--project", type=Path, default=Path("runs/vision"))
    parser.add_argument("--name", default="trade-state-channel-a-prototype")
    args = parser.parse_args()
    if not (args.dataset / "train").is_dir() or not (args.dataset / "val").is_dir():
        parser.error("dataset must contain train/ and val/ classification folders")
    try:
        from ultralytics import YOLO
    except ImportError:
        parser.error("install the visual backend with: pip install -e '.[train-yolo]'")

    model = YOLO(args.model)
    try:
        model.train(
            data=str(args.dataset),
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            workers=0,
            amp=args.amp,
            val=False,
            project=str(args.project),
            name=args.name,
            seed=42,
            deterministic=True,
        )
    except NotImplementedError as exc:
        if "torchvision::nms" not in str(exc):
            raise
        print("CUDA torchvision NMS is unavailable; training checkpoint was saved. "
              "Running validation separately on CPU.")
    save_dir = Path(model.trainer.save_dir)
    best = save_dir / "weights" / "best.pt"
    if not best.is_file():
        best = save_dir / "weights" / "last.pt"
    if best.is_file():
        trained = YOLO(str(best))
        val_dir = args.dataset / "val"
        if val_dir.is_dir() and any(val_dir.rglob("*.jpg")):
            metrics = trained.val(data=str(args.dataset), split="val", imgsz=args.imgsz,
                                  batch=args.batch, device="cpu", workers=0, plots=False)
            print(f"temporal holdout metrics: {metrics.results_dict}")
        external_not_open = list((args.dataset / "external" / "not_open").glob("*.jpg"))
        if external_not_open:
            predictions = trained.predict(source=[str(p) for p in external_not_open],
                                          imgsz=args.imgsz, device="cpu", verbose=False)
            open_id = next((idx for idx, name in trained.names.items() if name == "open"), None)
            false_open = sum(result.probs.top1 == open_id for result in predictions)
            print(f"external negative check: {false_open}/{len(external_not_open)} predicted open")
    print(f"best checkpoint: {best}")


if __name__ == "__main__":
    main()
