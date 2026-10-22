"""Run the visual-only state classifier on selected frames."""
import argparse
from pathlib import Path

import cv2
from ultralytics import YOLO


def parse_roi(value):
    values = tuple(float(v) for v in value.split(","))
    if len(values) != 4:
        raise argparse.ArgumentTypeError("ROI must be x,y,width,height")
    return values


def crop_roi(image, roi):
    if roi is None:
        return image
    height, width = image.shape[:2]
    x, y, box_w, box_h = roi
    return image[int(y * height):int((y + box_h) * height),
                 int(x * width):int((x + box_w) * width)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("frames", type=Path)
    parser.add_argument("--names", required=True, help="comma-separated frame filenames")
    parser.add_argument("--roi", type=parse_roi, default=None,
                        help="normalized x,y,width,height used during training")
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--device", default="0")
    args = parser.parse_args()

    paths = [args.frames / name for name in args.names.split(",") if name]
    images = []
    for path in paths:
        image = cv2.imread(str(path))
        if image is None:
            raise SystemExit(f"could not read image: {path}")
        images.append(crop_roi(image, args.roi))

    model = YOLO(str(args.model))
    results = model.predict(images, imgsz=args.imgsz, device=args.device, verbose=False)
    for path, result in zip(paths, results):
        probs = result.probs
        class_name = model.names[int(probs.top1)]
        confidence = float(probs.top1conf)
        print(f"{path.name}: {class_name} ({confidence:.3f})")


if __name__ == "__main__":
    main()
