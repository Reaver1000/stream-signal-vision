"""Per-streamer platform anchor: template-match a fixed screen region to tag frames.

Some streamers execute only on one platform (e.g., channel-b trades exclusively on
TopstepX, whose view shows a large TopstepX icon in the top-left), while
others execute inside TradingView. The anchor is a per-streamer profile rule,
never a global model behavior: it tags frames/events with the visible
execution platform so live events carry context. The pooled classifier is
never specialized by this.

Usage:
    # Build a template from frames the VLM labeled with a platform, then validate it:
    python scripts/platform_anchor.py build-validate <session_dir> `
        --platform topstepx --region 0.0,0.0,0.22,0.16 `
        --template-out <private_data_root>/platform_templates/topstepx.png

    # Cross-day validation of an existing template against another session:
    python scripts/platform_anchor.py validate <session_dir> `
        --template <private_data_root>/platform_templates/topstepx.png `
        --region 0.0,0.0,0.22,0.16
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

KNOWN_PLATFORMS = ("topstepx", "tradingview", "tradovate", "wealthcharts", "mt5")


def parse_region(value: str) -> tuple[float, float, float, float]:
    parts = [float(v) for v in value.split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("region must be x,y,width,height (normalized)")
    return parts


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def region_crop(image: Any, region: tuple[float, float, float, float]) -> Any:
    import cv2

    height, width = image.shape[:2]
    x, y, box_w, box_h = region
    crop = image[int(y * height):int((y + box_h) * height),
                 int(x * width):int((x + box_w) * width)]
    if crop.ndim == 3:
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    return crop


def match_score(crop: Any, template: Any) -> float:
    import cv2

    if crop.shape[0] < template.shape[0] or crop.shape[1] < template.shape[1]:
        return -1.0
    return float(cv2.matchTemplate(crop, template, cv2.TM_CCOEFF_NORMED).max())


def session_platform_rows(session_dir: Path, platform: str) -> tuple[list[str], list[str]]:
    rows = read_jsonl(session_dir / "cue-visuals.jsonl")
    positives, negatives = [], []
    for row in rows:
        name = Path(str(row.get("image_path", ""))).name
        if not name:
            continue
        plat = str(row.get("platform_id", "")).lower()
        if plat == platform:
            positives.append(name)
        elif plat in KNOWN_PLATFORMS:
            negatives.append(name)
    return positives, negatives


def build_and_validate(args: argparse.Namespace) -> int:
    import cv2

    frames_dir = args.session_dir / "frames"
    positives, negatives = session_platform_rows(args.session_dir, args.platform)
    if len(positives) < 3:
        print(f"only {len(positives)} '{args.platform}'-labeled frames; need >= 3 to build a robust template")
        return 1

    pos_images = []
    for name in positives[:12]:
        image = cv2.imread(str(frames_dir / name))
        if image is not None:
            pos_images.append((name, region_crop(image, args.region)))
    if not pos_images:
        print("no readable positive frames")
        return 1

    medoid_name, medoid_crop = pos_images[0]
    if len(pos_images) > 1:
        best_sum, best = -1.0, pos_images[0]
        for name_a, crop_a in pos_images:
            total = sum(match_score(crop_b, crop_a) for _, crop_b in pos_images)
            if total > best_sum:
                best_sum, best = total, (name_a, crop_a)
        medoid_name, medoid_crop = best

    args.template_out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.template_out), medoid_crop)
    print(f"template <- {medoid_name} region crop -> {args.template_out} "
          f"(shape {medoid_crop.shape[1]}x{medoid_crop.shape[0]})")

    return report_scores(frames_dir, args, medoid_crop, positives, negatives)


def validate(args: argparse.Namespace) -> int:
    import cv2

    template = cv2.imread(str(args.template), cv2.IMREAD_GRAYSCALE)
    if template is None:
        print(f"could not read template: {args.template}")
        return 1
    positives, negatives = session_platform_rows(args.session_dir, args.platform)
    return report_scores(args.session_dir / "frames", args, template, positives, negatives)


def report_scores(frames_dir: Path, args: argparse.Namespace, template: Any, positives: list[str], negatives: list[str]) -> int:
    import cv2

    def scores(names: list[str]) -> list[tuple[str, float]]:
        out = []
        for name in names:
            image = cv2.imread(str(frames_dir / name))
            if image is None:
                continue
            out.append((name, match_score(region_crop(image, args.region), template)))
        return out

    pos_scores = scores(positives[:40])
    neg_scores = scores(negatives[:60])
    if not pos_scores:
        print("no readable positive frames to validate")
        return 1

    pos_values = [s for _, s in pos_scores]
    print(f"positives ({args.platform}): n={len(pos_values)} mean={sum(pos_values)/len(pos_values):.3f} "
          f"min={min(pos_values):.3f} max={max(pos_values):.3f}")
    for name, score in sorted(pos_scores, key=lambda item: item[1])[:5]:
        print(f"  lowest positive: {score:.3f} {name}")
    if neg_scores:
        neg_values = [s for _, s in neg_scores]
        print(f"negatives (other platforms): n={len(neg_values)} mean={sum(neg_values)/len(neg_values):.3f} "
              f"min={min(neg_values):.3f} max={max(neg_values):.3f}")
        for name, score in sorted(neg_scores, key=lambda item: -item[1])[:5]:
            print(f"  highest negative: {score:.3f} {name}")
        if min(pos_values) > max(neg_values):
            suggested = (min(pos_values) + max(neg_values)) / 2
            print(f"clean separation: suggested --anchor-threshold {suggested:.2f}")
            return 0
        suggested = (min(pos_values) + max(neg_values)) / 2
        print(f"overlap between classes; threshold {suggested:.2f} will misclassify some frames")
        return 2
    print("no cross-platform negatives labeled in this session")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("build-validate", "validate"))
    parser.add_argument("session_dir", type=Path)
    parser.add_argument("--platform", default="topstepx", choices=KNOWN_PLATFORMS)
    parser.add_argument("--region", type=parse_region, default=(0.0, 0.0, 0.22, 0.16))
    parser.add_argument("--template", type=Path, default=None, help="existing template PNG (validate action)")
    parser.add_argument("--template-out", type=Path, default=None)
    args = parser.parse_args()

    if args.action == "build-validate":
        if args.template_out is None:
            parser.error("--template-out is required for build-validate")
        return build_and_validate(args)
    if args.template is None:
        parser.error("--template is required for validate")
    return validate(args)


if __name__ == "__main__":
    raise SystemExit(main())
