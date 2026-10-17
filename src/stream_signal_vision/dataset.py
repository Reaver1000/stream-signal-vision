"""Dataset manifest validation and session-safe YOLO label conversion helpers."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any


YOLO_CLASSES = (
    "entry_marker",
    "stop_line",
    "target_line",
    "quantity_badge",
    "position_panel",
)
CLASS_IDS = {name: index for index, name in enumerate(YOLO_CLASSES)}


class DatasetError(ValueError):
    """Raised when a capture annotation cannot safely enter a training set."""


def validate_record(record: dict[str, Any], dataset_root: Path) -> list[str]:
    errors: list[str] = []
    for key in ("image_path", "platform_id", "session_id"):
        if not record.get(key):
            errors.append(f"missing required field: {key}")
    image_path = record.get("image_path")
    if image_path:
        resolved = (dataset_root / image_path).resolve()
        if not resolved.is_relative_to(dataset_root.resolve()):
            errors.append("image_path must stay inside dataset_root")
        elif not resolved.is_file():
            errors.append(f"image not found: {image_path}")
    elements = record.get("elements", [])
    if not isinstance(elements, list):
        errors.append("elements must be a list")
        return errors
    for index, element in enumerate(elements):
        kind = element.get("kind")
        if kind not in CLASS_IDS:
            errors.append(f"elements[{index}].kind is not a YOLO class: {kind}")
            continue
        bbox = element.get("bbox")
        if not isinstance(bbox, list) or len(bbox) != 4:
            errors.append(f"elements[{index}].bbox must be [x, y, width, height]")
            continue
        try:
            x, y, width, height = (float(value) for value in bbox)
        except (TypeError, ValueError):
            errors.append(f"elements[{index}].bbox contains a non-number")
            continue
        if not (0 <= x < 1 and 0 <= y < 1 and 0 < width <= 1 and 0 < height <= 1):
            errors.append(f"elements[{index}].bbox values must be normalized to (0, 1]")
        elif x + width > 1 or y + height > 1:
            errors.append(f"elements[{index}].bbox extends outside the frame")
    return errors


def split_by_session(
    records: list[dict[str, Any]],
    *,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> dict[str, list[dict[str, Any]]]:
    """Split by whole session/VOD so adjacent frames never leak across splits."""
    ratios = (train_ratio, val_ratio, test_ratio)
    if any(r <= 0 for r in ratios) or abs(sum(ratios) - 1.0) > 1e-6:
        raise DatasetError("split ratios must be positive and sum to 1")
    sessions = sorted({str(r["session_id"]) for r in records if r.get("session_id")})
    if len(sessions) < 3:
        raise DatasetError("at least three distinct sessions are required for train/val/test")

    random.Random(seed).shuffle(sessions)
    n = len(sessions)
    n_val = max(1, round(n * val_ratio))
    n_test = max(1, round(n * test_ratio))
    if n_val + n_test >= n:
        n_val = 1
        n_test = 1
    n_train = n - n_val - n_test
    assignments: dict[str, str] = {}
    for session in sessions[:n_train]:
        assignments[session] = "train"
    for session in sessions[n_train:n_train + n_val]:
        assignments[session] = "val"
    for session in sessions[n_train + n_val:]:
        assignments[session] = "test"

    result = {"train": [], "val": [], "test": []}
    for record in records:
        session = str(record.get("session_id", ""))
        if session in assignments:
            result[assignments[session]].append(record)
    return result


def yolo_label_lines(record: dict[str, Any]) -> list[str]:
    """Convert normalized xywh annotations to YOLO class/xcenter/ycenter/w/h."""
    lines: list[str] = []
    for element in record.get("elements", []):
        class_id = CLASS_IDS[element["kind"]]
        x, y, width, height = (float(value) for value in element["bbox"])
        cx = x + width / 2
        cy = y + height / 2
        lines.append(f"{class_id} {cx:.6f} {cy:.6f} {width:.6f} {height:.6f}")
    return lines
