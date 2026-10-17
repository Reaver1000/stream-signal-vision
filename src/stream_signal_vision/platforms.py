"""Platform/layout profiles. Profiles describe UI families, not fixed pixels.

Every streamer can crop, resize, theme, or rearrange a platform. Coordinates
must be calibrated per screen/layout; the example profiles intentionally have
no regions until calibrated.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any

from .models import Box


@dataclass(frozen=True, slots=True)
class PlatformProfile:
    platform_id: str
    display_name: str
    layout_id: str = "default"
    reference_width: int = 1920
    reference_height: int = 1080
    regions: dict[str, Box] = field(default_factory=dict)
    element_hints: tuple[str, ...] = ()

    def validate(self) -> list[str]:
        errors: list[str] = []
        if self.reference_width <= 0 or self.reference_height <= 0:
            errors.append("reference dimensions must be positive")
        for name, box in self.regions.items():
            if not box.is_valid():
                errors.append(f"region '{name}' is outside normalized frame bounds")
        return errors


PLATFORM_HINTS: dict[str, tuple[str, ...]] = {
    "topstepx": ("chart", "position_panel", "order_lines", "quantity"),
    "tradingview": ("chart", "broker_panel", "position_panel", "order_lines"),
    "tradovate": ("chart", "dom_or_ladder", "position_panel", "working_orders"),
    "wealthcharts": ("chart", "position_panel", "order_lines", "quantity"),
    "mt5": ("chart", "terminal_trade_tab", "sl_tp_markers", "volume"),
}


def load_profiles(path: str | Path) -> dict[str, PlatformProfile]:
    data: dict[str, Any] = json.loads(Path(path).read_text(encoding="utf-8"))
    result: dict[str, PlatformProfile] = {}
    for platform_id, raw in data.get("platforms", {}).items():
        regions = {
            name: Box(**coords)
            for name, coords in raw.get("regions", {}).items()
            if coords is not None
        }
        profile = PlatformProfile(
            platform_id=platform_id,
            display_name=raw.get("display_name", platform_id),
            layout_id=raw.get("layout_id", "default"),
            reference_width=raw.get("reference_width", 1920),
            reference_height=raw.get("reference_height", 1080),
            regions=regions,
            element_hints=tuple(raw.get("element_hints", PLATFORM_HINTS.get(platform_id, ()))),
        )
        errors = profile.validate()
        if errors:
            raise ValueError(f"invalid profile '{platform_id}': {'; '.join(errors)}")
        result[platform_id] = profile
    return result
