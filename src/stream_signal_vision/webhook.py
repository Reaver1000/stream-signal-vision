"""Optional adapter to the local NT8 node's authenticated webhook intake."""

from __future__ import annotations

import json
from urllib.request import Request, urlopen

from .models import Action, Direction, SignalCandidate


def format_node_message(candidate: SignalCandidate) -> str:
    """Format a validated candidate in the node parser's canonical text dialect."""
    if candidate.action == Action.ENTRY:
        if candidate.instrument is None or candidate.direction is None:
            raise ValueError("entry candidate lacks instrument/direction")
        if candidate.entry_price is None or candidate.stop_price is None:
            raise ValueError("entry candidate lacks entry/stop")
        side = "Longs" if candidate.direction == Direction.LONG else "Shorts"
        lines = [f"{candidate.instrument} {side} @ {candidate.entry_price}"]
        if candidate.target_prices:
            lines.append("PT: " + ", ".join(str(p) for p in candidate.target_prices))
        lines.append(f"S/L: {candidate.stop_price}")
        return "\n".join(lines)
    if candidate.action == Action.MOVE_STOP:
        return "move stop to breakeven"
    if candidate.action == Action.FLATTEN:
        return "closed all"
    if candidate.action == Action.SCALE_OUT:
        fraction = candidate.scale_fraction or 0
        if fraction <= 0.35:
            return "scale out light"
        if fraction <= 0.78:
            return "scale out most"
        return "scale out aggressively"
    raise ValueError("no-action candidates are not sent")


class LocalNodeWebhookSink:
    """POSTs candidates to the member's localhost NT8 webhook listener.

    The node key is supplied by the user at runtime; it must never be checked
    into a repository or included in captured frames/logs.
    """

    def __init__(self, url: str, node_key: str, timeout_seconds: float = 3.0):
        if not url.startswith("http://127.0.0.1:") and not url.startswith("http://localhost:"):
            raise ValueError("sink must target a local loopback listener")
        if not node_key.strip():
            raise ValueError("node_key is required")
        self.url = url
        self.node_key = node_key
        self.timeout_seconds = timeout_seconds

    def build_payload(self, candidate: SignalCandidate) -> bytes:
        payload = {
            "key": self.node_key,
            "content": format_node_message(candidate),
            "source": "stream-signal-vision",
            "event_id": candidate.event_id,
        }
        return json.dumps(payload, separators=(",", ":")).encode("utf-8")

    def publish(self, candidate: SignalCandidate) -> None:
        request = Request(
            self.url,
            data=self.build_payload(candidate),
            headers={"Content-Type": "application/json", "X-Node-Key": self.node_key},
            method="POST",
        )
        with urlopen(request, timeout=self.timeout_seconds) as response:
            if response.status < 200 or response.status >= 300:
                raise RuntimeError(f"local node returned HTTP {response.status}")
