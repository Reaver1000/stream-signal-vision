"""Build a human-checkable timeline from draft VLM annotations.

Reads a draft annotations JSONL plus the frames.jsonl manifest (filename ->
video_timestamp) and prints per-frame trade_state plus the VLM's event hint.
Use consecutive state rows to review possible transitions; a single-frame
"entry" hint is not ground truth.
"""
import argparse
import json
from pathlib import Path


def stamp_map(manifest):
    m = {}
    with manifest.open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            name = Path(r["image_path"]).name
            m[name] = r.get("video_timestamp")
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("annotations", type=Path)
    ap.add_argument("--manifest", type=Path, default=None)
    a = ap.parse_args()

    stamps = {}
    for cand in [a.manifest, a.annotations.parent / "frames" / "frames.jsonl"]:
        if cand and cand.exists():
            stamps = stamp_map(cand)
            break

    rows = []
    with a.annotations.open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            ev = r.get("event") or {}
            targets = ev.get("target_prices") or []
            rows.append({
                "name": r["image_path"],
                "ts": stamps.get(Path(r["image_path"]).name, f"idx={r.get('frame_index')}"),
                "state": ev.get("trade_state") or "unknown",
                "action": ev.get("action") or "none",
                "dir": ev.get("direction") or "-",
                "ot": ev.get("order_type") or "-",
                "inst": ev.get("instrument") or "-",
                "entry": ev.get("entry_price") or "-",
                "sl": ev.get("stop_price") or "-",
                "tps": ",".join(map(str, targets)) or "-",
                "qty": ev.get("displayed_quantity") or "-",
                "els": len(r.get("elements") or []),
                "review": ",".join(r.get("needs_review") or []),
            })
    rows.sort(key=lambda x: (x["ts"], x["name"]))
    print(f"{'time':>7}  {'state':8} {'hint':8} {'dir':5} {'type':6} {'inst':8} {'entry':10} {'sl':10} {'tps':14} {'qty':4} els review")
    for r in rows:
        print(f"{r['ts']:>7}  {r['state']:8} {r['action']:8} {r['dir']:5} {r['ot']:6} {r['inst']:8} {str(r['entry']):10} {str(r['sl']):10} {str(r['tps']):14} {r['qty']:4} {r['els']:>3} {r['review']}")


if __name__ == "__main__":
    main()
