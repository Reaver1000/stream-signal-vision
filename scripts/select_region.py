"""Drag a box over anything visible on screen.

Used by ``watch_screen.py`` on every launch so the scan area can be moved when a
streamer changes layout. Returns screen-relative coordinates.
"""
from __future__ import annotations

from pathlib import Path

import mss
from PIL import Image, ImageTk

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT = REPO_ROOT / "config" / "screen_roi.json"


def choose_region(monitor_index: int = 1, title: str = "Drag around the POSITION value - Enter to start, Esc to reuse last box"):
    """Show a screenshot and return the chosen box, or None if cancelled."""
    import tkinter as tk

    with mss.mss() as sct:
        monitor = sct.monitors[monitor_index]
        shot = sct.grab(monitor)
        image = Image.frombytes("RGB", shot.size, shot.bgra, "raw", "BGRX")

    root = tk.Tk()
    root.title(title)
    root.attributes("-topmost", True)
    photo = ImageTk.PhotoImage(image)
    canvas = tk.Canvas(root, width=image.width, height=image.height, highlightthickness=0)
    canvas.pack()
    canvas.create_image(0, 0, anchor=tk.NW, image=photo)

    state = {"x0": 0, "y0": 0, "rect": None, "result": None}

    def on_press(event):
        state["x0"], state["y0"] = event.x, event.y
        if state["rect"] is not None:
            canvas.delete(state["rect"])
        state["rect"] = canvas.create_rectangle(
            event.x, event.y, event.x, event.y, outline="#00ff00", width=3
        )

    def on_drag(event):
        if state["rect"] is not None:
            canvas.coords(state["rect"], state["x0"], state["y0"], event.x, event.y)

    def on_save(_event=None):
        coords = canvas.coords(state["rect"]) if state["rect"] else None
        if not coords:
            return
        x0, y0, x1, y1 = coords
        left, top = int(min(x0, x1)), int(min(y0, y1))
        width, height = int(abs(x1 - x0)), int(abs(y1 - y0))
        if width < 20 or height < 12:
            print("[select] box too small, drag again")
            return
        state["result"] = {
            "monitor": monitor_index,
            "left": left + monitor["left"],
            "top": top + monitor["top"],
            "width": width,
            "height": height,
            "screen_width": monitor["width"],
            "screen_height": monitor["height"],
        }
        root.destroy()

    canvas.bind("<ButtonPress-1>", on_press)
    canvas.bind("<B1-Motion>", on_drag)
    root.bind("<Return>", on_save)
    root.bind("<Escape>", lambda _e: root.destroy())
    root.mainloop()
    return state["result"]


def main() -> int:
    import json

    region = choose_region()
    if region is None:
        print("cancelled")
        return 1
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(region, indent=2) + "\n", encoding="utf-8")
    print(f"saved {OUT}: {region}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
