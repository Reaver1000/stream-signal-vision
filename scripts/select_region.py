"""Drag a box over anything on screen and save it as the position scan area.

Run it, drag a rectangle around the POSITION value (e.g. "5@29539.25"), press
Enter. Esc cancels. Saves config/screen_roi.json relative to the chosen monitor.
"""
from __future__ import annotations

import json
import tkinter as tk
from pathlib import Path

import mss
from PIL import Image, ImageTk

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT = REPO_ROOT / "config" / "screen_roi.json"


def main() -> int:
    with mss.mss() as sct:
        monitor = sct.monitors[1]
        shot = sct.grab(monitor)
        image = Image.frombytes("RGB", shot.size, shot.bgra, "raw", "BGRX")

    root = tk.Tk()
    root.title("Select the POSITION area - drag, then Enter")
    root.attributes("-topmost", True)
    photo = ImageTk.PhotoImage(image)
    canvas = tk.Canvas(root, width=image.width, height=image.height, highlightthickness=0)
    canvas.pack()
    canvas.create_image(0, 0, anchor=tk.NW, image=photo)

    state = {"x0": 0, "y0": 0, "rect": None}

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
        x0, y0 = state["x0"], state["y0"]
        coords = canvas.coords(state["rect"]) if state["rect"] else None
        if not coords:
            return
        x1, y1 = coords[2], coords[3]
        left, top = int(min(x0, x1)), int(min(y0, y1))
        width, height = int(abs(x1 - x0)), int(abs(y1 - y0))
        if width < 20 or height < 12:
            print("selection too small")
            return
        data = {
            "monitor": 1,
            "left": left + monitor["left"],
            "top": top + monitor["top"],
            "width": width,
            "height": height,
            "screen_width": monitor["width"],
            "screen_height": monitor["height"],
        }
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        print(f"saved {OUT}: {data}")
        root.destroy()

    canvas.bind("<ButtonPress-1>", on_press)
    canvas.bind("<B1-Motion>", on_drag)
    root.bind("<Return>", on_save)
    root.bind("<Escape>", lambda _e: root.destroy())
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
