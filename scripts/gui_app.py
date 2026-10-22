#!/usr/bin/env python3
"""Tkinter control panel for stream-signal-vision.

Paste a live-stream URL, a past-stream (VOD) URL, or a batch of YouTube URLs
and press Start. The panel runs the existing tested CLI scripts
(``scripts/watch_live.py``, ``scripts/train_channel.py``) as subprocesses, so
no behaviour is duplicated and logs stream into the window.

Usage:

    python scripts/gui_app.py

Stdlib only (Tkinter). Field values persist to gitignored
``gui_settings.local.json`` next to the repository root.
"""
from __future__ import annotations

import datetime
import json
import os
import queue
import re
import subprocess
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, scrolledtext, ttk

REPO_ROOT = Path(__file__).resolve().parents[1]
SETTINGS_PATH = REPO_ROOT / "gui_settings.local.json"

STREAMER_PRESETS = {
    "nso": {"platform": "topstepx", "roi": "0.0,0.08,0.78,0.82"},
    "keyflow": {"platform": "tradingview", "roi": "0.02,0.11,0.69,0.87"},
    "rips": {"platform": "tradingview", "roi": "0.02,0.11,0.69,0.87"},
}
PLATFORMS = ("tradingview", "topstepx", "tradovate", "wealthcharts", "mt5")
DEVICES = ("cpu", "0", "cuda:0")

DATA_ROOT_CANDIDATES = (
    REPO_ROOT.parent / "Default Project" / "stream-signal-vision" / "data" / "raw",
    REPO_ROOT / "data" / "raw",
)

_CHANNEL_URL_RE = re.compile(r"/@|/channel/|/c/|/streams\b|/live\b")
_WEBHOOK_RE = re.compile(r"webhooks/\S+")


def is_channel_url(url: str) -> bool:
    return bool(url.strip()) and bool(_CHANNEL_URL_RE.search(url.strip()))


def mask_secrets(line: str) -> str:
    return _WEBHOOK_RE.sub("webhooks/***", line)


def default_data_root() -> Path:
    for cand in DATA_ROOT_CANDIDATES:
        if (cand / "channel_ledger.json").is_file():
            return cand
    for cand in DATA_ROOT_CANDIDATES:
        if cand.is_dir():
            return cand
    return DATA_ROOT_CANDIDATES[-1]


def default_routes_config() -> str:
    for cand in DATA_ROOT_CANDIDATES:
        p = cand / "webhook_routes.json"
        if p.is_file():
            return str(p)
    p = REPO_ROOT / "data" / "webhook_routes.json"
    return str(p) if p.is_file() else ""


def find_models() -> list[Path]:
    models: list[Path] = []
    for pat in ("runs/classify/runs/vision/*/weights/best.pt", "hf/**/*.pt"):
        models.extend(p for p in REPO_ROOT.glob(pat) if p.is_file())
    sibling = REPO_ROOT.parent / "stream-signal-weights"
    if sibling.is_dir():
        models.extend(p for p in sibling.glob("*.pt") if p.is_file())
    seen: set[str] = set()
    out: list[Path] = []
    for m in sorted(models, key=lambda p: p.stat().st_mtime, reverse=True):
        key = str(m).lower()
        if key not in seen:
            seen.add(key)
            out.append(m)
    return out


def model_label(path: Path) -> str:
    if path.parent.name == "weights":
        return path.parent.parent.name
    return path.stem


def preferred_model(models: list[Path]) -> str:
    for m in models:
        if "v16" in model_label(m).lower():
            return model_label(m)
    return model_label(models[0]) if models else ""


def find_anchor_templates() -> list[Path]:
    root = REPO_ROOT / "data" / "platform_templates"
    if not root.is_dir():
        return []
    return sorted(p for p in root.glob("*.png") if p.is_file())


def events_out_path(source_id: str) -> Path:
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    slug = re.sub(r"[^a-z0-9_-]+", "-", source_id.strip().lower()).strip("-") or "live"
    return REPO_ROOT / "runs" / "events" / f"{slug}_{stamp}.jsonl"


def parse_url_list(text: str) -> list[str]:
    return [
        ln.strip()
        for ln in text.splitlines()
        if ln.strip() and not ln.strip().startswith("#")
    ]


def plan_batch(text: str, channel_url: str = "") -> list[tuple[str, str]]:
    plan: list[tuple[str, str]] = []
    if channel_url.strip():
        plan.append(("channel", channel_url.strip()))
    for u in parse_url_list(text):
        plan.append(("channel" if is_channel_url(u) else "vod", u))
    seen: set[tuple[str, str]] = set()
    out: list[tuple[str, str]] = []
    for item in plan:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def build_watch_command(
    *,
    url: str,
    model: str,
    source_id: str,
    roi: str = "",
    device: str = "cpu",
    fps: str = "2.0",
    threshold: str = "0.85",
    routes_config: str = "",
    anchor_template: str = "",
    anchor_threshold: str = "0.72",
    read_details: bool = False,
    verbose: bool = False,
    events_out: str | None = None,
) -> list[str]:
    argv = [
        sys.executable,
        "scripts/watch_live.py",
        "--url",
        url,
        "--model",
        str(model),
        "--source-id",
        source_id,
        "--device",
        str(device),
        "--fps",
        str(fps),
        "--threshold",
        str(threshold),
    ]
    if roi:
        argv += ["--roi", roi]
    if routes_config:
        argv += ["--routes-config", str(routes_config)]
    if anchor_template:
        argv += [
            "--anchor-template",
            str(anchor_template),
            "--anchor-threshold",
            str(anchor_threshold),
        ]
    if read_details:
        argv.append("--read-details")
    if verbose:
        argv.append("--verbose")
    if events_out:
        argv += ["--events-out", str(events_out)]
    return argv


def build_train_command(
    *,
    streamer: str,
    platform: str,
    mode: str,
    url: str = "",
    channel: str = "",
    session_dir: str = "",
    roi: str = "",
    data_root: str = "",
    intervals: str = "",
    model: str = "",
    run_name: str = "",
    max_vods: str = "",
    train_detector: bool = False,
    cleanup_media: bool = False,
    dry_run: bool = False,
) -> list[str]:
    argv = [
        sys.executable,
        "scripts/train_channel.py",
        "--streamer",
        streamer,
        "--platform",
        platform,
        "--mode",
        mode,
    ]
    if url:
        argv += ["--url", url]
    if channel:
        argv += ["--channel", channel]
    if session_dir:
        argv += ["--session-dir", session_dir]
    if roi:
        argv += ["--roi", roi]
    if data_root:
        argv += ["--data-root", data_root]
    if intervals:
        argv += ["--intervals", intervals]
    if model:
        argv += ["--model", model]
    if run_name:
        argv += ["--run-name", run_name]
    if max_vods:
        argv += ["--max-vods", max_vods]
    if train_detector:
        argv.append("--train-detector")
    if cleanup_media:
        argv.append("--cleanup-media")
    if dry_run:
        argv.append("--dry-run")
    return argv


class CommandRunner:
    """Runs one CLI subprocess, streaming masked output lines to a callback."""

    def __init__(self, log_fn) -> None:
        self._log_fn = log_fn
        self._proc: subprocess.Popen | None = None

    def is_running(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def start(self, argv: list[str], on_done=None) -> bool:
        if self.is_running():
            self._log_fn("[busy] a command is already running in this panel")
            return False
        creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        self._log_fn("$ " + " ".join(mask_secrets(a) for a in argv))
        try:
            self._proc = subprocess.Popen(
                argv,
                cwd=str(REPO_ROOT),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                creationflags=creationflags,
            )
        except OSError as exc:
            self._log_fn(f"[error] failed to start: {exc}")
            return False
        lines: list[str] = []
        proc = self._proc

        def _pump() -> None:
            assert proc.stdout is not None
            for raw in proc.stdout:
                clean = mask_secrets(raw.rstrip())
                lines.append(clean)
                self._log_fn(clean)
            rc = proc.wait()
            self._log_fn(f"[exit {rc}]")
            if on_done is not None:
                on_done(rc, lines)

        threading.Thread(target=_pump, daemon=True).start()
        return True

    def stop(self) -> None:
        if not self.is_running():
            return
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(self._proc.pid)],
                capture_output=True,
            )
        else:
            self._proc.kill()


class LogPane(ttk.Frame):
    def __init__(self, master):
        super().__init__(master)
        self._queue: queue.Queue[str] = queue.Queue()
        self.text = scrolledtext.ScrolledText(
            self, height=16, state=tk.DISABLED, font=("Consolas", 9), wrap=tk.WORD
        )
        self.text.pack(fill=tk.BOTH, expand=True)
        self.after(150, self._drain)

    def append(self, line: str) -> None:
        self._queue.put(line)

    def _drain(self) -> None:
        dirty = False
        try:
            while True:
                line = self._queue.get_nowait()
                self.text.configure(state=tk.NORMAL)
                self.text.insert(tk.END, line + "\n")
                self.text.configure(state=tk.DISABLED)
                dirty = True
        except queue.Empty:
            pass
        if dirty:
            self.text.see(tk.END)
        self.after(150, self._drain)


class _TextVar:
    def __init__(self, widget: tk.Text) -> None:
        self.widget = widget

    def get(self) -> str:
        return self.widget.get("1.0", "end-1c")

    def set(self, value: str) -> None:
        self.widget.delete("1.0", tk.END)
        self.widget.insert("1.0", value)


class BaseTab(ttk.Frame):
    def __init__(self, master, app, title: str):
        super().__init__(master, padding=10)
        self.app = app
        self.title = title
        self._row = 0
        self.log = LogPane(self)

    def _next_row(self) -> int:
        row = self._row
        self._row += 1
        return row

    def add_entry(
        self,
        label: str,
        key: str,
        default: str = "",
        width: int = 56,
        browse: str | None = None,
        browse_save: bool = False,
    ) -> tk.StringVar:
        row = self._next_row()
        ttk.Label(self, text=label).grid(row=row, column=0, sticky="w", padx=(0, 8), pady=2)
        var = tk.StringVar(value=default)
        ttk.Entry(self, textvariable=var, width=width).grid(row=row, column=1, sticky="we", pady=2)
        if browse:
            ttk.Button(self, text="Browse", width=8,
                       command=lambda: self._browse(var, browse, browse_save)).grid(
                row=row, column=2, sticky="w", padx=(6, 0)
            )
        self.app.bind_var(key, var)
        return var

    def add_combo(
        self,
        label: str,
        key: str,
        values: tuple[str, ...] | list[str],
        default: str = "",
        readonly: bool = True,
    ) -> tk.StringVar:
        row = self._next_row()
        ttk.Label(self, text=label).grid(row=row, column=0, sticky="w", padx=(0, 8), pady=2)
        var = tk.StringVar(value=default)
        box = ttk.Combobox(
            self,
            textvariable=var,
            values=list(values),
            width=54,
            state="readonly" if readonly else tk.NORMAL,
        )
        box.grid(row=row, column=1, sticky="we", pady=2)
        self.app.bind_var(key, var)
        return var

    def add_check(self, label: str, key: str, default: bool = False) -> tk.BooleanVar:
        row = self._next_row()
        var = tk.BooleanVar(value=default)
        ttk.Checkbutton(self, text=label, variable=var).grid(
            row=row, column=0, columnspan=2, sticky="w", pady=2
        )
        self.app.bind_var(key, var)
        return var

    def attach_log(self, rows: int = 14) -> None:
        self.log.text.configure(height=rows)
        row = self._next_row()
        self.log.grid(row=row, column=0, columnspan=3, sticky="nsew", pady=(10, 0))
        self.columnconfigure(1, weight=1)
        self.rowconfigure(row, weight=1)

    def _browse(self, var: tk.StringVar, kind: str, save: bool) -> None:
        if kind == "dir":
            path = filedialog.askdirectory(initialdir=REPO_ROOT)
        elif save:
            path = filedialog.asksaveasfilename(initialdir=REPO_ROOT)
        else:
            path = filedialog.askopenfilename(initialdir=REPO_ROOT)
        if path:
            var.set(path)


class LiveTab(BaseTab):
    def __init__(self, master, app):
        super().__init__(master, app, "Watch live")
        self.runner = CommandRunner(self.log.append)
        self.model_map: dict[str, Path] = {}
        self.add_entry("Live stream URL", "live.url")
        self.streamer = self.add_combo("Streamer", "live.streamer", sorted(STREAMER_PRESETS), "nso")
        self.source_id = self.add_entry("Source ID (routes key)", "live.source_id", "nso-live")
        self.model = self.add_combo("Model", "live.model", [])
        ttk.Button(self, text="Refresh models", command=self.refresh_models).grid(
            row=self._row - 1, column=2, sticky="w", padx=(6, 0), pady=2
        )
        self.roi = self.add_entry("Chart ROI x,y,w,h", "live.roi", STREAMER_PRESETS["nso"]["roi"])
        self.device = self.add_combo("Device", "live.device", DEVICES, "cpu")
        self.fps = self.add_entry("Decode FPS", "live.fps", "2.0", width=10)
        self.threshold = self.add_entry("Open threshold", "live.threshold", "0.85", width=10)
        self.routes_config = self.add_entry(
            "Webhook routes JSON", "live.routes_config", default_routes_config(), browse="file"
        )
        self.anchor_templates: list[Path] = []
        self.anchor_template = self.add_combo("Anchor template PNG", "live.anchor_template", [])
        self.anchor_threshold = self.add_entry("Anchor threshold", "live.anchor_threshold", "0.72", width=10)
        self.read_details = self.add_check(
            "Read trade details on open events (VLM, ~3-8 s per event)", "live.read_details"
        )
        self.verbose = self.add_check("Verbose per-frame log", "live.verbose")
        self.save_events = self.add_check("Save events JSONL (runs/events/)", "live.save_events", True)
        buttons = ttk.Frame(self)
        buttons.grid(row=self._next_row(), column=0, columnspan=3, sticky="w", pady=(8, 0))
        self.start_btn = ttk.Button(buttons, text="Start", command=self.on_start)
        self.stop_btn = ttk.Button(buttons, text="Stop", command=self.on_stop, state=tk.DISABLED)
        self.start_btn.pack(side=tk.LEFT)
        self.stop_btn.pack(side=tk.LEFT, padx=(8, 0))
        self.attach_log()
        self.streamer.trace_add("write", lambda *_: self.apply_preset())
        self.refresh_models()
        self.refresh_anchors()
        self.after(300, self._sync)

    def refresh_models(self) -> None:
        models = find_models()
        self.model_map = {model_label(m): m for m in models}
        self.model["values"] = list(self.model_map)
        current = self.model.get()
        if current in self.model_map:
            return
        pref = preferred_model(models)
        if pref:
            self.model.set(pref)

    def refresh_anchors(self) -> None:
        self.anchor_templates = find_anchor_templates()
        self.anchor_template["values"] = [str(p) for p in self.anchor_templates]
        if not self.anchor_template.get() and self.anchor_templates:
            self.anchor_template.set(str(self.anchor_templates[0]))

    def apply_preset(self) -> None:
        name = self.streamer.get().strip().lower()
        preset = STREAMER_PRESETS.get(name)
        if not preset:
            return
        self.roi.set(preset["roi"])
        self.source_id.set(f"{name}-live")

    def on_start(self) -> None:
        url = self.app.get("live.url", self).strip()
        if not url:
            messagebox.showerror("Missing input", "Paste a live stream URL first.")
            return
        model = self.model_map.get(self.model.get())
        if model is None:
            messagebox.showerror("Missing model", "Pick a trained model (Refresh models if empty).")
            return
        events_out = None
        if self.save_events.get():
            events_out = str(events_out_path(self.source_id.get()))
        argv = build_watch_command(
            url=url,
            model=str(model),
            source_id=self.source_id.get().strip() or "live-stream",
            roi=self.roi.get().strip(),
            device=self.device.get(),
            fps=self.fps.get().strip() or "2.0",
            threshold=self.threshold.get().strip() or "0.85",
            routes_config=self.routes_config.get().strip(),
            anchor_template=self.anchor_template.get().strip(),
            anchor_threshold=self.anchor_threshold.get().strip() or "0.72",
            read_details=self.read_details.get(),
            verbose=self.verbose.get(),
            events_out=events_out,
        )
        self.runner.start(argv)

    def on_stop(self) -> None:
        self.runner.stop()

    def _sync(self) -> None:
        running = self.runner.is_running()
        self.start_btn.configure(state=tk.DISABLED if running else tk.NORMAL)
        self.stop_btn.configure(state=tk.NORMAL if running else tk.DISABLED)
        self.after(300, self._sync)


class VodTab(BaseTab):
    def __init__(self, master, app, process_lock: threading.Lock):
        super().__init__(master, app, "Past stream (VOD)")
        self.lock = process_lock
        self.runner = CommandRunner(self.log.append)
        self.model_map: dict[str, Path] = {}
        self.add_entry("VOD URL", "vod.url")
        self.streamer = self.add_combo("Streamer", "vod.streamer", sorted(STREAMER_PRESETS), "nso")
        self.platform = self.add_combo("Platform", "vod.platform", PLATFORMS, "topstepx")
        self.mode = self.add_combo("Mode", "vod.mode", ("ingest", "loop", "train", "evaluate"), "ingest")
        self.roi = self.add_entry("Chart ROI x,y,w,h", "vod.roi", STREAMER_PRESETS["nso"]["roi"])
        self.model = self.add_combo("Classifier checkpoint (train/evaluate)", "vod.model", [])
        ttk.Button(self, text="Refresh models", command=self.refresh_models).grid(
            row=self._row - 1, column=2, sticky="w", padx=(6, 0), pady=2
        )
        self.data_root = self.add_entry(
            "Private data root", "vod.data_root", str(default_data_root()), browse="dir"
        )
        self.intervals = self.add_entry(
            "Pooled state_intervals.jsonl (blank = default)", "vod.intervals", "", browse="file"
        )
        self.run_name = self.add_entry("Training run name (blank = auto)", "vod.run_name", "", width=30)
        self.cleanup_media = self.add_check(
            "Delete downloaded VOD video after extraction (recommended)", "vod.cleanup_media", True
        )
        self.train_detector = self.add_check("Also train pooled UI detector", "vod.train_detector")
        self.dry_run = self.add_check("Dry run", "vod.dry_run")
        buttons = ttk.Frame(self)
        buttons.grid(row=self._next_row(), column=0, columnspan=3, sticky="w", pady=(8, 0))
        self.start_btn = ttk.Button(buttons, text="Start", command=self.on_start)
        self.stop_btn = ttk.Button(buttons, text="Stop", command=self.on_stop, state=tk.DISABLED)
        self.start_btn.pack(side=tk.LEFT)
        self.stop_btn.pack(side=tk.LEFT, padx=(8, 0))
        self.attach_log()
        self.streamer.trace_add("write", lambda *_: self.apply_preset())
        self.refresh_models()
        self.after(300, self._sync)

    def refresh_models(self) -> None:
        models = find_models()
        self.model_map = {model_label(m): m for m in models}
        self.model["values"] = list(self.model_map)
        current = self.model.get()
        if current in self.model_map:
            return
        pref = preferred_model(models)
        if pref:
            self.model.set(pref)

    def apply_preset(self) -> None:
        name = self.streamer.get().strip().lower()
        preset = STREAMER_PRESETS.get(name)
        if preset:
            self.platform.set(preset["platform"])
            self.roi.set(preset["roi"])

    def on_start(self) -> None:
        url = self.app.get("vod.url", self).strip()
        if not url:
            messagebox.showerror("Missing input", "Paste a past-stream (VOD) URL first.")
            return
        if not self.streamer.get().strip():
            messagebox.showerror("Missing input", "Streamer is required.")
            return
        if not self.lock.acquire(blocking=False):
            messagebox.showerror(
                "Busy", "A processing command is already running (VOD and Batch share one slot)."
            )
            return
        model_path = self.model_map.get(self.model.get())
        argv = build_train_command(
            streamer=self.streamer.get().strip(),
            platform=self.platform.get(),
            mode=self.mode.get(),
            url=url,
            roi=self.roi.get().strip(),
            data_root=self.data_root.get().strip(),
            intervals=self.intervals.get().strip(),
            model=str(model_path) if model_path else "",
            run_name=self.run_name.get().strip(),
            train_detector=self.train_detector.get(),
            cleanup_media=self.cleanup_media.get(),
            dry_run=self.dry_run.get(),
        )

        def _done(rc: int, lines: list[str]) -> None:
            self.lock.release()

        if not self.runner.start(argv, on_done=_done):
            self.lock.release()

    def on_stop(self) -> None:
        self.runner.stop()

    def _sync(self) -> None:
        running = self.runner.is_running()
        self.start_btn.configure(state=tk.DISABLED if running else tk.NORMAL)
        self.stop_btn.configure(state=tk.NORMAL if running else tk.DISABLED)
        self.after(300, self._sync)


class BatchTab(BaseTab):
    _MAX_CHANNEL_PASSES = 200

    def __init__(self, master, app, process_lock: threading.Lock):
        super().__init__(master, app, "Batch")
        self.lock = process_lock
        self.runner = CommandRunner(self.log.append)
        self._cancel = False
        self._progress = ""
        self.add_entry(
            "Channel/streams page URL (processes every unseen VOD)",
            "batch.channel",
        )
        self.max_vods = self.add_entry("Max VODs per discovery", "batch.max_vods", "10", width=10)
        row = self._next_row()
        ttk.Label(self, text="URLs (one per line)").grid(row=row, column=0, sticky="nw", padx=(0, 8))
        self.urls_text = scrolledtext.ScrolledText(self, height=6, font=("Consolas", 9), wrap=tk.WORD)
        self.urls_text.grid(row=row, column=1, sticky="we", pady=2)
        self.app.bind_var("batch.urls", _TextVar(self.urls_text))
        self.urls_text.bind("<<Modified>>", lambda e: self.app._persist_later())
        self.streamer = self.add_combo("Streamer", "batch.streamer", sorted(STREAMER_PRESETS), "nso")
        self.platform = self.add_combo("Platform", "batch.platform", PLATFORMS, "topstepx")
        self.mode = self.add_combo("Mode", "batch.mode", ("ingest", "loop"), "ingest")
        self.roi = self.add_entry("Chart ROI x,y,w,h", "batch.roi", STREAMER_PRESETS["nso"]["roi"])
        self.data_root = self.add_entry(
            "Private data root", "batch.data_root", str(default_data_root()), browse="dir"
        )
        self.cleanup_media = self.add_check(
            "Delete downloaded VOD video after extraction (recommended)", "batch.cleanup_media", True
        )
        self.train_detector = self.add_check("Also train pooled UI detector", "batch.train_detector")
        self.dry_run = self.add_check("Dry run (list what would run)", "batch.dry_run")
        buttons = ttk.Frame(self)
        buttons.grid(row=self._next_row(), column=0, columnspan=3, sticky="w", pady=(8, 0))
        self.start_btn = ttk.Button(buttons, text="Start batch", command=self.on_start)
        self.stop_btn = ttk.Button(buttons, text="Stop", command=self.on_stop, state=tk.DISABLED)
        self.start_btn.pack(side=tk.LEFT)
        self.stop_btn.pack(side=tk.LEFT, padx=(8, 0))
        self.progress_lbl = ttk.Label(buttons, text="")
        self.progress_lbl.pack(side=tk.LEFT, padx=(12, 0))
        self.attach_log(rows=12)
        self.streamer.trace_add("write", lambda *_: self.apply_preset())
        self.after(300, self._sync)

    def apply_preset(self) -> None:
        name = self.streamer.get().strip().lower()
        preset = STREAMER_PRESETS.get(name)
        if preset:
            self.platform.set(preset["platform"])
            self.roi.set(preset["roi"])

    def on_start(self) -> None:
        plan = plan_batch(self.urls_text.get("1.0", "end-1c"), self.app.get("batch.channel", self))
        if not plan:
            messagebox.showerror(
                "Missing input", "Provide a channel page URL and/or a list of VOD URLs."
            )
            return
        if not self.streamer.get().strip():
            messagebox.showerror("Missing input", "Streamer is required.")
            return
        if not self.lock.acquire(blocking=False):
            messagebox.showerror(
                "Busy", "A processing command is already running (VOD and Batch share one slot)."
            )
            return
        self._cancel = False
        threading.Thread(target=self._worker, args=(plan,), daemon=True).start()

    def on_stop(self) -> None:
        self._cancel = True
        self.runner.stop()

    def _worker(self, plan: list[tuple[str, str]]) -> None:
        try:
            for idx, (kind, url) in enumerate(plan, 1):
                if self._cancel:
                    break
                self._progress = f"[{idx}/{len(plan)}] {url}"
                self.log.append(self._progress)
                if kind == "vod":
                    self._run_one(channel="", url=url)
                else:
                    self._drain_channel(url)
            self._progress = "[batch] cancelled" if self._cancel else "[batch] done"
            self.log.append(self._progress)
        finally:
            self.lock.release()

    def _run_one(self, *, channel: str, url: str) -> tuple[int, list[str]]:
        argv = build_train_command(
            streamer=self.streamer.get().strip(),
            platform=self.platform.get(),
            mode=self.mode.get(),
            url=url,
            channel=channel,
            roi=self.roi.get().strip(),
            data_root=self.data_root.get().strip(),
            max_vods=self.max_vods.get().strip(),
            train_detector=self.train_detector.get(),
            cleanup_media=self.cleanup_media.get(),
            dry_run=self.dry_run.get(),
        )
        done = threading.Event()
        holder: dict[str, tuple[int, list[str]]] = {}

        def _on_done(rc: int, lines: list[str]) -> None:
            holder["result"] = (rc, lines)
            done.set()

        if not self.runner.start(argv, on_done=_on_done):
            holder["result"] = (-1, [])
            done.set()
        done.wait()
        return holder["result"]

    def _drain_channel(self, url: str) -> None:
        if self.dry_run.get():
            self._run_one(channel=url, url="")
            return
        for _ in range(self._MAX_CHANNEL_PASSES):
            if self._cancel:
                return
            rc, lines = self._run_one(channel=url, url="")
            if rc != 0:
                return
            if any("already been processed" in ln for ln in lines):
                return

    def _sync(self) -> None:
        running = self.runner.is_running()
        self.start_btn.configure(state=tk.DISABLED if running else tk.NORMAL)
        self.stop_btn.configure(state=tk.NORMAL if running else tk.DISABLED)
        if self._progress and self.progress_lbl.cget("text") != self._progress:
            self.progress_lbl.configure(text=self._progress[:120])
        self.after(300, self._sync)


class App(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("stream-signal-vision")
        self.geometry("1060x760")
        self._settings: dict[str, str] = {}
        self._bound: list[tuple[str, object]] = []
        self._persist_scheduled = False
        self._load_settings()
        process_lock = threading.Lock()
        self.note = ttk.Notebook(self)
        self.live_tab = LiveTab(self.note, self)
        self.vod_tab = VodTab(self.note, self, process_lock)
        self.batch_tab = BatchTab(self.note, self, process_lock)
        for tab, name in (
            (self.live_tab, "Watch live"),
            (self.vod_tab, "Past stream (VOD)"),
            (self.batch_tab, "Batch"),
        ):
            self.note.add(tab, text=name)
        self.note.pack(fill=tk.BOTH, expand=True)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def get(self, key: str, widget=None) -> str:
        for k, var in self._bound:
            if k == key:
                return str(var.get())
        return ""

    def bind_var(self, key: str, var) -> None:
        if key in self._settings:
            try:
                var.set(self._settings[key])
            except tk.TclError:
                pass
        self._bound.append((key, var))
        if hasattr(var, "trace_add"):
            var.trace_add("write", lambda *_: self._persist_later())

    def _persist_later(self) -> None:
        if not self._persist_scheduled:
            self._persist_scheduled = True
            self.after(500, self._persist)

    def _persist(self) -> None:
        self._persist_scheduled = False
        data = {k: var.get() for k, var in self._bound}
        try:
            SETTINGS_PATH.write_text(
                json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
        except OSError:
            pass

    def _load_settings(self) -> None:
        try:
            if SETTINGS_PATH.is_file():
                loaded = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    self._settings = {str(k): v for k, v in loaded.items() if isinstance(v, (str, bool, int, float))}
        except (OSError, json.JSONDecodeError):
            self._settings = {}

    def _on_close(self) -> None:
        self._persist()
        self.live_tab.runner.stop()
        self.vod_tab.runner.stop()
        self.batch_tab.runner.stop()
        self.batch_tab._cancel = True
        self.destroy()


def main() -> int:
    app = App()
    app.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
