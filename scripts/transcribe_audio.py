"""Transcribe VOD/live-audio recordings with timestamped trading cues.

The transcript is a search/confluence signal only. An utterance never becomes
a trade event until the synchronized visual stream confirms the position/order.

Usage:
    python scripts/transcribe_audio.py recording.mp4 --out transcript.jsonl
"""
import argparse
import json
import os
import re
import sys
from pathlib import Path


PROMPT = (
    "Futures day trading discussion. Terms may include NQ, MNQ, ES, YM, long, "
    "short, market order, limit order, filled, contracts, stop loss, stop, "
    "take profit, target, breakeven, scale out, runner, flat. Preserve numbers "
    "and trading terms exactly; do not invent missing words."
)


def parse_time(value):
    parts = [float(part) for part in value.split(":")]
    if len(parts) == 3:
        return parts[0] * 3600 + parts[1] * 60 + parts[2]
    if len(parts) == 2:
        return parts[0] * 60 + parts[1]
    return parts[0]


def format_time(seconds):
    seconds = max(0, int(seconds))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def classify_cue(text):
    text = text.lower().replace("’", "'")
    directions = set()
    for match in re.finditer(r"\b(long|short|buy|sell)\b", text):
        prefix = text[max(0, match.start() - 24):match.start()]
        if re.search(r"\bnot(?:\W+\w+){0,2}\W*$", prefix):
            continue
        word = match.group(1)
        directions.add({"buy": "long", "sell": "short"}.get(word, word))
    direction = next(iter(directions)) if len(directions) == 1 else None
    if re.search(r"\b(move|moved|moving|raise|lower|tighten|trail|put|set|keep)\b.{0,30}\bstop\b|\bstop\b.{0,30}\b(breakeven|break even|entry|now|up|down)\b", text):
        cue = "stop_update"
    elif re.search(r"\b(move|moved|moving|adjust|adjusted|change|changed|target|take profit|tp\s*[123])\b", text) and re.search(r"\b(target|take profit|tp\s*[123])\b", text):
        cue = "target_update"
    elif re.search(r"\b(may|might|could|looks like|probably)\b.{0,40}\b(stop me out|stop out|stopped out)\b", text):
        cue = "exit_warning"
    elif re.search(r"\b(flat|closed (my |the )?(position|trade)|out of (the )?trade|stopped out|stop(ped)? me out|i'?m out|took profit)\b", text):
        cue = "exit_claim"
    elif re.search(r"\b(i'?m|i am|we're|we are)\s+(?:(?:currently|still|already)\s+|in a\s+)?(long|short)\b|\b(got|just got|filled|position filled)\b.{0,20}\b(long|short)\b|\b(long|short)\b.{0,20}\b(filled|in now|i'?m in|we're in)\b", text):
        cue = "position_claim"
    elif re.search(r"\b(long|short)\s+(trade|position)\b|\b(position|trade)\b.{0,20}\b(long|short)\b", text):
        cue = "position_claim"
    elif re.search(
        r"\b(place|placed|put in|working|pending|looking for|set up|setting up)\b.{0,32}\b(order|limit|long|short|buy|sell)\b"
        r"|\b(limit|market)\s+(buy|sell|order)\b"
        r"|\b(long|short)\b.{0,32}\b(off|on|from|here|setup|cisd|gap)\b"
        r"|\b(?:i'?m|i am|we'll|we will|let's)\s+(?:gonna\s+|going to\s+)?(?:try|take|enter)\s+(?:it|this|a\s+long|a\s+short|long|short)\b",
        text,
    ) and not re.search(
        r"\b(not|never|wouldn't|shouldn't|can't|won't|don't|nowhere)\b.{0,32}\b(try|short|long|setup|order|buy|sell|entry)\b"
        r"|\bnowhere else\b.{0,64}\b(short|long|setup|order|buy|sell|entry)\b",
        text,
    ):
        cue = "order_intent"
    else:
        cue = "none"
    return cue, direction


def configure_cuda_dll_paths():
    site_packages = Path(sys.prefix) / "Lib" / "site-packages"
    candidates = [site_packages / "torch" / "lib"]
    candidates.extend((site_packages / "nvidia").glob("*"))
    handles = []
    for candidate in candidates:
        directory = candidate / "bin" if candidate.parent.name == "nvidia" else candidate
        if directory.is_dir():
            os.environ["PATH"] = str(directory) + os.pathsep + os.environ.get("PATH", "")
            handles.append(os.add_dll_directory(str(directory)))
    return handles


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("media", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", default="large-v3-turbo")
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    parser.add_argument("--compute-type", default="int8_float16")
    parser.add_argument("--language", default="en")
    parser.add_argument("--start", default=None, help="optional clip start HH:MM:SS")
    parser.add_argument("--end", default=None, help="optional clip end HH:MM:SS")
    args = parser.parse_args()

    dll_handles = configure_cuda_dll_paths() if args.device == "cuda" else []
    from faster_whisper import WhisperModel

    model = WhisperModel(args.model, device=args.device, compute_type=args.compute_type)
    clip = None
    if args.start is not None:
        clip = [parse_time(args.start), parse_time(args.end)] if args.end else [parse_time(args.start)]
    segments, info = model.transcribe(
        str(args.media),
        language=args.language,
        task="transcribe",
        beam_size=5,
        temperature=0,
        initial_prompt=PROMPT,
        hotwords="NQ, MNQ, ES, long, short, breakeven, stop loss, take profit",
        vad_filter=True,
        word_timestamps=False,
        clip_timestamps=clip or "0",
        condition_on_previous_text=False,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    count = cue_count = 0
    with args.out.open("w", encoding="utf-8") as out:
        for segment in segments:
            text = segment.text.strip()
            cue, direction = classify_cue(text)
            start = float(segment.start)
            end = float(segment.end)
            row = {
                "start_sec": round(start, 3),
                "end_sec": round(end, 3),
                "start": format_time(start),
                "end": format_time(end),
                "text": text,
                "cue": cue,
                "direction": direction,
                "avg_logprob": round(float(segment.avg_logprob), 4),
                "no_speech_prob": round(float(segment.no_speech_prob), 4),
            }
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
            if cue != "none":
                cue_count += 1
                print(f"{row['start']} [{cue}] {text}")
    print(f"transcribed {count} segments ({cue_count} trading cues); detected language={info.language}; saved {args.out}")


if __name__ == "__main__":
    main()
