"""Step 0a — transcript acquisition and normalization.

The pipeline consumes a normalized transcript: a list of segments

    {"start": float, "end": float, "text": str,
     "words": [{"start": float, "end": float, "word": str}, ...]?}

Sources, in order of preference:
  1. An existing subtitle/transcript file (.vtt, .srt, or whisper-style .json)
     — pass it with `vu transcribe --from FILE`.
  2. Local ASR via faster-whisper (`pip install videounderstander[asr]`),
     run with word timestamps so the step-2 interleave aligns tightly.

Word-level timestamps matter: segment-level timestamps drift seconds, which
silently breaks frame/speech adjacency downstream.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from .artifact import Artifact, parse_ts

_TIME_LINE = re.compile(
    r"(\d{1,2}:)?\d{1,2}:\d{2}[.,]\d{1,3}\s*-->\s*(\d{1,2}:)?\d{1,2}:\d{2}[.,]\d{1,3}")
_TAG = re.compile(r"<[^>]+>")


def _clean_cue_text(lines: list[str]) -> str:
    text = " ".join(lines)
    text = _TAG.sub("", text)
    return re.sub(r"\s+", " ", text).strip()


def parse_vtt_or_srt(text: str) -> list[dict]:
    """Parse WebVTT or SRT into normalized segments (no word timestamps)."""
    segments: list[dict] = []
    cur: dict | None = None
    cur_lines: list[str] = []

    def flush():
        nonlocal cur, cur_lines
        if cur is not None:
            body = _clean_cue_text(cur_lines)
            if body:
                cur["text"] = body
                segments.append(cur)
        cur, cur_lines = None, []

    for raw in text.splitlines():
        line = raw.strip("﻿").rstrip()
        if _TIME_LINE.search(line):
            flush()
            start_s, end_s = [p.strip() for p in line.split("-->")[:2]]
            # strip cue settings after the end timestamp ("00:02.000 line:0")
            end_s = end_s.split()[0]
            cur = {"start": parse_ts(start_s), "end": parse_ts(end_s)}
        elif cur is not None:
            if line == "":
                flush()
            elif not (line.isdigit() and not cur_lines):
                cur_lines.append(line)
    flush()

    # auto-captions often repeat the previous cue's text; drop exact repeats
    deduped: list[dict] = []
    for seg in segments:
        if deduped and seg["text"] == deduped[-1]["text"]:
            deduped[-1]["end"] = seg["end"]
        else:
            deduped.append(seg)
    return deduped


def parse_whisper_json(obj: dict) -> list[dict]:
    """Normalize whisper / faster-whisper style JSON output."""
    segments = []
    for seg in obj.get("segments", []):
        entry = {
            "start": float(seg["start"]),
            "end": float(seg["end"]),
            "text": re.sub(r"\s+", " ", seg["text"]).strip(),
        }
        if seg.get("words"):
            entry["words"] = [
                {"start": float(w["start"]), "end": float(w["end"]),
                 "word": w.get("word", w.get("text", "")).strip()}
                for w in seg["words"]
            ]
        segments.append(entry)
    return segments


def load_transcript_file(path: Path) -> list[dict]:
    text = path.read_text()
    suffix = path.suffix.lower()
    if suffix == ".json":
        return parse_whisper_json(json.loads(text))
    if suffix in (".vtt", ".srt"):
        return parse_vtt_or_srt(text)
    raise ValueError(f"unsupported transcript format: {suffix} "
                     "(expected .vtt, .srt, or whisper .json)")


def run_asr(video: Path, model_size: str = "medium") -> list[dict]:
    """Local ASR with word timestamps via faster-whisper (optional extra)."""
    try:
        from faster_whisper import WhisperModel  # type: ignore
    except ImportError as e:
        raise RuntimeError(
            "faster-whisper is not installed. Either install it "
            "(`pip install 'videounderstander[asr]'`) or supply an existing "
            "transcript with `vu transcribe --from FILE.vtt`") from e

    model = WhisperModel(model_size)
    segments, _info = model.transcribe(str(video), word_timestamps=True)
    out = []
    for seg in segments:
        entry = {
            "start": float(seg.start),
            "end": float(seg.end),
            "text": re.sub(r"\s+", " ", seg.text).strip(),
        }
        if seg.words:
            entry["words"] = [
                {"start": float(w.start), "end": float(w.end),
                 "word": w.word.strip()} for w in seg.words
            ]
        out.append(entry)
    return out


def run_transcribe(art: Artifact, video: Path | None = None,
                   from_file: Path | None = None,
                   model_size: str = "medium") -> list[dict]:
    art.ensure()
    if from_file is not None:
        segments = load_transcript_file(from_file)
    elif video is not None:
        segments = run_asr(video, model_size)
    else:
        raise ValueError("need either a video (for ASR) or --from FILE")
    Artifact.write_json(art.transcript_json, segments)
    return segments


def transcript_window(segments: list[dict], center: float,
                      radius: float = 30.0) -> str:
    """Speech within ±radius seconds of `center`, for context-aware vision calls."""
    parts = [seg["text"] for seg in segments
             if seg["end"] >= center - radius and seg["start"] <= center + radius]
    return " ".join(parts)


def full_text(segments: list[dict]) -> str:
    return " ".join(seg["text"] for seg in segments)
