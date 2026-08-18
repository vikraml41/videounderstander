"""Step 1 — dense sampling + tile-grid structural deduplication.

Strategy (PLAN.md): sample at a fixed fps, compare consecutive frames on a
coarse tile grid, and classify each transition:

  stable  — no tile changed meaningfully; same visual state
  build   — a small, *localized* region changed (new bullet, next stroke of a
            derivation): end the interval, keep going in the same accretion
            group
  cut     — a global change (scene cut, camera move): new group

Global wobble and brightness shifts move *every* tile a little; subtracting
the median tile-diff before thresholding cancels them, so only changes that
are large relative to the frame's overall motion count.

For each stable interval the *final* frame is kept (the completed build
step). Dedup math runs on tiny grayscale PGM thumbnails so it needs no
numpy/Pillow — ffmpeg writes them, stdlib parses them.
"""

from __future__ import annotations

import json
import shutil
import statistics
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from .artifact import Artifact, fmt_ts

THUMB_WIDTH = 64          # dedup thumbnail width in px (height keeps aspect)
GRID = 8                  # tile grid is GRID x GRID
TILE_THRESHOLD = 14.0     # mean abs pixel diff (0-255) for a tile to count as changed
BUILD_FRAC = 0.25         # <= this fraction of tiles changed -> build step
GLOBAL_CUT_THRESHOLD = 25.0  # median tile-diff above this = whole frame changed
DEFAULT_FPS = 1.0


class FfmpegMissing(RuntimeError):
    pass


def _require(binary: str) -> str:
    path = shutil.which(binary)
    if not path:
        raise FfmpegMissing(
            f"'{binary}' not found on PATH — install ffmpeg to run sampling")
    return path


# ---------------------------------------------------------------------------
# PGM parsing (stdlib; ffmpeg emits binary P5)
# ---------------------------------------------------------------------------

def parse_pgm(data: bytes) -> tuple[int, int, bytes]:
    """Parse a binary (P5) PGM with maxval <= 255 into (width, height, pixels)."""
    tokens: list[bytes] = []
    i = 0
    while len(tokens) < 4:
        while i < len(data) and data[i : i + 1].isspace():
            i += 1
        if data[i : i + 1] == b"#":
            while i < len(data) and data[i] != 0x0A:
                i += 1
            continue
        start = i
        while i < len(data) and not data[i : i + 1].isspace():
            i += 1
        tokens.append(data[start:i])
    if tokens[0] != b"P5":
        raise ValueError(f"not a binary PGM (magic {tokens[0]!r})")
    width, height, maxval = int(tokens[1]), int(tokens[2]), int(tokens[3])
    if maxval > 255:
        raise ValueError("16-bit PGM not supported")
    pixels = data[i + 1 : i + 1 + width * height]
    if len(pixels) != width * height:
        raise ValueError("truncated PGM pixel data")
    return width, height, pixels


# ---------------------------------------------------------------------------
# Tile diff + transition classification (pure functions, unit-tested)
# ---------------------------------------------------------------------------

def tile_diffs(a: bytes, b: bytes, width: int, height: int,
               grid: int = GRID) -> list[float]:
    """Mean absolute pixel difference per tile of a grid x grid split."""
    diffs: list[float] = []
    for ty in range(grid):
        y0 = ty * height // grid
        y1 = (ty + 1) * height // grid
        for tx in range(grid):
            x0 = tx * width // grid
            x1 = (tx + 1) * width // grid
            total = 0
            count = 0
            for y in range(y0, y1):
                row = y * width
                seg_a = a[row + x0 : row + x1]
                seg_b = b[row + x0 : row + x1]
                total += sum(abs(pa - pb) for pa, pb in zip(seg_a, seg_b))
                count += x1 - x0
            diffs.append(total / count if count else 0.0)
    return diffs


def classify_transition(diffs: list[float],
                        tile_threshold: float = TILE_THRESHOLD,
                        build_frac: float = BUILD_FRAC) -> str:
    """Classify a frame transition as 'stable', 'build', or 'cut'.

    The median tile-diff is subtracted before thresholding so *small* global
    motion (camera wobble, brightness shifts) doesn't register — only tiles
    that changed more than the frame overall count. But when the median
    itself is large, most of the frame genuinely changed: that's a cut, and
    subtracting it would cancel the whole transition.
    """
    med = statistics.median(diffs)
    if med > GLOBAL_CUT_THRESHOLD:
        return "cut"
    changed = sum(1 for d in diffs if d - med > tile_threshold)
    if changed == 0:
        return "stable"
    if changed / len(diffs) <= build_frac:
        return "build"
    return "cut"


@dataclass
class Interval:
    """A maximal run of visually-stable frames."""
    start_idx: int
    end_idx: int          # inclusive
    group: int            # accretion group id; builds share a group, cuts break it
    entered_by: str       # 'start' | 'build' | 'cut' — how this interval began
    exited_by: str = "end"  # filled when the interval closes


def segment_frames(frames: list[bytes], width: int, height: int,
                   tile_threshold: float = TILE_THRESHOLD,
                   build_frac: float = BUILD_FRAC,
                   grid: int = GRID) -> list[Interval]:
    """Split a frame sequence into stable intervals with accretion groups."""
    if not frames:
        return []
    intervals: list[Interval] = []
    group = 0
    current = Interval(0, 0, group, "start")
    for i in range(1, len(frames)):
        kind = classify_transition(
            tile_diffs(frames[i - 1], frames[i], width, height, grid),
            tile_threshold, build_frac)
        if kind == "stable":
            current.end_idx = i
            continue
        current.exited_by = kind
        intervals.append(current)
        if kind == "cut":
            group += 1
        current = Interval(i, i, group, kind)
    intervals.append(current)
    return intervals


def select_keepers(intervals: list[Interval]) -> list[Interval]:
    """Drop transition junk: a single-frame interval both entered and exited
    by a cut is a fade/wipe artifact, not content."""
    kept = []
    for iv in intervals:
        if (iv.end_idx == iv.start_idx and iv.entered_by == "cut"
                and iv.exited_by == "cut"):
            continue
        kept.append(iv)
    return kept


# ---------------------------------------------------------------------------
# ffmpeg orchestration
# ---------------------------------------------------------------------------

def probe_duration(video: Path) -> float:
    ffprobe = _require("ffprobe")
    out = subprocess.run(
        [ffprobe, "-v", "error", "-print_format", "json", "-show_format",
         str(video)],
        check=True, capture_output=True, text=True).stdout
    return float(json.loads(out)["format"]["duration"])


def extract_thumbs(video: Path, thumbs_dir: Path, fps: float) -> list[Path]:
    """Write small grayscale PGM thumbnails at `fps` for dedup."""
    ffmpeg = _require("ffmpeg")
    thumbs_dir.mkdir(parents=True, exist_ok=True)
    for old in thumbs_dir.glob("*.pgm"):
        old.unlink()
    subprocess.run(
        [ffmpeg, "-v", "error", "-i", str(video),
         "-vf", f"fps={fps},scale={THUMB_WIDTH}:-2,format=gray",
         "-f", "image2", "-c:v", "pgm",
         str(thumbs_dir / "%06d.pgm")],
        check=True, capture_output=True)
    return sorted(thumbs_dir.glob("*.pgm"))


def extract_frame(video: Path, t: float, out_path: Path) -> None:
    """Extract one full-resolution frame at time t as high-quality JPEG."""
    ffmpeg = _require("ffmpeg")
    subprocess.run(
        [ffmpeg, "-v", "error", "-ss", f"{t:.3f}", "-i", str(video),
         "-frames:v", "1", "-q:v", "2", "-y", str(out_path)],
        check=True, capture_output=True)


def run_sampling(video: Path, art: Artifact, fps: float = DEFAULT_FPS,
                 tile_threshold: float = TILE_THRESHOLD,
                 build_frac: float = BUILD_FRAC) -> list[dict]:
    """Full step-1 pass: thumbs -> intervals -> kept full-res frames + manifest."""
    art.ensure()
    thumb_paths = extract_thumbs(video, art.thumbs_dir, fps)
    if not thumb_paths:
        raise RuntimeError("ffmpeg produced no thumbnails — is the input a video?")

    first = parse_pgm(thumb_paths[0].read_bytes())
    width, height = first[0], first[1]
    frames = [parse_pgm(p.read_bytes())[2] for p in thumb_paths]

    intervals = select_keepers(
        segment_frames(frames, width, height, tile_threshold, build_frac))

    manifest: list[dict] = []
    for iv in intervals:
        # thumbnail n (1-based on disk, 0-based here) samples t ~= idx / fps;
        # the *final* frame of the interval is the completed build step
        t_keep = iv.end_idx / fps
        t_start = iv.start_idx / fps
        # decisecond resolution so names stay unique at fps > 1
        name = f"frame_{t_keep:08.1f}.jpg"
        out_path = art.frames_dir / name
        extract_frame(video, t_keep, out_path)
        manifest.append({
            "t": t_keep,
            "ts": fmt_ts(t_keep),
            "interval_start": t_start,
            "interval_end": t_keep,
            "file": f"frames/{name}",
            "group": iv.group,
            "entered_by": iv.entered_by,
        })

    Artifact.write_json(art.frames_manifest, manifest)
    return manifest
