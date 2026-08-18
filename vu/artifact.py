"""Artifact directory layout and shared helpers.

artifact/
  source.json          # metadata: source path/URL, duration, assumed_context
  transcript.json      # normalized transcript segments (post-ASR, pre-correction)
  transcript.corrected.json  # after OCR-based correction
  frames/              # kept full-res frames, frame_SSSSSS.S.jpg + manifest.json
  thumbs/              # small grayscale PGMs used for dedup (disposable)
  ocr.json             # on-screen text per kept frame
  descriptions.json    # step-3 output per kept frame
  reconstructions/     # step-4 output files + index.json
  aligned.md           # step-2 interleaved context
  resolved.md          # step-5 deixis-resolved transcript
  distilled.yaml       # step-6 schema output
  qa/                  # verification questions / answers / reports per iteration
  cache/               # LLM response cache keyed by request hash
"""

from __future__ import annotations

import json
import os
from pathlib import Path


class Artifact:
    """Handle to one artifact directory; owns paths and small JSON I/O."""

    def __init__(self, root: str | os.PathLike):
        self.root = Path(root)

    # -- directories -------------------------------------------------------
    @property
    def frames_dir(self) -> Path:
        return self.root / "frames"

    @property
    def thumbs_dir(self) -> Path:
        return self.root / "thumbs"

    @property
    def reconstructions_dir(self) -> Path:
        return self.root / "reconstructions"

    @property
    def qa_dir(self) -> Path:
        return self.root / "qa"

    @property
    def cache_dir(self) -> Path:
        return self.root / "cache"

    # -- files -------------------------------------------------------------
    @property
    def source_json(self) -> Path:
        return self.root / "source.json"

    @property
    def transcript_json(self) -> Path:
        return self.root / "transcript.json"

    @property
    def corrected_transcript_json(self) -> Path:
        return self.root / "transcript.corrected.json"

    @property
    def frames_manifest(self) -> Path:
        return self.frames_dir / "manifest.json"

    @property
    def ocr_json(self) -> Path:
        return self.root / "ocr.json"

    @property
    def descriptions_json(self) -> Path:
        return self.root / "descriptions.json"

    @property
    def reconstructions_index(self) -> Path:
        return self.reconstructions_dir / "index.json"

    @property
    def aligned_md(self) -> Path:
        return self.root / "aligned.md"

    @property
    def resolved_md(self) -> Path:
        return self.root / "resolved.md"

    @property
    def distilled_yaml(self) -> Path:
        return self.root / "distilled.yaml"

    def ensure(self) -> "Artifact":
        for d in (self.root, self.frames_dir, self.thumbs_dir,
                  self.reconstructions_dir, self.qa_dir, self.cache_dir):
            d.mkdir(parents=True, exist_ok=True)
        return self

    # -- JSON helpers ------------------------------------------------------
    @staticmethod
    def read_json(path: Path, default=None):
        if not path.exists():
            if default is not None:
                return default
            raise FileNotFoundError(
                f"{path} not found — run the earlier pipeline stage first")
        return json.loads(path.read_text())

    @staticmethod
    def write_json(path: Path, obj) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n")

    def best_transcript(self) -> list[dict]:
        """The corrected transcript when present, else the raw one."""
        if self.corrected_transcript_json.exists():
            return self.read_json(self.corrected_transcript_json)
        return self.read_json(self.transcript_json)


def fmt_ts(seconds: float) -> str:
    """Format seconds as [MM:SS] or [H:MM:SS], used everywhere timestamps appear."""
    s = max(0, int(round(seconds)))
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    if h:
        return f"{h}:{m:02d}:{sec:02d}"
    return f"{m:02d}:{sec:02d}"


def parse_ts(text: str) -> float:
    """Parse H:MM:SS(.mmm), MM:SS(.mmm), or bare seconds into float seconds."""
    text = text.strip()
    parts = text.split(":")
    if len(parts) == 1:
        return float(parts[0])
    parts = [float(p.replace(",", ".")) for p in parts]
    sec = parts[-1]
    minutes = parts[-2] if len(parts) >= 2 else 0
    hours = parts[-3] if len(parts) >= 3 else 0
    return hours * 3600 + minutes * 60 + sec
