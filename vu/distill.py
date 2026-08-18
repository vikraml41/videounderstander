"""Step 6 — schema-forced distillation with verbatim-quote provenance.

Chunks the resolved transcript on argument boundaries, distills each chunk
against the schema (claim / mechanism / contrast / example / scope /
epistemic status / novelty flag / verbatim quote), carrying a running
definitions-and-established-claims block between chunks.

The quote check is the mechanical anti-novelty-collapse guard: every claim's
quote must actually occur in the transcript (after normalization, with a
fuzzy fallback for sub-word ASR noise). A claim whose quote fails the check
is marked unverified in distilled.yaml — a loud failure instead of a silent
rewrite.
"""

from __future__ import annotations

import difflib
import re

import yaml

from .artifact import Artifact, fmt_ts
from .llm import LLM, extract_json
from . import prompts

TARGET_CHUNK_MIN = 5    # minutes
TARGET_CHUNK_MAX = 12


# ---------------------------------------------------------------------------
# Verbatim-quote checking (pure, unit-tested)
# ---------------------------------------------------------------------------

def _normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def quote_in_transcript(quote: str, transcript: str,
                        fuzzy_threshold: float = 0.85) -> bool:
    """True if `quote` occurs in `transcript`, tolerant of punctuation/case
    and (via fuzzy fallback) small ASR-noise differences."""
    q = _normalize(quote)
    t = _normalize(transcript)
    if not q:
        return False
    if q in t:
        return True
    q_words = q.split()
    t_words = t.split()
    if len(q_words) > len(t_words):
        return False
    window = len(q_words)
    step = max(1, window // 4)
    best = 0.0
    for i in range(0, len(t_words) - window + 1, step):
        candidate = " ".join(t_words[i : i + window])
        ratio = difflib.SequenceMatcher(None, q, candidate).ratio()
        if ratio > best:
            best = ratio
            if best >= fuzzy_threshold:
                return True
    return False


def check_quotes(chunks: list[dict], transcript_text: str) -> int:
    """Annotate every claim with quote_verified; return the failure count."""
    failures = 0
    for chunk in chunks:
        for claim in chunk.get("claims") or []:
            quote = claim.get("quote") or ""
            ok = quote_in_transcript(quote, transcript_text)
            claim["quote_verified"] = ok
            if not ok:
                failures += 1
    return failures


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def find_boundaries(llm: LLM, segments: list[dict]) -> list[float]:
    duration = segments[-1]["end"] if segments else 0
    if duration <= TARGET_CHUNK_MAX * 60:
        return []
    transcript = "\n".join(
        f"[{fmt_ts(s['start'])}] {s['text']}" for s in segments)
    raw = llm.ask(prompts.CHUNK_BOUNDARIES.format(
        target_min=TARGET_CHUNK_MIN, target_max=TARGET_CHUNK_MAX,
        transcript=transcript), max_tokens=4000)
    bounds = [float(b) for b in extract_json(raw)]
    return sorted(b for b in bounds if 0 < b < duration)


def split_lines_by_time(resolved_lines: list[str],
                        boundaries: list[float]) -> list[list[str]]:
    """Split resolved-transcript lines into chunks at boundary timestamps,
    keying on each line's leading [MM:SS] / [H:MM:SS] prefix."""
    from .artifact import parse_ts
    ts_prefix = re.compile(r"^\[(\d+:)?\d+:\d+")
    chunks: list[list[str]] = [[] for _ in range(len(boundaries) + 1)]
    idx = 0
    for line in resolved_lines:
        m = ts_prefix.match(line.strip())
        if m:
            t = parse_ts(m.group(0)[1:])
            while idx < len(boundaries) and t >= boundaries[idx]:
                idx += 1
        chunks[idx].append(line)
    return [c for c in chunks if any(l.strip() for l in c)]


# ---------------------------------------------------------------------------
# Distillation
# ---------------------------------------------------------------------------

def _visual_context(art: Artifact, t_lo: float, t_hi: float) -> str:
    parts: list[str] = []
    descs = Artifact.read_json(art.descriptions_json, default=[])
    recon_index = Artifact.read_json(art.reconstructions_index, default=[])
    recon_by_frame = {r["frame"]: r for r in recon_index}
    for d in descs:
        if t_lo <= d["t"] <= t_hi:
            entry = f"[{fmt_ts(d['t'])}] {d['file']}: {d.get('description', '')}"
            if d.get("pointer_target"):
                entry += f" (pointer on: {d['pointer_target']})"
            recon = recon_by_frame.get(d["file"])
            if recon:
                body = (art.root / recon["file"]).read_text()
                entry += f"\n  reconstruction ({recon['file']}):\n{body}"
            parts.append(entry)
    return "\n".join(parts) or "(no visual content in this span)"


def run_distill(art: Artifact, llm: LLM) -> dict:
    segments = art.best_transcript()
    resolved = (art.resolved_md.read_text() if art.resolved_md.exists()
                else "\n".join(f"[{fmt_ts(s['start'])}] {s['text']}"
                               for s in segments))
    resolved_lines = [l for l in resolved.splitlines()
                      if l.strip() and not l.startswith("#")]

    boundaries = find_boundaries(llm, segments)
    line_chunks = split_lines_by_time(resolved_lines, boundaries)

    from .artifact import parse_ts
    ts_prefix = re.compile(r"^\[((\d+:)?\d+:\d+)")

    def span_of(lines: list[str]) -> tuple[float, float]:
        times = [parse_ts(m.group(1)) for l in lines
                 if (m := ts_prefix.match(l.strip()))]
        return (min(times), max(times)) if times else (0.0, 0.0)

    carried_state = "(nothing yet — this is the first chunk)"
    chunks_out: list[dict] = []
    for chunk_id, lines in enumerate(line_chunks, start=1):
        t_lo, t_hi = span_of(lines)
        raw = llm.call(
            [{"role": "user", "content": [
                {"type": "text", "text": prompts.DISTILL_CHUNK.format(
                    carried_state=carried_state,
                    chunk_id=chunk_id,
                    t_start=fmt_ts(t_lo), t_end=fmt_ts(t_hi),
                    chunk_text="\n".join(lines),
                    visual_context=_visual_context(art, t_lo, t_hi))}]}],
            system=prompts.DISTILL_SYSTEM, max_tokens=32000, effort="high")
        chunk_yaml = _parse_yaml(raw)
        chunks_out.append(chunk_yaml)
        new_state = chunk_yaml.get("new_state") or []
        carried = [s for c in chunks_out for s in (c.get("new_state") or [])]
        if carried:
            carried_state = "\n".join(f"- {s}" for s in carried)

    # artifact-level header (title, one-liner, assumed_context)
    header_raw = llm.ask(prompts.DISTILL_HEADER.format(
        distilled=yaml.safe_dump(chunks_out, sort_keys=False,
                                 allow_unicode=True)[:20000]),
        max_tokens=2000)
    header = _parse_yaml(header_raw)

    # mechanical anti-novelty-collapse guard
    transcript_text = " ".join(seg["text"] for seg in segments)
    failures = check_quotes(chunks_out, transcript_text)

    doc = {**header, "quote_check_failures": failures, "chunks": chunks_out}
    art.distilled_yaml.write_text(
        yaml.safe_dump(doc, sort_keys=False, allow_unicode=True, width=100))
    return doc


def _parse_yaml(raw: str) -> dict:
    m = re.search(r"```(?:yaml)?\s*\n(.*?)\n```", raw, re.DOTALL)
    if m:
        raw = m.group(1)
    return yaml.safe_load(raw)
