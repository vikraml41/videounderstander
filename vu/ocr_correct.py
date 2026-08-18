"""Step 0b — OCR the kept frames and post-correct the transcript.

On-screen text is ground truth for the spelling of names, symbols, and the
speaker's novel terminology — exactly the vocabulary ASR silently normalizes
to nearest-common-words. One OCR vision call per kept frame (one per
accretion group's final frame would also do; we keep per-frame because the
cache makes re-runs free), then one correction call per transcript batch.
"""

from __future__ import annotations

import json

from .artifact import Artifact
from .llm import LLM, extract_json
from . import prompts

CORRECTION_BATCH = 120  # transcript segments per correction call


def run_ocr(art: Artifact, llm: LLM) -> dict[str, str]:
    manifest = Artifact.read_json(art.frames_manifest)
    results: dict[str, str] = {}
    for fr in manifest:
        text = llm.ask(prompts.OCR, images=[art.root / fr["file"]],
                       max_tokens=4000, effort="low").strip()
        results[fr["file"]] = "" if text == "NO_TEXT" else text
    Artifact.write_json(art.ocr_json, results)
    return results


def _ocr_vocabulary(ocr: dict[str, str], limit_chars: int = 20000) -> str:
    """Deduplicated on-screen lines, most-frequent first, as correction context."""
    counts: dict[str, int] = {}
    for text in ocr.values():
        for line in text.splitlines():
            line = line.strip()
            if len(line) > 2:
                counts[line] = counts.get(line, 0) + 1
    ordered = sorted(counts, key=lambda k: -counts[k])
    out: list[str] = []
    used = 0
    for line in ordered:
        if used + len(line) > limit_chars:
            break
        out.append(line)
        used += len(line) + 1
    return "\n".join(out)


def run_correction(art: Artifact, llm: LLM) -> list[dict]:
    segments = Artifact.read_json(art.transcript_json)
    ocr = Artifact.read_json(art.ocr_json, default={})
    if not ocr:
        # nothing on screen to correct against; corrected == raw
        Artifact.write_json(art.corrected_transcript_json, segments)
        return segments

    vocab = _ocr_vocabulary(ocr)
    corrected = [dict(seg) for seg in segments]
    for batch_start in range(0, len(segments), CORRECTION_BATCH):
        batch = segments[batch_start : batch_start + CORRECTION_BATCH]
        payload = json.dumps(
            [{"i": batch_start + j, "text": seg["text"]}
             for j, seg in enumerate(batch)], ensure_ascii=False)
        raw = llm.ask(prompts.TRANSCRIPT_CORRECTION.format(
            ocr_vocab=vocab, segments=payload), max_tokens=32000)
        for change in extract_json(raw):
            i = int(change["i"])
            if 0 <= i < len(corrected):
                corrected[i]["text"] = change["text"]
                # word timestamps no longer match edited text; drop them for
                # this segment rather than carry wrong alignments
                corrected[i].pop("words", None)
    Artifact.write_json(art.corrected_transcript_json, corrected)
    return corrected
