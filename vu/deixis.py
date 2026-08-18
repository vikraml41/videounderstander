"""Step 5 — explicit deixis-resolution pass.

Rewrites the transcript with every visual demonstrative replaced by its
referent (from step-3 descriptions and pointer targets). The output,
resolved.md, is the text later stages and downstream readers consume: ugly
to read, dramatically more useful.
"""

from __future__ import annotations

from .artifact import Artifact
from .interleave import render_aligned
from .llm import LLM
from . import prompts

WINDOW_SEGMENTS = 40  # transcript segments per resolution call


def run_resolve(art: Artifact, llm: LLM) -> str:
    manifest = Artifact.read_json(art.frames_manifest)
    segments = art.best_transcript()
    descriptions = {d["file"]: d
                    for d in Artifact.read_json(art.descriptions_json)}

    out_parts: list[str] = []
    for start in range(0, len(segments), WINDOW_SEGMENTS):
        window = segments[start : start + WINDOW_SEGMENTS]
        t_lo = window[0]["start"]
        t_hi = window[-1]["end"]
        # frames whose stable interval overlaps this window
        frames = [fr for fr in manifest
                  if fr["interval_start"] <= t_hi and fr["t"] >= t_lo - 60]
        context = render_aligned(frames, window, descriptions)
        resolved = llm.ask(
            prompts.DEIXIS_RESOLUTION.format(context=context),
            max_tokens=32000)
        out_parts.append(resolved.strip())

    text = "# Deixis-resolved transcript\n\n" + "\n".join(out_parts) + "\n"
    art.resolved_md.write_text(text)
    return text
