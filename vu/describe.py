"""Step 3 — context-aware frame descriptions.

One vision call per kept frame, carrying the ±30s of surrounding speech, so
the output says what's relevant *to what's being said* — and, critically,
where the cursor/pointer/highlight is, which is the deixis anchor in
screencasts.
"""

from __future__ import annotations

from .artifact import Artifact, fmt_ts
from .llm import LLM, extract_json
from .transcribe import transcript_window
from . import prompts


def run_describe(art: Artifact, llm: LLM) -> list[dict]:
    manifest = Artifact.read_json(art.frames_manifest)
    segments = art.best_transcript()
    results: list[dict] = []
    for fr in manifest:
        speech = transcript_window(segments, fr["t"]) or "(no speech near this frame)"
        raw = llm.ask(
            prompts.FRAME_DESCRIPTION.format(ts=f"[{fr['ts']}]", speech=speech),
            images=[art.root / fr["file"]], max_tokens=4000)
        desc = extract_json(raw)
        desc["file"] = fr["file"]
        desc["t"] = fr["t"]
        results.append(desc)
    Artifact.write_json(art.descriptions_json, results)
    return results
