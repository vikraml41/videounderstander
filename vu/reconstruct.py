"""Step 4 — reconstruct structural visuals instead of describing them.

For frames flagged `needs_reconstruction` in step 3 (charts, math, code,
diagrams, tables), re-encode the content in a verifiable format, then run a
round-trip check: a second vision call compares the reconstruction against
the original frame and reports discrepancies. Reconstructions with a
"mismatch" verdict are regenerated once with the discrepancies fed back.

Both the reconstruction and the original frame stay in the artifact, so a
downstream reader can open the image when the reconstruction is ambiguous.
"""

from __future__ import annotations

from .artifact import Artifact
from .llm import LLM, extract_json
from .transcribe import transcript_window
from . import prompts

_EXT = {"equation": ".tex", "chart": ".md", "diagram": ".mmd",
        "code": ".md", "slide": ".md", "whiteboard": ".md",
        "terminal": ".md", "table": ".md"}


def _roundtrip(llm: LLM, frame_path, reconstruction: str) -> dict:
    raw = llm.ask(prompts.ROUNDTRIP_CHECK.format(reconstruction=reconstruction),
                  images=[frame_path], max_tokens=4000)
    return extract_json(raw)


def run_reconstruct(art: Artifact, llm: LLM) -> list[dict]:
    descriptions = Artifact.read_json(art.descriptions_json)
    segments = art.best_transcript()
    art.reconstructions_dir.mkdir(parents=True, exist_ok=True)

    index: list[dict] = []
    for desc in descriptions:
        if not desc.get("needs_reconstruction"):
            continue
        frame_path = art.root / desc["file"]
        speech = transcript_window(segments, desc["t"]) or "(none)"
        content_type = desc.get("content_type", "other")

        reconstruction = llm.ask(
            prompts.RECONSTRUCTION.format(speech=speech,
                                          content_type=content_type),
            images=[frame_path], max_tokens=16000)
        check = _roundtrip(llm, frame_path, reconstruction)

        if check.get("verdict") == "mismatch":
            # one repair round with the discrepancies fed back
            issues = "\n".join(f"- {d}" for d in check.get("discrepancies", []))
            reconstruction = llm.ask(
                prompts.RECONSTRUCTION.format(speech=speech,
                                              content_type=content_type)
                + "\n\nA previous attempt had these verified discrepancies "
                  "with the frame — fix them:\n" + issues,
                images=[frame_path], max_tokens=16000)
            check = _roundtrip(llm, frame_path, reconstruction)

        stem = frame_path.stem  # frame_000412
        ext = _EXT.get(content_type, ".md")
        out_name = f"{stem}{ext}"
        (art.reconstructions_dir / out_name).write_text(reconstruction + "\n")
        index.append({
            "frame": desc["file"],
            "t": desc["t"],
            "content_type": content_type,
            "file": f"reconstructions/{out_name}",
            "roundtrip": check,
        })

    Artifact.write_json(art.reconstructions_index, index)
    return index
