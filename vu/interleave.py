"""Step 2 — interleave frames and transcript by timestamp.

The highest-leverage move in the pipeline: a frame reference is emitted at
the moment the frame's visual state first appears, immediately before the
speech uttered while it was on screen, so "this curve" sits adjacent to the
pixels it refers to. Never a transcript block followed by an image block.

Output is Markdown (aligned.md). Frame references use the form
`![t](frames/frame_xxxxxx.jpg)` so the file renders, plus the accretion
group marker when a diagram builds across several frames.
"""

from __future__ import annotations

from .artifact import Artifact, fmt_ts


def build_events(manifest: list[dict], segments: list[dict]) -> list[dict]:
    """Merge frames and speech into one time-ordered event list.

    Frames sort by the start of their stable interval (when the state
    appeared); speech by segment start. On ties the frame comes first, so
    speech about a visual follows the visual.
    """
    events: list[dict] = []
    for fr in manifest:
        events.append({"kind": "frame", "t": fr["interval_start"], "frame": fr})
    for seg in segments:
        events.append({"kind": "speech", "t": seg["start"], "seg": seg})
    events.sort(key=lambda e: (e["t"], 0 if e["kind"] == "frame" else 1))
    return events


def _frame_line(fr: dict) -> str:
    # label with the span the state was on screen; the file is its final frame
    label = f"[{fmt_ts(fr['interval_start'])}–{fr['ts']}]"
    line = f"{label} ![frame at {fr['ts']}]({fr['file']})"
    if fr.get("entered_by") == "build":
        line += f"  <!-- build step, accretion group {fr['group']} -->"
    return line


def render_aligned(manifest: list[dict], segments: list[dict],
                   descriptions: dict[str, dict] | None = None) -> str:
    """Render the interleaved context document.

    `descriptions` (optional, from step 3) maps frame file -> description
    dict; when present, each frame line is followed by its description and
    pointer target, which is what the deixis pass consumes.
    """
    lines = ["# Aligned multimodal context", ""]
    for ev in build_events(manifest, segments):
        if ev["kind"] == "frame":
            fr = ev["frame"]
            lines.append(_frame_line(fr))
            desc = (descriptions or {}).get(fr["file"])
            if desc:
                lines.append(f"    frame content: {desc.get('description', '')}")
                if desc.get("pointer_target"):
                    lines.append(f"    pointer on: {desc['pointer_target']}")
        else:
            seg = ev["seg"]
            span = f"[{fmt_ts(seg['start'])}–{fmt_ts(seg['end'])}]"
            lines.append(f'{span} "{seg["text"]}"')
    lines.append("")
    return "\n".join(lines)


def run_align(art: Artifact) -> str:
    manifest = Artifact.read_json(art.frames_manifest)
    segments = art.best_transcript()
    descriptions = None
    if art.descriptions_json.exists():
        descriptions = {d["file"]: d
                        for d in Artifact.read_json(art.descriptions_json)}
    text = render_aligned(manifest, segments, descriptions)
    art.aligned_md.write_text(text)
    return text
