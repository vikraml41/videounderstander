"""Step 7 — the verification harness: turn "did I lose understanding" into a
number, and locate the loss.

  1. Generate comprehension questions from the FULL multimodal source
     (interleaved transcript + actual frame images), with coverage forced
     across chunks and schema dimensions, plus the question types generators
     skip (exact numbers, contrasts, retractions, order of construction).
  2. A fresh call answers them from the artifact text ALONE (distilled.yaml
     + resolved.md) — no video access, "NOT IN ARTIFACT" when missing.
  3. Grade against ground truth; every miss is a specific hole.
  4. Reverse check: sample the artifact's claims against the transcript to
     catch hallucinated/normalized claims, not just omissions.

Each run writes qa/iter_NN/{questions,answers,grades,reverse,report}.json.
Patch the artifact where report.holes points, re-run; convergence is
typically 2–3 iterations.
"""

from __future__ import annotations

import json

from .artifact import Artifact, fmt_ts
from .llm import LLM, extract_json, image_block, text_block
from . import prompts

MAX_SOURCE_IMAGES = 40
DEFAULT_N_QUESTIONS = 18


def _source_content(art: Artifact) -> list[dict]:
    """Interleaved multimodal source as API content blocks: speech text with
    real frame images inline, capped at MAX_SOURCE_IMAGES (visual-heavy
    frames kept preferentially)."""
    manifest = Artifact.read_json(art.frames_manifest)
    segments = art.best_transcript()
    descs = {d["file"]: d
             for d in Artifact.read_json(art.descriptions_json, default=[])}

    frames = sorted(manifest, key=lambda f: f["interval_start"])
    if len(frames) > MAX_SOURCE_IMAGES:
        # keep every reconstruction-worthy frame first, then sample the rest evenly
        rich = [f for f in frames
                if descs.get(f["file"], {}).get("needs_reconstruction")]
        rest = [f for f in frames if f not in rich]
        budget = MAX_SOURCE_IMAGES - min(len(rich), MAX_SOURCE_IMAGES)
        stride = max(1, len(rest) // budget) if budget else len(rest) + 1
        frames = sorted(rich[:MAX_SOURCE_IMAGES] + rest[::stride],
                        key=lambda f: f["interval_start"])[:MAX_SOURCE_IMAGES]

    events: list[tuple[float, int, dict]] = []
    for fr in frames:
        events.append((fr["interval_start"], 0, fr))
    for seg in segments:
        events.append((seg["start"], 1, seg))
    events.sort(key=lambda e: (e[0], e[1]))

    content: list[dict] = []
    text_buf: list[str] = []
    for _t, kind, obj in events:
        if kind == 0:
            if text_buf:
                content.append(text_block("\n".join(text_buf)))
                text_buf = []
            content.append(text_block(f"[{obj['ts']}] frame {obj['file']}:"))
            content.append(image_block(art.root / obj["file"]))
        else:
            text_buf.append(
                f"[{fmt_ts(obj['start'])}–{fmt_ts(obj['end'])}] \"{obj['text']}\"")
    if text_buf:
        content.append(text_block("\n".join(text_buf)))
    return content


def artifact_text(art: Artifact) -> str:
    """What a downstream reader actually gets: distilled.yaml + resolved.md."""
    parts = []
    if art.distilled_yaml.exists():
        parts.append("## distilled.yaml\n\n" + art.distilled_yaml.read_text())
    if art.resolved_md.exists():
        parts.append("## resolved.md\n\n" + art.resolved_md.read_text())
    if not parts:
        raise FileNotFoundError("no distilled.yaml or resolved.md — run "
                                "distill (and resolve) before verify")
    return "\n\n".join(parts)


def _next_iter_dir(art: Artifact) -> "Path":
    from pathlib import Path
    n = 1
    while (art.qa_dir / f"iter_{n:02d}").exists():
        n += 1
    d = art.qa_dir / f"iter_{n:02d}"
    d.mkdir(parents=True)
    return d


def run_verify(art: Artifact, llm: LLM,
               n_questions: int = DEFAULT_N_QUESTIONS) -> dict:
    it_dir = _next_iter_dir(art)

    # 1. questions from the full multimodal source
    content = _source_content(art)
    content.append(text_block(
        prompts.QUESTION_GEN.format(source="(interleaved above)",
                                    n=n_questions)))
    questions = extract_json(llm.call(
        [{"role": "user", "content": content}], max_tokens=16000,
        effort="high"))
    Artifact.write_json(it_dir / "questions.json", questions)

    # 2. blind answering from the artifact alone (fresh call, no images)
    art_text = artifact_text(art)
    answers = extract_json(llm.ask(
        prompts.ANSWER_FROM_ARTIFACT.format(
            artifact=art_text,
            questions=json.dumps(
                [{"i": i, "q": q["q"]} for i, q in enumerate(questions)],
                ensure_ascii=False)),
        max_tokens=16000))
    answer_by_i = {int(a["i"]): a["answer"] for a in answers}
    Artifact.write_json(it_dir / "answers.json", answers)

    # 3. grading
    rows = json.dumps(
        [{"i": i, "question": q["q"], "ground_truth": q["ground_truth"],
          "artifact_answer": answer_by_i.get(i, "(no answer)")}
         for i, q in enumerate(questions)], ensure_ascii=False, indent=1)
    grades = extract_json(llm.ask(
        prompts.GRADE_ANSWERS.format(rows=rows), max_tokens=16000,
        effort="high"))
    Artifact.write_json(it_dir / "grades.json", grades)

    # 4. reverse check: artifact claims vs source transcript
    reverse = _reverse_check(art, llm)
    Artifact.write_json(it_dir / "reverse.json", reverse)

    # 5. report
    verdicts = {int(g["i"]): g for g in grades}
    n = len(questions)
    correct = sum(1 for g in grades if g["verdict"] == "correct")
    holes = []
    for i, q in enumerate(questions):
        g = verdicts.get(i)
        if g and g["verdict"] != "correct":
            holes.append({"i": i, "verdict": g["verdict"],
                          "kind": q.get("kind"), "probes": q.get("probes"),
                          "t": q.get("t"), "question": q["q"],
                          "hole": g.get("hole", "")})
    unsupported = [r for r in reverse if not r.get("supported", True)]
    report = {
        "iteration": it_dir.name,
        "score": f"{correct}/{n}",
        "correct": correct, "total": n,
        "holes": holes,
        "unsupported_claims": unsupported,
    }
    Artifact.write_json(it_dir / "report.json", report)
    return report


def _reverse_check(art: Artifact, llm: LLM) -> list[dict]:
    if not art.distilled_yaml.exists():
        return []
    import yaml
    doc = yaml.safe_load(art.distilled_yaml.read_text())
    claims = [{"i": i, "claim": c.get("claim"), "quote": c.get("quote")}
              for i, c in enumerate(
                  c for ch in doc.get("chunks", [])
                  for c in (ch.get("claims") or []))]
    if not claims:
        return []
    segments = art.best_transcript()
    transcript = "\n".join(
        f"[{fmt_ts(s['start'])}] {s['text']}" for s in segments)
    return extract_json(llm.ask(
        prompts.REVERSE_CHECK.format(
            transcript=transcript,
            claims=json.dumps(claims, ensure_ascii=False)),
        max_tokens=16000, effort="high"))
