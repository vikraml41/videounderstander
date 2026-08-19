---
name: video-knowledge
description: >
  Consult the project's video-derived knowledge base (video-knowledge/) when
  implementing or explaining anything the user attributes to a video, talk,
  lecture, tutorial, or screencast — "like in the video", "the approach from
  that talk", "implement what he showed". Also use when a task touches a
  concept a knowledge-base artifact covers, or when the user asks to ingest
  a new video.
---

# Using video-derived knowledge

Videos in this project have been converted into text artifacts under
`video-knowledge/<video-name>/`. Each artifact is a verified distillation of
one video — treat it as the source of truth for what the video actually said.

## Reading order

1. **`distilled.yaml`** first. Skim `title`, `one_line`, and
   `assumed_context`, then the claims. Each claim carries:
   - `mechanism`, `contrast`, `example`, `scope` — use these, not your prior
     on what the technique "usually" means; the `contrast` field says what
     the speaker explicitly ruled out.
   - `status` — weight `demonstrated_on_screen` > `asserted` > `speculated`;
     `cited` claims belong to someone else, not the speaker.
   - `novel: true` — the speaker's idea does NOT map onto the standard
     concept with a similar name. Implement what the claim says verbatim,
     not the nearest familiar pattern.
   - `quote` + `t` — the verbatim transcript evidence and timestamp.
   - `quote_verified: false` — this claim FAILED the mechanical quote check
     and may be a distillation artifact. Verify it against `resolved.md`
     around timestamp `t` before relying on it; say so if you can't.
2. **`resolved.md`** for detail: the full transcript with every "this/that"
   already replaced by its visual referent. Search it by the claim's
   timestamp.
3. **`reconstructions/`** for exact structure: equations as LaTeX, charts as
   data tables, diagrams as Mermaid, code as code. Prefer these over prose
   descriptions. Check `reconstructions/index.json` — a `roundtrip.verdict`
   other than "match" lists known discrepancies with the source frame.
4. **`frames/`** — the original images. Open the frame a claim or
   reconstruction cites (Read the .jpg) whenever the text is ambiguous or
   two artifacts seem to conflict. The image wins.

## Rules

- Preserve the speaker's terminology in code and comments when it names
  their concept (a `novel` term is an API name, not a typo to fix).
- Mind `assumed_context`: short clips are fragments of a larger framework;
  don't implement a fragment as if it were self-contained.
- If the artifact doesn't answer the question, say "not in the video
  artifact" rather than filling the gap from general knowledge silently —
  offer both, labeled.
- `qa/iter_*/report.json` shows what the artifact is known to have lost;
  check the latest report's `holes` before declaring something absent from
  the video.

## Ingesting a new video

The pipeline lives in the videounderstander repo (installed as `vu`):

```bash
vu -a video-knowledge/<short-name> run <video-file> [--transcript subs.vtt]
vu -a video-knowledge/<short-name> verify        # extra iterations if score is low
```

Requires ffmpeg and ANTHROPIC_API_KEY. After ingesting, read the verify
report and tell the user the score and remaining holes. Commit the artifact
directory except `cache/` and `thumbs/`.
