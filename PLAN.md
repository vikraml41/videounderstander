# Video Understanding Pipeline — Revised Plan

Goal: convert a video into a text-first artifact that Claude Code can consume without
re-feeding the video, with measurable (not felt) information loss.

## What actually gets lost

Each loss has a different fix:

- **Deixis.** "This line here," "that spike." A transcript is full of pronouns with no
  referents. Largest source of silent degradation — the text reads fine, it just doesn't
  mean anything.
- **Accreting visuals.** Whiteboard derivations and built-up slides mean something
  different at each step. A single final screenshot loses the order of construction,
  which is where the intuition lives.
- **Emphasis structure.** Prosody and repetition mark what's load-bearing; flat text
  makes the aside and the central claim look identical.
- **Novelty collapse.** A distillation model meeting an unfamiliar idea pattern-matches
  it to the nearest familiar concept and writes *that* down. Well-written summary of
  something the video didn't say; undetectable downstream.
- **ASR terminology loss** *(added in audit)*. Speech recognition "corrects" the
  speaker's idiosyncratic or novel terms to nearest-common-words — novelty collapse
  before distillation even runs. The transcript is not ground truth.

## The pipeline

### 0. Transcript hardening *(new)*

- ASR with **word-level timestamps** (or forced alignment after). Segment-level
  timestamps drift seconds; the interleave in step 2 depends on tight alignment.
- **OCR the kept frames and post-correct the transcript against on-screen text.**
  On-screen spelling of terms, names, and symbols is ground truth; use it to fix
  ASR's normalization of novel vocabulary.
- Diarize if multi-speaker.

### 1. Dense sampling, structural deduplication

- Sample at 1 fps (short/rapid-cut videos: 2–4 fps; cost is trivial).
- **Dedupe with a tile-grid pixel diff, not perceptual hash alone.** pHash fails both
  ways on this content: a new bullet on a dense slide flips too few bits (build step
  silently deduped away); camera wobble flips too many (never stabilizes). Grid diff
  gives locality: small localized change = build step, keep it; global change =
  cut/wobble. pHash is fine as a cheap prefilter.
- For each stable interval keep the **final** frame (the completed build step). Where a
  diagram builds continuously, keep the whole sequence and mark it as one build.
- Keep frames at **full resolution** — they feed reconstruction (step 4).

### 2. Interleave by timestamp — never concatenate

Highest-leverage move. Hand the model:

```
[04:12] <frame_0412.jpg>
[04:12–04:31] "so if you look at this curve, the interesting part
                isn't the peak, it's how fast it decays after"
[04:31] <frame_0431.jpg>
```

Attention can bind "this curve" to pixels only if they're adjacent in the sequence.
A frame is placed at the timestamp it first *appears*, so the speech that follows it
refers to it.

### 3. Describe frames with transcript context — and locate the pointer

- Pass the surrounding ±30s of speech into the vision call; ask what's relevant *to
  what's being said*, not "describe this image."
- **Explicitly ask: where is the cursor / selection / laser pointer / speaker's hand,
  and what is it on?** In screencasts the cursor is the deixis anchor; this output is
  what makes step 5 resolution grounded rather than salience-guessing.

### 4. Reconstruct visuals rather than describing them

- LaTeX for equations, extracted-point data tables for charts, Mermaid/SVG for
  diagrams, actual code for code. Prose description of structure is unrecoverably lossy.
- Keep both the reconstruction and the original frame in the artifact.
- **Round-trip check** *(new)*: render the reconstruction back to an image and run a
  vision comparison against the original frame ("same structure? same numbers?").
  Run or lint reconstructed code. Reconstruction is only verifiable if you verify it.

### 5. Resolve deixis as an explicit pass

Rewrite the aligned transcript with every demonstrative replaced by its referent, drawn
from step-3 outputs (including pointer targets):
`"the interesting part isn't the peak"` → `"the interesting part of [the exponential
decay curve, x-axis = days since event, plotted from the 2019 sample] isn't the peak."`
Ugly, dramatically more useful.

### 6. Distill against a schema that forces out what summaries drop

Not "summarize" — fill in, per claim:

- The claim, one sentence, in the speaker's own vocabulary
- The **mechanism** — why would this be true
- What it's **contrasted against** — catches novelty collapse better than anything else
- The worked example, with the actual numbers
- Scope conditions and stated failure modes
- **Epistemic status**: asserted / demonstrated on screen / speculated / cited to someone
- **A verbatim supporting quote + timestamp** *(new)*. Mechanically checkable: a script
  string-matches every quote against the transcript. A claim the model normalized away
  from the speaker's words won't have a matching quote — the worst silent failure
  becomes a loud one.
- **Corrections/retractions** *(new)*: things the speaker said then fixed. Naive
  distillation keeps the first, wrong version.
- Frame-file citations, so downstream Claude can open the original image when a
  reconstruction is ambiguous.

Plus a glossary of the speaker's idiosyncratic term definitions, and the hard prompt
constraint: preserve exact terminology and all numbers verbatim; if a claim doesn't fit
a familiar frame, record it in the speaker's words and flag it.

## Verification: the part that actually minimizes loss

Generate 15–20 comprehension questions from the full multimodal source — including
visual-only questions. A fresh model with no video access answers them from the
artifact alone. Diff against ground truth; every miss is a specific hole; patch,
re-run. Two or three iterations usually converges.

**Circularity fix** *(new)* — the question generator shares the distiller's blindness,
so the one non-obvious move can fail to become a question and the loop converges with
the hole intact:

- **Coverage requirement**: every schema field of every chunk must be probed by ≥1
  question. Nothing skipped by salience.
- **Reverse direction**: also check every artifact claim against the source for
  unsupported additions — catch hallucination, not just omission.
- **Forced question types** the generator wouldn't volunteer: exact numbers, contrasts,
  "what was drawn between X and Y," what got retracted.

## Structural notes

- **Long videos need carried state.** Chunk on argument boundaries (discourse markers +
  slide transitions), passing a running definitions-and-established-claims block into
  each chunk, so a minute-40 callback to a minute-3 term resolves.
- **Reels are fragments** — one move extracted from a framework the creator assumes.
  Carry an explicit `assumed_context` field; a claim with its preconditions stripped is
  worse than no claim.
- **Gemini native ingestion** can replace steps 1–3 as a first pass (it solves
  alignment), but it samples ~1 fps at reduced resolution — dense slide text and code
  can be illegible to it. Step 4 always runs on full-resolution extracted frames, and
  the verification loop runs regardless. The artifact is still the deliverable; never
  re-feed video per query.
- **Cache by frame hash.** Vision calls are the cost center; verification iterations
  must not re-bill the whole video.

## Artifact layout (Claude Code is the consumer)

```
artifact/
  source.md            # metadata: URL, duration, speakers, assumed_context
  transcript.vtt       # corrected, word-timestamped
  aligned.md           # interleaved frames + transcript (step 2 output)
  resolved.md          # deixis-resolved transcript (step 5 output)
  frames/              # kept full-res frames, named by timestamp
  reconstructions/     # .tex / .mmd / .csv / code, named to match frames
  distilled.yaml       # step 6 schema output, claims cite quotes + frames
  qa/                  # verification questions, ground truth, diffs per iteration
```

## Build order

1. Sampling + tile-diff dedup script
2. ASR + OCR correction + alignment
3. Interleaved-context builder
4. Frame description / reconstruction / deixis passes
5. Distillation schema + prompt
6. Verification harness (question gen, blind answering, diff, coverage check,
   quote-match check, reconstruction round-trip)
