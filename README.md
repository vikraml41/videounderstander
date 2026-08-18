# videounderstander

Turn a video into a **verified, text-first artifact** that Claude (or Claude
Code) can consume without re-feeding the video on every query — with
information loss measured as a number, not a feeling.

The design rationale lives in [PLAN.md](PLAN.md). The short version: naive
transcript-plus-screenshots loses deixis ("this curve here"), the order of
accreting visuals, emphasis structure, and — worst — silently normalizes
novel ideas into familiar ones. Each stage below targets one of those losses,
and the verification loop at the end measures what actually survived.

## Install

```bash
pip install -e .            # needs Python 3.10+, plus ffmpeg on PATH
pip install -e '.[asr]'     # optional: local ASR via faster-whisper
export ANTHROPIC_API_KEY=…  # or `ant auth login`
```

LLM/vision calls default to `claude-opus-5` (override with `--model` or
`VU_MODEL`).

## Usage

```bash
# everything, end to end:
vu -a artifact/ run lecture.mp4 --transcript lecture.vtt   # or omit --transcript to run ASR

# or stage by stage (each is idempotent; LLM calls are disk-cached):
vu -a artifact/ sample lecture.mp4        # 1  dense sampling + tile-grid dedup
vu -a artifact/ transcribe --from l.vtt   # 0a transcript ingest (or --video for ASR)
vu -a artifact/ ocr                       # 0b OCR kept frames…
vu -a artifact/ correct                   #    …and fix ASR terminology against it
vu -a artifact/ describe                  # 3  context-aware frame descriptions + pointer targets
vu -a artifact/ align                     # 2  interleave frames + speech by timestamp
vu -a artifact/ reconstruct               # 4  LaTeX/Mermaid/tables/code + round-trip check
vu -a artifact/ resolve                   # 5  deixis resolution
vu -a artifact/ distill                   # 6  schema distillation + verbatim-quote check
vu -a artifact/ verify                    # 7  comprehension-question loop
vu -a artifact/ status
```

`verify` prints a score (`14/18`) and a hole list — each miss names the
specific information the artifact lacks and roughly where in the video it
lives. Patch the artifact (or re-run the stage that dropped it), run
`verify` again; iterations land in `artifact/qa/iter_NN/`. Two or three
rounds usually converge.

## The artifact

```
artifact/
  source.json                 metadata, assumed_context
  transcript.json             normalized segments (word timestamps kept)
  transcript.corrected.json   after OCR-based ASR correction
  frames/                     kept full-res frames + manifest.json
  ocr.json                    on-screen text per frame
  descriptions.json           step-3 output (incl. pointer targets)
  aligned.md                  interleaved frames + speech
  reconstructions/            .tex / .mmd / .md re-encodings + round-trip verdicts
  resolved.md                 deixis-resolved transcript  ← main reading surface
  distilled.yaml              claims with mechanism/contrast/example/scope/
                              epistemic status/verbatim quote + timestamp
  qa/iter_NN/                 verification questions, answers, grades, report
  cache/                      content-addressed LLM response cache
```

Point Claude Code at `distilled.yaml` and `resolved.md`; every claim carries
timestamps and frame files, so it can open the original image whenever a
reconstruction is ambiguous.

### The mechanical guards

- **Verbatim-quote check** (`distill`): every claim must carry a quote that
  string-matches the transcript (normalized, with a fuzzy fallback for ASR
  noise). A claim the model normalized away from the speaker's words fails
  loudly — `quote_verified: false` in `distilled.yaml`.
- **Round-trip check** (`reconstruct`): each reconstruction is compared
  against its source frame by a vision call; mismatches get one repair round
  and are flagged in `reconstructions/index.json` if still off.
- **Reverse check** (`verify`): sampled claims are checked against the
  source transcript to catch hallucinated additions, not just omissions.

## Development

Pure-stdlib core (sampling math, subtitle parsing, interleaving, quote
matching) is unit-tested without ffmpeg or an API key:

```bash
python -m unittest discover -s tests
```
