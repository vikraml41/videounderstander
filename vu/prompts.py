"""Every prompt template in the pipeline, in one place.

Conventions:
- Prompts that need machine-readable output ask for JSON and are parsed with
  llm.extract_json; the schemas are stated inline in the prompt.
- Anti-novelty-collapse constraints (verbatim terminology, no normalizing of
  unfamiliar claims) are repeated wherever the model rewrites speech.
"""

OCR = """Transcribe ALL legible on-screen text in this frame, exactly as \
written — code, equations, labels, slide text, terminal output, titles. \
Preserve original spelling, casing, and symbols precisely; do not correct \
apparent typos. If a region is too small or blurry to read with confidence, \
mark it [illegible] rather than guessing. Output plain text only, roughly in \
reading order. If there is no text, output exactly: NO_TEXT"""

TRANSCRIPT_CORRECTION = """You are correcting speech-recognition errors in a \
video transcript using the video's on-screen text as ground truth for \
spelling and terminology.

ON-SCREEN TEXT (OCR from the video's frames — authoritative for names, \
technical terms, symbols, and novel vocabulary):
{ocr_vocab}

TRANSCRIPT SEGMENTS (JSON array of {{"i": index, "text": ...}}):
{segments}

Fix ONLY clear ASR errors: misheard technical terms, names, and numbers where \
the on-screen text shows the intended form (e.g. ASR wrote "cage entropy" but \
slides show "K-entropy"). Do NOT paraphrase, do NOT fix grammar or filler \
words, do NOT change anything you are not confident is an ASR error. Keep \
every segment, same order, same index.

Output a JSON array: [{{"i": index, "text": corrected_text}}, ...] containing \
ONLY the segments you changed (empty array if none)."""

FRAME_DESCRIPTION = """This frame is from a video at {ts}. The speaker is \
saying (within ±30s of this frame):

"{speech}"

Describe what in this frame is relevant TO WHAT IS BEING SAID — not a generic \
scene description. Then answer the structured fields.

CRITICAL — pointer target: if there is a mouse cursor, text selection, \
highlight, laser-pointer dot, or the speaker's hand/pen visible, state \
exactly what it is on or pointing at. In screencasts this is what phrases \
like "this line here" refer to; report it even if it seems incidental.

Output JSON:
{{
  "description": "what's shown, framed by its relevance to the speech",
  "pointer_target": "what the cursor/highlight/hand is on, or null",
  "content_type": "slide|chart|code|diagram|equation|whiteboard|terminal|talking_head|demo|other",
  "needs_reconstruction": true/false,  // true for charts, math, code, diagrams, tables — anything whose structure prose can't carry
  "on_screen_text_gist": "one line: the key text/labels visible, or null"
}}"""

RECONSTRUCTION = """Re-encode the structural content of this frame in a \
verifiable format. The speaker context: "{speech}"

Frame content type: {content_type}

Choose the format that loses nothing:
- equations -> LaTeX
- charts/plots -> a Markdown table of extracted data points (read actual \
values off the axes; estimate honestly and mark estimates with ~) plus axis \
labels, units, and scale (linear/log)
- diagrams/flowcharts/architecture -> Mermaid
- code -> the code itself, verbatim, in a fenced block with language
- tables -> Markdown table

Rules: transcribe text and numbers EXACTLY as shown; mark anything illegible \
as [illegible] rather than inventing it; if the frame mixes types, emit each \
part under a short heading. Output only the reconstruction (Markdown), no \
commentary."""

ROUNDTRIP_CHECK = """You are verifying a reconstruction against its source \
frame. The image is the original video frame; below is a reconstruction that \
claims to encode its structural content:

---
{reconstruction}
---

Compare them strictly. Do the structure, labels, numbers, and relationships \
in the reconstruction match the frame? Output JSON:
{{
  "verdict": "match|minor_issues|mismatch",
  "discrepancies": ["specific difference 1", ...]  // empty if match
}}"""

DEIXIS_RESOLUTION = """Rewrite this transcript excerpt with every deictic \
reference made explicit. The excerpt is interleaved with the frames the \
speaker was showing and descriptions of them (including what the cursor or \
pointer was on).

{context}

Rules:
- Replace demonstratives whose referent is visual ("this curve", "that line \
there", "as you can see", "over here") with the referent in square brackets, \
drawn from the frame content: "the peak" -> "the peak of [the exponential \
decay curve, x-axis = days since event]".
- When the pointer target is known, it is the referent for "this/that ... \
here".
- Change NOTHING else: keep wording, order, filler, and all timestamps \
exactly. Do not summarize, do not fix grammar, do not drop sentences.
- If a referent cannot be determined from the frames, leave the phrase as-is \
and append [referent unclear].

Output the rewritten transcript lines only, keeping the [MM:SS] prefixes."""

CHUNK_BOUNDARIES = """Below is a timestamped video transcript. Split it into \
chunks at ARGUMENT boundaries — where the speaker completes one line of \
reasoning or topic and starts another (transitions like "okay, so now", a new \
slide topic, a new worked example). Aim for chunks of roughly {target_min}–\
{target_max} minutes; never split mid-derivation.

{transcript}

Output a JSON array of boundary timestamps in seconds (floats), ascending, \
excluding 0 and the end. Empty array if the video is one continuous argument."""

DISTILL_SYSTEM = """You distill technical video content into a structured \
artifact. Hard constraints, in priority order:

1. PRESERVE the speaker's exact terminology and ALL numbers verbatim. Never \
substitute a standard term for the speaker's idiosyncratic one.
2. If a claim doesn't fit a familiar frame, record it in the speaker's own \
words and set "novel": true — do NOT normalize it to the nearest familiar \
concept. A weird claim faithfully recorded beats a familiar claim the video \
didn't make.
3. Every claim MUST carry a "quote": a verbatim supporting excerpt copied \
character-for-character from the transcript (it will be mechanically checked \
against the transcript — paraphrases fail the check), and "t": its timestamp \
in seconds.
4. Record what the speaker CONTRASTS each idea against — concepts are defined \
by what they exclude.
5. Distinguish epistemic status per claim: asserted | demonstrated_on_screen \
| speculated | cited (attributed to someone else).
6. Capture corrections: if the speaker states something then retracts or \
fixes it, record only the corrected version and note the retraction."""

DISTILL_CHUNK = """CARRIED STATE from earlier in the video (definitions and \
claims already established — use these to resolve callbacks, do not restate \
them as new claims):
{carried_state}

CHUNK {chunk_id} of the video, {t_start}–{t_end}, deixis-resolved transcript \
with frame references:

{chunk_text}

Frame descriptions and reconstructions for this chunk:
{visual_context}

Fill in this schema for THIS CHUNK. Output YAML only:

chunk: {chunk_id}
span: "{t_start}–{t_end}"
claims:
  - claim: one sentence, speaker's vocabulary
    mechanism: why would this be true — the generative story given (or null)
    contrast: what it is contrasted against / what it excludes (or null)
    example: the worked example with the actual numbers used (or null)
    scope: stated scope conditions and failure modes (or null)
    status: asserted|demonstrated_on_screen|speculated|cited
    novel: true if this doesn't fit a familiar frame
    corrected: note if the speaker retracted/fixed an earlier version (or null)
    quote: "verbatim transcript excerpt supporting this claim"
    t: timestamp_seconds
    frames: [frame files that show it, e.g. frames/frame_000412.jpg]
glossary:
  - term: speaker's idiosyncratic term
    definition: as the speaker defines it
    t: timestamp_seconds
new_state:  # additions to carried state for later chunks
  - one line per definition or established claim later chunks may reference"""

DISTILL_HEADER = """Also produce the artifact-level header. Output YAML only:

title: best title for this video's content
one_line: what this video is about, one sentence
assumed_context: >
  what the speaker assumes the viewer already knows or has seen — the
  framework this content is extracted from. Critical for short clips/reels:
  a claim with its preconditions stripped is worse than no claim. null if
  genuinely self-contained.

Base it on this distilled content:
{distilled}"""

QUESTION_GEN = """You are generating a comprehension test for a video, from \
its FULL multimodal source (interleaved transcript + frames below). The test \
will be answered by a model that can only see a text artifact derived from \
the video — the diff measures what the artifact lost.

{source}

Generate {n} questions with ground-truth answers. Requirements:
- Cover EVERY chunk and probe every schema dimension somewhere: mechanisms, \
contrasts, worked examples (exact numbers), scope conditions, epistemic \
status, glossary terms.
- Include questions answerable ONLY from visual content (chart values, \
diagram structure, code shown, what was drawn between two moments).
- Include the forced types generators skip: exact numbers, what X was \
contrasted with, what the speaker corrected or retracted, order of \
construction in built-up visuals.
- At least one question about the single most non-obvious move in the video \
— the thing a good-sounding summary would silently drop.

Output JSON:
[{{"q": "...", "ground_truth": "...", "kind": "verbal|visual|both",
   "probes": "claim|mechanism|contrast|example|scope|status|glossary|construction",
   "t": approx_timestamp_seconds}}, ...]"""

ANSWER_FROM_ARTIFACT = """Answer the questions below using ONLY the artifact \
text provided — you have no access to the video it describes. If the \
artifact does not contain the answer, say exactly "NOT IN ARTIFACT" for that \
question; do not guess from general knowledge.

ARTIFACT:
{artifact}

QUESTIONS (JSON):
{questions}

Output JSON: [{{"i": question_index, "answer": "..."}}, ...]"""

GRADE_ANSWERS = """Grade artifact-derived answers against ground truth from \
the original video. Be strict on numbers, terminology, and directionality; \
lenient on phrasing.

{rows}

Output JSON:
[{{"i": question_index, "verdict": "correct|partial|wrong|missing",
   "hole": "if not correct: what specific information the artifact lacks or \
misstates, one line"}}, ...]"""

REVERSE_CHECK = """Below are claims from a distilled artifact, and the \
source transcript of the video it was distilled from. For each claim, check \
whether the SOURCE actually supports it — this catches hallucinated or \
normalized claims, not just omissions.

SOURCE TRANSCRIPT:
{transcript}

CLAIMS (JSON):
{claims}

Output JSON: [{{"i": claim_index, "supported": true/false,
  "note": "if unsupported: what the source actually says instead"}}, ...]"""
