# Wiring video knowledge into a Claude Code project

Three pieces, all copy-paste:

## 1. Install the pipeline (once, on your machine)

```bash
git clone https://github.com/vikraml41/videounderstander
cd videounderstander && pip install -e .
# plus: ffmpeg on PATH, and ANTHROPIC_API_KEY set (or `ant auth login`)
```

## 2. Ingest videos into your coding project

Artifacts live *inside the project the agent works on*, so they're on disk
when it needs them:

```bash
cd ~/my-coding-project
mkdir -p video-knowledge
vu -a video-knowledge/attention-talk run ~/Videos/attention-talk.mp4
```

Tip: for YouTube videos, `yt-dlp --write-auto-subs` gets you the video plus
a .vtt to pass as `--transcript` (skips local ASR).

Check the verify score it prints; if there are holes, run
`vu -a video-knowledge/attention-talk verify` after patching, until it
converges. Commit the artifact, minus the disposable dirs:

```gitignore
video-knowledge/*/cache/
video-knowledge/*/thumbs/
```

## 3. Teach the agent about it

Copy the skill into the project:

```bash
mkdir -p .claude/skills
cp -r /path/to/videounderstander/integration/video-knowledge .claude/skills/
```

And add one pointer to your project's `CLAUDE.md` so the knowledge base is
discoverable even when the skill doesn't fire:

```markdown
## Video knowledge base

`video-knowledge/<name>/` holds distilled artifacts of videos this project
draws on. When implementing anything attributed to a video, consult the
`video-knowledge` skill — read the artifact's distilled.yaml first, and
open the cited frames/ images when text is ambiguous. Do not answer from
memory of "how that technique usually works" when an artifact covers it.
```

That's it. "Implement the decay-curve trick from the attention talk" will
now resolve against what the talk actually said — with timestamps, quotes,
and the original frames one Read away.
