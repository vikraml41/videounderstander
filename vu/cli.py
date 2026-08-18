"""Command-line entry point.

  vu run VIDEO -a artifact/ [--transcript subs.vtt]   # full pipeline
  vu sample VIDEO -a artifact/                        # or stage by stage:
  vu transcribe -a artifact/ --from subs.vtt | --video VIDEO
  vu ocr | align | describe | reconstruct | resolve | distill | verify
  vu status -a artifact/

Every stage is idempotent and cached: re-running a stage after upstream
changes only re-bills the LLM calls whose inputs actually changed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .artifact import Artifact
from .llm import LLM, DEFAULT_MODEL


def _llm(art: Artifact, args) -> LLM:
    return LLM(cache_dir=art.cache_dir, model=args.model)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="vu", description="Turn videos into verified text artifacts")
    parser.add_argument("-a", "--artifact", default="artifact",
                        help="artifact directory (default: ./artifact)")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help=f"Claude model id (default: {DEFAULT_MODEL})")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("sample", help="step 1: sample + dedup frames")
    p.add_argument("video", type=Path)
    p.add_argument("--fps", type=float, default=1.0)

    p = sub.add_parser("transcribe", help="step 0a: transcript ingest / ASR")
    p.add_argument("--video", type=Path, default=None)
    p.add_argument("--from", dest="from_file", type=Path, default=None,
                   help="existing .vtt/.srt/whisper-.json transcript")
    p.add_argument("--asr-model", default="medium")

    sub.add_parser("ocr", help="step 0b(i): OCR kept frames")
    sub.add_parser("correct", help="step 0b(ii): fix ASR errors against OCR")
    sub.add_parser("align", help="step 2: interleave frames + transcript")
    sub.add_parser("describe", help="step 3: context-aware frame descriptions")
    sub.add_parser("reconstruct", help="step 4: re-encode structural visuals")
    sub.add_parser("resolve", help="step 5: deixis resolution")
    sub.add_parser("distill", help="step 6: schema distillation + quote check")

    p = sub.add_parser("verify", help="step 7: comprehension-question loop")
    p.add_argument("-n", "--questions", type=int, default=18)

    p = sub.add_parser("run", help="full pipeline end to end")
    p.add_argument("video", type=Path)
    p.add_argument("--transcript", type=Path, default=None,
                   help="existing transcript file (skips ASR)")
    p.add_argument("--fps", type=float, default=1.0)
    p.add_argument("--asr-model", default="medium")

    sub.add_parser("status", help="show which stages have run")

    args = parser.parse_args(argv)
    art = Artifact(args.artifact).ensure()

    try:
        return _dispatch(args, art)
    except (FileNotFoundError, RuntimeError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1


def _dispatch(args, art: Artifact) -> int:
    from . import (deixis, describe, distill, interleave, ocr_correct,
                   reconstruct, sampling, transcribe, verify)

    if args.cmd == "sample":
        manifest = sampling.run_sampling(args.video, art, fps=args.fps)
        print(f"kept {len(manifest)} frames -> {art.frames_manifest}")

    elif args.cmd == "transcribe":
        segs = transcribe.run_transcribe(
            art, video=args.video, from_file=args.from_file,
            model_size=args.asr_model)
        print(f"{len(segs)} segments -> {art.transcript_json}")

    elif args.cmd == "ocr":
        ocr = ocr_correct.run_ocr(art, _llm(art, args))
        n = sum(1 for v in ocr.values() if v)
        print(f"OCR on {len(ocr)} frames ({n} with text) -> {art.ocr_json}")

    elif args.cmd == "correct":
        ocr_correct.run_correction(art, _llm(art, args))
        print(f"corrected transcript -> {art.corrected_transcript_json}")

    elif args.cmd == "align":
        interleave.run_align(art)
        print(f"-> {art.aligned_md}")

    elif args.cmd == "describe":
        descs = describe.run_describe(art, _llm(art, args))
        print(f"described {len(descs)} frames -> {art.descriptions_json}")

    elif args.cmd == "reconstruct":
        index = reconstruct.run_reconstruct(art, _llm(art, args))
        bad = [r for r in index
               if r["roundtrip"].get("verdict") == "mismatch"]
        print(f"{len(index)} reconstructions -> {art.reconstructions_dir}"
              + (f" ({len(bad)} still mismatched after repair — inspect them)"
                 if bad else ""))

    elif args.cmd == "resolve":
        deixis.run_resolve(art, _llm(art, args))
        print(f"-> {art.resolved_md}")

    elif args.cmd == "distill":
        doc = distill.run_distill(art, _llm(art, args))
        n_claims = sum(len(c.get("claims") or []) for c in doc["chunks"])
        msg = f"{n_claims} claims -> {art.distilled_yaml}"
        if doc["quote_check_failures"]:
            msg += (f"  WARNING: {doc['quote_check_failures']} claims failed "
                    "the verbatim-quote check (possible novelty collapse) — "
                    "grep quote_verified: false")
        print(msg)

    elif args.cmd == "verify":
        report = verify.run_verify(art, _llm(art, args),
                                   n_questions=args.questions)
        print(f"{report['iteration']}: score {report['score']}, "
              f"{len(report['holes'])} holes, "
              f"{len(report['unsupported_claims'])} unsupported claims")
        for h in report["holes"]:
            print(f"  [{h['verdict']}] ({h['probes']}, t~{h['t']}) {h['hole']}")

    elif args.cmd == "run":
        llm = _llm(art, args)
        print("1/8 sampling + dedup ...")
        sampling.run_sampling(args.video, art, fps=args.fps)
        print("2/8 transcript ...")
        transcribe.run_transcribe(art, video=args.video,
                                  from_file=args.transcript,
                                  model_size=args.asr_model)
        print("3/8 OCR + ASR correction ...")
        ocr_correct.run_ocr(art, llm)
        ocr_correct.run_correction(art, llm)
        print("4/8 frame descriptions ...")
        describe.run_describe(art, llm)
        print("5/8 align ...")
        interleave.run_align(art)
        print("6/8 reconstructions ...")
        reconstruct.run_reconstruct(art, llm)
        print("7/8 deixis + distill ...")
        deixis.run_resolve(art, llm)
        distill.run_distill(art, llm)
        print("8/8 verification ...")
        report = verify.run_verify(art, llm)
        print(f"done — {report['iteration']} score {report['score']}; "
              f"see {art.qa_dir / report['iteration'] / 'report.json'}")

    elif args.cmd == "status":
        checks = [
            ("frames", art.frames_manifest),
            ("transcript", art.transcript_json),
            ("ocr", art.ocr_json),
            ("corrected transcript", art.corrected_transcript_json),
            ("descriptions", art.descriptions_json),
            ("aligned", art.aligned_md),
            ("reconstructions", art.reconstructions_index),
            ("resolved", art.resolved_md),
            ("distilled", art.distilled_yaml),
        ]
        for name, path in checks:
            print(f"  [{'x' if path.exists() else ' '}] {name}")
        iters = sorted(art.qa_dir.glob("iter_*/report.json"))
        for r in iters:
            rep = json.loads(r.read_text())
            print(f"  verify {rep['iteration']}: {rep['score']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
