"""videounderstander: convert videos into verified text-first artifacts.

Pipeline stages (see PLAN.md):
  0. transcribe  — ASR / transcript ingest, word timestamps, OCR correction
  1. sample      — dense frame sampling + tile-grid structural dedup
  2. align       — interleave frames and transcript by timestamp
  3. describe    — context-aware frame descriptions (incl. pointer targets)
  4. reconstruct — re-encode structural visuals (LaTeX/Mermaid/tables/code)
  5. resolve     — deixis resolution pass
  6. distill     — schema-forced distillation with verbatim-quote provenance
  7. verify      — comprehension-question loop measuring information loss
"""

__version__ = "0.1.0"
