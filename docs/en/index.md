---
layout: home
hero:
  name: Quran Muaalem
  text: Correct Holy Quran recitations by analyzing phonemes, diacritization, articulatory attributes, and Tajweed rules using source-available AI
  tagline: Model + tools + transcript pipeline in one repo
  actions:
    - theme: brand
      text: Getting Started
      link: /en/getting-started
    - theme: alt
      text: Quran Transcript
      link: /en/quran-transcript/
features:
  - title: Multi‑level CTC
    details: The model decodes phonemes and Tajweed‑related attributes using a multi‑level CTC head.
  - title: Reference‑aware analysis
    details: Inference compares predicted phonemes with a reference phonetic script from quran-transcript.
  - title: Gradio UI
    details: A ready‑to‑run UI is wired in src/quran_muaalem/gradio_app.py.
---

## Project links

- GitHub: https://github.com/obadx/quran-muaalem
- PyPI: https://pypi.org/project/quran-muaalem/
- Hugging Face model: https://huggingface.co/obadx/muaalem-model-v3_2
- Hugging Face dataset: https://huggingface.co/datasets/obadx/muaalem-annotated-v3
- Paper: https://arxiv.org/abs/2509.00094

## What this repo contains

- Quran Muaalem inference and UI code under `src/quran_muaalem/`.
- A full Quran transcript toolkit under `quran-transcript/` (packaged as `quran-transcript`).
- Deployment and experiment helpers under `deploy/`, `tests/`, and `assets/`.

## For researchers: where to start

If your goal is evaluation or reproducibility:

1. Read **Architecture** to understand multi‑level CTC.
2. Review **Pipeline Steps** for label generation.
3. Use **Evaluation and Metrics** to standardize reporting.
4. Check **Outputs** for the exact output schema.

## Practical notes

- The model expects 16 kHz audio.
- Output quality depends on the reference phonetic script produced by `quran_transcript.quran_phonetizer`.
- Recommended metrics and reporting templates live under **Evaluation and Metrics**.

Use the navigation to dive into each component.
