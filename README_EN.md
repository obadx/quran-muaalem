<div align="center">

**English** | [العربية](README.md)

</div>

---

# Quran Muaalem

<div align="center">
<strong>With the help and guidance of Allah alone, we present the Intelligent Quran Teacher capable of detecting recitation errors, tajweed rules, and letter characteristics</strong>

[![PyPI][pypi-badge]][pypi-url]
[![Python Versions][python-badge]][python-url]
[![Hugging Face Model][hf-model-badge]][hf-model-url]
[![Hugging Face Dataset][hf-dataset-badge]][hf-dataset-url]
[![Google Colab][colab-badge]][colab-url]
[![arXiv][arxiv-badge]][arxiv-url]
[![QML License][qml-badge]][qml-url]
[![Discord][discord-badge]][discord-url]

</div>

[pypi-badge]: https://img.shields.io/pypi/v/quran-muaalem.svg
[pypi-url]: https://pypi.org/project/quran-muaalem/
[qml-badge]: https://img.shields.io/badge/License-QML%201.0-blue.svg
[qml-url]: https://github.com/obadx/quran-muaalem/blob/main/LICENSE
[python-badge]: https://img.shields.io/pypi/pyversions/quran-muaalem.svg
[python-url]: https://pypi.org/project/quran-muaalem/
[colab-badge]: https://img.shields.io/badge/Google%20Colab-Open%20in%20Colab-F9AB00?logo=google-colab&logoColor=white
[colab-url]: https://colab.research.google.com/drive/1If0G9NtdXiSRu6PVGtIMvLwxizF2jspn?usp=sharing
[hf-model-badge]: https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-blue
[hf-model-url]: https://huggingface.co/obadx/muaalem-model-v3_0
[hf-dataset-badge]: https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Dataset-orange
[hf-dataset-url]: https://huggingface.co/datasets/obadx/muaalem-annotated-v3
[arxiv-badge]: https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg
[arxiv-url]: https://arxiv.org/abs/2509.00094
[discord-badge]: https://img.shields.io/badge/Discord-Join%20Community-7289da?logo=discord&logoColor=white
[discord-url]: https://discord.gg/hJWW6fCH

<div align="center" style="background-color: #f0f8ff; border-left: 5px solid #4CAF50; padding: 15px; margin: 20px 0; border-radius: 5px;">
  <h3 style="color: #2c3e50; margin-top: 0;">📖 Try the Quran Muaalem Demo</h3>
  <p style="margin: 10px 0;">Click to try it out:</p>
  <a href="https://662a040e1863a5445c.gradio.live" style="display: inline-block; background-color: #4CAF50; color: white; padding: 10px 20px; text-decoration: none; border-radius: 5px; font-weight: bold; margin: 10px 0;">Demo Link</a>
  <p style="background-color: #ffeb3b; padding: 8px; border-radius: 3px; display: inline-block; margin: 10px 0;">
    ⚠️ <strong>Note:</strong> This link will expire on <span style="color: #d32f2f; font-weight: bold;">August 27, 2025</span>
  </p>
</div>

[![ALT_TEXT](https://img.youtube.com/vi/CsFoznO08-Q/0.jpg)](https://www.youtube.com/watch?v=CsFoznO08-Q)


## Table of Contents

- [AI Advisory Notice](#ai-advisory-notice)
- [Features](#features)
- [Architecture](#architecture)
- [Development Steps](#development-steps)
- [Using the Model](#using-the-model)
- [Basic Usage Example](#basic-usage-example)
- [API Servers](#api-servers)
- [Engine Settings](#engine-settings)
- [App Settings](#app-settings)
- [Endpoints](#endpoints)
- [Moshaf Attributes](#moshaf-attributes)
- [Tajweed Rules](#tajweed-rules)
- [Example: Quran Search](#example-quran-search)
- [Complete Example: Recitation Correction](#complete-example-recitation-correction)
- [Interactive OpenAPI Documentation](#interactive-openapi-documentation)
- [License](#license)

## AI Advisory Notice

> **English:** This Quranic Teacher is an artificial-intelligence model that may make mistakes. It is intended for assistance only and must not be relied upon. It is not a replacement for a proficient, authorized teacher, because the Quran is only received through oral transmission (mushafahah).
>
> **العربية:** هذا المعلّم القرآني نموذج ذكاء اصطناعي قد يخطئ؛ فهو للإعانة فقط ولا يُعتمد عليه، وليس بديلاً عن معلّمٍ مُتقِنٍ مُجاز؛ لأنّ القرآن لا يُؤخذ إلا بالمشافهة.

The Arabic text controls in case of any discrepancy.

## Features

* Trained on phonetic transcription of the Holy Quran: [quran-transcript](https://github.com/obadx/quran-transcript) - capable of detecting letter errors, tajweed, and letter characteristics
* Reasonable model size: 660M parameters
* Requires only 1.5 GB of GPU memory
* Innovative architecture: Multi-level CTC

## Architecture

Innovative architecture: Multi-level CTC, where each level trains on a specific aspect.

![multi-lvel-ctc](./assets/figures/mutli-level-ctc.png)

## Development Steps

* Collecting Quranic recitations from proficient reciters: [prepare-quran-dataset](https://github.com/obadx/prepare-quran-dataset)
* Segmenting recitations by pause points (not verses) using the [segmenter](https://github.com/obadx/recitations-segmenter)
* Extracting Quranic text from audio segments using the [Tarteel model](https://huggingface.co/tarteel-ai/whisper-base-ar-quran)
* Correcting extracted text using the [tasme'a (memorization verification) algorithm](https://github.com/obadx/quran-transcript)
* Converting standard script to Uthmani script: [quran-transcript](https://github.com/obadx/quran-transcript)
* Converting Uthmani script to Quranic phonetic script describing all tajweed rules (except ishmam): [quran-transcript](https://github.com/obadx/quran-transcript)
* Training the model on [Wav2Vec2BERT](https://huggingface.co/docs/transformers/model_doc/wav2vec2-bert) architecture


## Using the Model


### Using the Model via Gradio Interface

Install [uv](https://docs.astral.sh/uv/):

```bash
pip install uv
```
Or:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then install `ffmpeg`:

```bash
sudo apt-get update
sudo apt-get install -y ffmpeg
```

Or via `anaconda`:
```bash
conda install ffmpeg
```

Run `gradio` with a single command:
```bash
uvx --no-cache --from https://github.com/obadx/quran-muaalem.git[ui]  quran-muaalem-ui
```
Or:
```bash
uvx quran-muaalem[ui]  quran-muaalem-ui
```

### Via Python API


#### Installation

First, install the required dependencies:

```bash
# Install system dependencies
sudo apt-get install -y ffmpeg libsndfile1 portaudio19-dev

# Install Python packages
pip install quran-muaalem librosa "numba>=0.61.2"
```

## Basic Usage Example

```python
"""
Basic example of using the Quran Muaalem package for phonetic analysis of Quranic recitation.
"""

from dataclasses import asdict
import json
import logging

from quran_transcript import Aya, quran_phonetizer, MoshafAttributes
import torch
from librosa.core import load

# Import the main Muaalem class (adjust import based on your actual package structure)
from quran_muaalem import Muaalem

# Setup logging to see informative messages
logging.basicConfig(level=logging.INFO)

def analyze_recitation(audio_path):
    """
    Analyze a Quranic recitation audio file using the Muaalem model.

    Args:
        audio_path (str): Path to the audio file to analyze
    """
    # Configuration
    sampling_rate = 16000  # Must be 16000 Hz
    device = "cuda" if torch.cuda.is_available() else "cpu"  # Use GPU if available

    # Step 1: Prepare the Quranic reference text
    # Get the Uthmani script for a specific verse (Aya 8, Surah 75 in this example)
    uthmani_ref = Aya(8, 75).get_by_imlaey_words(17, 9).uthmani

    # Step 2: Configure the recitation style (Moshaf attributes)
    moshaf = MoshafAttributes(
        rewaya="hafs",        # Recitation style (Hafs is most common)
        madd_monfasel_len=2,  # Length of separated elongation
        madd_mottasel_len=4,  # Length of connected elongation
        madd_mottasel_waqf=4, # Length of connected elongation when stopping
        madd_aared_len=2,     # Length of necessary elongation
    )
    # see: https://github.com/obadx/prepare-quran-dataset?tab=readme-ov-file#moshaf-attributes-docs

    # Step 3: Convert text to phonetic representation
    # see docs for phnetizer: https://github.com/obadx/quran-transcript
    phonetizer_out = quran_phonetizer(uthmani_ref, moshaf, remove_spaces=True)

    # Step 4: Initialize the Muaalem model
    muaalem = Muaalem(device=device)

    # Step 5: Load and prepare the audio
    wave, _ = load(audio_path, sr=sampling_rate, mono=True)

    # Step 6: Process the audio with the model
    # The model analyzes the phonetic properties of the recitation
    outs = muaalem(
        [wave],           # Audio data
        [phonetizer_out],          # Phonetic reference
        sampling_rate=sampling_rate
    )

    # Step 7: Display the results
    for out in outs:
        print("Predicted Phonemes:", out.phonemes.text)

        # Display detailed phonetic features for each phoneme
        for sifa in out.sifat:
            print(json.dumps(asdict(sifa), indent=2, ensure_ascii=False))
            print("*" * 30)
        print("-" * 40)

    # Explaining Results
    explain_for_terminal(
        outs[0].phonemes.text,
        phonetizer_out.phonemes,
        outs[0].sifat,
        phonetizer_out.sifat,
    )


if __name__ == "__main__":
    # Replace with the path to your audio file
    audio_path = "./assets/test.wav"

    try:
        analyze_recitation(audio_path)
    except Exception as e:
        logging.error(f"Error processing audio: {e}")
```

Output:

```bash
ءِننننَللَااهَبِكُللِشَيءِنعَلِۦۦمُ۾۾۾بَرَااااءَتُممممِنَللَااهِوَرَسُۥۥلِه
```

| Phonemes | Tafashie | Qalqla | Ghonna | Hams Or Jahr | Safeer | Tikraar | Tafkheem Or Taqeeq | Istitala | Shidda Or Rakhawa | Itbaq |
|:--------:|:--------:|:------:|:------:|:------------:|:------:|:-------:|:-----------------:|:--------:|:-----------------:|:-----:|
| ءِ | not_motafashie | not_moqalqal | not_maghnoon | jahr | no_safeer | not_mokarar | moraqaq | not_mostateel | shadeed | monfateh |
| ننننَ | not_motafashie | not_moqalqal | maghnoon | jahr | no_safeer | not_mokarar | moraqaq | not_mostateel | between | monfateh |
| للَ | not_motafashie | not_moqalqal | not_maghnoon | jahr | no_safeer | not_mokarar | mofakham | not_mostateel | between | monfateh |
| اا | not_motafashie | not_moqalqal | not_maghnoon | jahr | no_safeer | not_mokarar | mofakham | not_mostateel | rikhw | monfateh |
| هَ | not_motafashie | not_moqalqal | not_maghnoon | hams | no_safeer | not_mokarar | moraqaq | not_mostateel | rikhw | monfateh |
| بِ | not_motafashie | not_moqalqal | not_maghnoon | jahr | no_safeer | not_mokarar | moraqaq | not_mostateel | shadeed | monfateh |
| كُ | not_motafashie | not_moqalqal | not_maghnoon | hams | no_safeer | not_mokarar | moraqaq | not_mostateel | shadeed | monfateh |
| للِ | not_motafashie | not_moqalqal | not_maghnoon | jahr | no_safeer | not_mokarar | moraqaq | not_mostateel | between | monfateh |
| شَ | motafashie | not_moqalqal | not_maghnoon | hams | no_safeer | not_mokarar | moraqaq | not_mostateel | rikhw | monfateh |
| ي | not_motafashie | not_moqalqal | not_maghnoon | jahr | no_safeer | not_mokarar | moraqaq | not_mostateel | rikhw | monfateh |
| ءِ | not_motafashie | not_moqalqal | not_maghnoon | jahr | no_safeer | not_mokarar | moraqaq | not_mostateel | shadeed | monfateh |
| ن | not_motafashie | not_moqalqal | maghnoon | jahr | no_safeer | not_mokarar | moraqaq | not_mostateel | between | monfateh |
| عَ | not_motafashie | not_moqalqal | not_maghnoon | jahr | no_safeer | not_mokarar | moraqaq | not_mostateel | between | monfateh |
| لِ | not_motafashie | not_moqalqal | not_maghnoon | jahr | no_safeer | not_mokarar | moraqaq | not_mostateel | between | monfateh |
| ۦۦ | not_motafashie | not_moqalqal | not_maghnoon | jahr | no_safeer | not_mokarar | moraqaq | not_mostateel | rikhw | monfateh |
| مُ | not_motafashie | not_moqalqal | maghnoon | jahr | no_safeer | not_mokarar | moraqaq | not_mostateel | between | monfateh |
| ۾۾۾ | not_motafashie | not_moqalqal | maghnoon | jahr | no_safeer | not_mokarar | moraqaq | not_mostateel | rikhw | monfateh |
| بَ | not_motafashie | not_moqalqal | not_maghnoon | jahr | no_safeer | not_mokarar | moraqaq | not_mostateel | shadeed | monfateh |
| رَ | not_motafashie | not_moqalqal | not_maghnoon | jahr | no_safeer | mokarar | mofakham | not_mostateel | between | monfateh |
| اااا | not_motafashie | not_moqalqal | not_maghnoon | jahr | no_safeer | not_mokarar | mofakham | not_mostateel | rikhw | monfateh |
| ءَ | not_motafashie | not_moqalqal | not_maghnoon | jahr | no_safeer | not_mokarar | moraqaq | not_mostateel | shadeed | monfateh |
| تُ | not_motafashie | not_moqalqal | not_maghnoon | hams | no_safeer | not_mokarar | moraqaq | not_mostateel | shadeed | monfateh |
| ممممِ | not_motafashie | not_moqalqal | maghnoon | jahr | no_safeer | not_mokarar | moraqaq | not_mostateel | between | monfateh |
| نَ | not_motafashie | not_moqalqal | maghnoon | jahr | no_safeer | not_mokarar | moraqaq | not_mostateel | between | monfateh |
| للَ | not_motafashie | not_moqalqal | not_maghnoon | jahr | no_safeer | not_mokarar | mofakham | not_mostateel | between | monfateh |
| اا | not_motafashie | not_moqalqal | not_maghnoon | jahr | no_safeer | not_mokarar | mofakham | not_mostateel | rikhw | monfateh |
| هِ | not_motafashie | not_moqalqal | not_maghnoon | hams | no_safeer | not_mokarar | moraqaq | not_mostateel | rikhw | monfateh |
| وَ | not_motafashie | not_moqalqal | not_maghnoon | jahr | no_safeer | not_mokarar | moraqaq | not_mostateel | rikhw | monfateh |
| رَ | not_motafashie | not_moqalqal | not_maghnoon | jahr | no_safeer | mokarar | mofakham | not_mostateel | between | monfateh |
| سُ | not_motafashie | not_moqalqal | not_maghnoon | hams | safeer | not_mokarar | moraqaq | not_mostateel | rikhw | monfateh |
| ۥۥ | not_motafashie | not_moqalqal | not_maghnoon | jahr | no_safeer | not_mokarar | moraqaq | not_mostateel | rikhw | monfateh |
| لِ | not_motafashie | not_moqalqal | not_maghnoon | jahr | no_safeer | not_mokarar | moraqaq | not_mostateel | between | monfateh |
| ه | not_motafashie | not_moqalqal | not_maghnoon | hams | no_safeer | not_mokarar | moraqaq | not_mostateel | rikhw | monfateh |


### API Docs

```python
class Muaalem:
    def __init__(
        self,
        model_name_or_path: str = "obadx/muaalem-model-v3_2",
        device: str = "cpu",
        dtype=torch.bfloat16,
    ):
        """
        Initializing Muallem Model

        Args:
            model_name_or_path: the huggingface model name or path
            device: the device to run model on
            dtype: the torch dtype. Default is `torch.bfloat16` as the model was trained on
        """

    @torch.no_grad()
    def __call__(
        self,
        waves: list[list[float] | torch.FloatTensor | NDArray],
        ref_quran_phonetic_script_list: list[QuranPhoneticScriptOutput],
        sampling_rate: int,
    ) -> list[MuaalemOutput]:
        """Infrence Funcion for the Quran Muaalem Project

                waves: input waves  batch , seq_len with different formats described above
                ref_quran_phonetic_script_list (list[QuranPhoneticScriptOutput]): list of the
                    phonetized ouput of `quran_transcript.quran_phonetizer` with `remove_space=True`

                sampleing_rate (int): has to be 16000

        Returns:
            list[MuaalemOutput]:
                A list of output objects, each containing phoneme predictions and their
                phonetic features (sifat) for a processed input.

            Each MuaalemOutput contains:
                phonemes (Unit):
                    A dataclass representing the predicted phoneme sequence with:
                        text (str): Concatenated string of all phonemes.
                        probs (Union[torch.FloatTensor, list[float]]):
                            Confidence probabilities for each predicted phoneme.
                        ids (Union[torch.LongTensor, list[int]]):
                            Token IDs corresponding to each phoneme.

                sifat (list[Sifa]):
                    A list of phonetic feature dataclasses (one per phoneme) with the
                    following optional properties (each is a SingleUnit or None):
                        - phonemes_group (str): the phonemes associated with the `sifa`
                        - hams_or_jahr (SingleUnit): either `hams` or `jahr`
                        - shidda_or_rakhawa (SingleUnit): either `shadeed`, `between`, or `rikhw`
                        - tafkheem_or_taqeeq (SingleUnit): either `mofakham`, `moraqaq`, or `low_mofakham`
                        - itbaq (SingleUnit): either `monfateh`, or `motbaq`
                        - safeer (SingleUnit): either `safeer`, or `no_safeer`
                        - qalqla (SingleUnit): eithr `moqalqal`, or `not_moqalqal`
                        - tikraar (SingleUnit): either `mokarar` or `not_mokarar`
                        - tafashie (SingleUnit): either `motafashie`, or `not_motafashie`
                        - istitala (SingleUnit): either `mostateel`, or `not_mostateel`
                        - ghonna (SingleUnit): either `maghnoon`, or `not_maghnoon`

            Each SingleUnit in Sifa properties contains:
                text (str): The feature's categorical label (e.g., "hams", "shidda").
                prob (float): Confidence probability for this feature.
                idx (int): Identifier for the feature class.
        """
```


---

## API Servers

The engine has two main features:
1. Voice search in the Holy Quran
2. Correcting Quranic recitations with tajweed rules

The API consists of:

1. **Engine**: Runs the Wav2Vec2-BERT model to convert audio to phonemes
2. **App**: Provides search, correction, and transcription interfaces

### Installation

```bash
uv add quran-muaalem[engine]
```

### Running the Servers

```bash
# First terminal: Run the engine (port 8000)
uv run quran-muaalem-engine

# Second terminal: Run the app (port 8001)
uv run quran-muaalem-app
```

---

## Engine Settings

Settings are located in `src/quran_muaalem/engine/settings.py`:

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `model_name_or_path` | string | `obadx/muaalem-model-v3_2` | HuggingFace model path |
| `dtype` | string | `bfloat16` | Data type: `float32`, `float16`, `bfloat16` |
| `max_audio_seconds` | float | `15` | Maximum audio length in seconds |
| `max_batch_size` | int | `128` | Maximum batch size for processing |
| `batch_timeout` | float | `0.4` | Batch wait timeout in seconds |
| `host` | string | `0.0.0.0` | Server bind address |
| `port` | int | `8000` | Server port |
| `accelerator` | string | `cuda` | Hardware accelerator: `cuda`, `cpu`, `mps` |
| `devices` | int | `1` | Number of devices |
| `workers_per_device` | int | `1` | Number of workers per device |
| `timeout` | float | `90.0` | Request timeout in seconds |

---

## App Settings

Settings are located in `src/quran_muaalem/app/settings.py`:

| Setting | Default | Description |
|---------|---------|-------------|
| `engine_url` | `http://0.0.0.0:8000/predict` | URL for the engine's `/predict` endpoint |
| `host` | `0.0.0.0` | Server bind address |
| `port` | `8001` | Server port |
| `error_ratio` | `0.1` | Allowed error ratio for search (0.0-1.0) |
| `max_workers_phonetic_search` | `cpu_count // 2` | Number of concurrent phonetic search workers |
| `max_workers_phonetization` | `cpu_count // 2` | Number of concurrent phonetization workers |

---

## Endpoints

### Engine (Port 8000)

| Endpoint | Description |
|----------|-------------|
| `/predict` | Convert audio to phonemes |
| `/health` | Server health check |
| `/docs` | Interactive OpenAPI documentation |
| `/redoc` | Alternative ReDoc documentation |

### App (Port 8001)

| Endpoint | Description |
|----------|-------------|
| `/health` | Check app status and connection to the engine |
| `/search` | Search the Quran by audio or phonetic text |
| `/correct-recitation` | Analyze recitation and detect tajweed errors |
| `/transcript` | Transcribe audio to phonetic text (proxy to engine) |
| `/docs` | Interactive OpenAPI documentation |
| `/redoc` | Alternative ReDoc documentation |

---

## Moshaf Attributes

These attributes define the recitation rules for Hafs reading. All fields are optional:

| Attribute | Arabic | Values | Default | Description |
|-----------|--------|--------|---------|-------------|
| `rewaya` | الرواية | `hafs` (حفص) | `hafs` | Type of Quran recitation |
| `recitation_speed` | سرعة التلاوة | `mujawad` (مجود), `above_murattal` (فويق المرتل), `murattal` (مرتل), `hadr` (حدر) | `murattal` | Recitation speed ordered from slowest to fastest |
| `takbeer` | التكبير | `no_takbeer` (لا تكبير), `beginning_of_sharh` (التكبير من أول الشرح لأول الناس), `end_of_doha` (التكبير من آخر الضحى لآخر الناس), `general_takbeer` (التكبير أول كل سورة إلا التوبة) | `no_takbeer` | Methods of adding takbeer (Allahu Akbar) after isti'adha and between end of surah and beginning of surah |
| `madd_monfasel_len` | مد المنفصل | `2`, `3`, `4`, `5` | `4` | Length of separated madd (madd al-munfasil) for Hafs reading |
| `madd_mottasel_len` | مقدار المد المتصل | `4`, `5`, `6` | `4` | Length of connected madd for Hafs reading |
| `madd_mottasel_waqf` | مقدار المد المتصل وقفا | `4`, `5`, `6` | `4` | Length of connected madd when stopping for Hafs reading |
| `madd_aared_len` | مقدار مد العارض | `2`, `4`, `6` | `4` | Length of madd al-'arid lil-sukun (temporary madd due to stopping) |
| `madd_alleen_len` | مقدار مد اللين | `2`, `4`, `6` | `None` | Length of leen madd when stopping (defaults to madd_aared_len) |
| `ghonna_lam_and_raa` | غنة اللام و الراء | `ghonna` (غنة), `no_ghonna` (لا غنة) | `no_ghonna` | Ghunna in idgham of noon with lam and raa for Hafs reading |
| `meem_aal_imran` | ميم آل عمران | `waqf` (وقف), `wasl_2` (فتح الميم ومدها حركتين), `wasl_6` (فتح الميم ومدها ستة حركات) | `waqf` | Method of reciting {الم الله} in connected recitation |
| `madd_yaa_alayn_alharfy` | مقدار المد اللازم الحرفي للعين | `2`, `4`, `6` | `6` | Length of required letter madd for letter 'ayn in Surah Maryam and Ash-Shura |
| `saken_before_hamz` | الساكن قبل الهمز | `tahqeek` (تحقيق), `general_sakt` (سكت عام), `local_sakt` (سكت خاص) | `tahqeek` | How to recite the silent letter before hamza for Hafs reading |
| `sakt_iwaja` | السكت عند عوجا في الكهف | `sakt` (سكت), `waqf` (وقف), `idraj` (إدراج) | `waqf` | How to recite 'iwaja in Surah Al-Kahf |
| `sakt_marqdena` | السكت عند مرقدنا في يس | `sakt` (سكت), `waqf` (وقف), `idraj` (إدراج) | `waqf` | How to recite 'marqadena' in Surah Ya-Sin |
| `sakt_man_raq` | السكت عند من راق في القيامة | `sakt` (سكت), `waqf` (وقف), `idraj` (إدراج) | `sakt` | How to recite 'man raq' in Surah Al-Qiyamah |
| `sakt_bal_ran` | السكت عند بل ران في المطففين | `sakt` (سكت), `waqf` (وقف), `idraj` (إدراج) | `sakt` | How to recite 'bal ran' in Surah Al-Mutaffifin |
| `sakt_maleeyah` | وجه قوله {ماليه هلك} بالحاقة | `sakt` (سكت), `waqf` (وقف), `idgham` (إدغام) | `waqf` | How to recite 'maaliyah halak' in Surah Al-Haqqah |
| `between_anfal_and_tawba` | وجه بين الأنفال والتوبة | `waqf` (وقف), `sakt` (سكت), `wasl` (وصل) | `waqf` | How to recite the transition between Surah Al-Anfal and Surah At-Tawbah |
| `noon_and_yaseen` | الإظهار في النون | `izhar` (إظهار), `idgham` (إدغام) | `izhar` | Idgham of noon in Ya-Sin and Noon wal-Qalam |
| `yaa_athan` | إثبات الياء وحذفها وقفا | `wasl` (وصل), `hadhf` (حذف), `ithbat` (إثبات) | `wasl` | Affirmation or deletion of yaa in {آتاني} in Surah An-Naml |
| `start_with_ism` | وجه البدأ بكلمة {الاسم} | `wasl` (وصل), `lism` (لسم), `alism` (ألسم) | `wasl` | Ruling on starting with the word 'al-ism' in Surah Al-Hujurat |
| `yabsut` | السين والصاد في {يقبض ويبسط} | `seen` (سين), `saad` (صاد) | `seen` | Pronunciation in Surah Al-Baqarah |
| `bastah` | السين والصاد في {بسطة} | `seen` (سين), `saad` (صاد) | `seen` | Pronunciation in Surah Al-A'raf |
| `almusaytirun` | السين والصاد في {المصيطرون} | `seen` (سين), `saad` (صاد) | `saad` | Pronunciation in Surah At-Tur |
| `bimusaytir` | السين والصاد في {بمصيطر} | `seen` (سين), `saad` (صاد) | `saad` | Pronunciation in Surah Al-Ghashiyah |
| `tasheel_or_madd` | همزة الوصل | `tasheel` (تسهيل), `madd` (مد) | `madd` | Tasheel or madd of hamzat al-wasl in {آلذكرين} |
| `yalhath_dhalik` | الإدغام في {يلهث ذلك} | `izhar` (إظهار), `idgham` (إدغام), `waqf` (وقف) | `idgham` | Idgham in Surah Al-A'raf |
| `irkab_maana` | الإدغام في {اركب معنا} | `izhar` (إظهار), `idgham` (إدغام), `waqf` (وقف) | `idgham` | Idgham in Surah Hud |
| `noon_tamnna` | الإشمام والروم في {تأمنا} | `ishmam` (إشمام), `rawm` (روم) | `ishmam` | Ishmam and rawm in Surah Yusuf |
| `harakat_daaf` | حركة الضاد في {ضعف} | `fath` (فتح), `dam` (ضم) | `fath` | Vowel of daad in Surah Ar-Rum |
| `alif_salasila` | الألف في {سلاسلا} | `hadhf` (حذف), `ithbat` (إثبات), `wasl` (وصل) | `wasl` | Affirmation or deletion of alif in Surah Al-Insan |
| `idgham_nakhluqkum` | إدغام القاف في الكاف | `idgham_kamil` (إدغام كامل), `idgham_naqis` (إدغام ناقص) | `idgham_kamil` | Idgham of qaf into kaf in Surah Al-Mursalat |
| `raa_firq` | راء {فرق} في الشعراء | `waqf` (وقف), `tafkheem` (تفخيم), `tarqeeq` (ترقيق) | `tafkheem` | Tafkheem and tarqeeq of raa in Surah Ash-Shu'ara |
| `raa_alqitr` | راء {القطر} في سبأ | `wasl` (وصل), `tafkheem` (تفخيم), `tarqeeq` (ترقيق) | `wasl` | Tafkheem and tarqeeq of raa in Surah Saba |
| `raa_misr` | راء {مصر} في يونس | `wasl` (وصل), `tafkheem` (تفخيم), `tarqeeq` (ترقيق) | `wasl` | Tafkheem and tarqeeq of raa in Surah Yunus |
| `raa_nudhur` | راء {نذر} في القمر | `wasl` (وصل), `tafkheem` (تفخيم), `tarqeeq` (ترقيق) | `tafkheem` | Tafkheem and tarqeeq of raa in Surah Al-Qamar |
| `raa_yasr` | راء {يسر} بالفجر | `wasl` (وصل), `tafkheem` (تفخيم), `tarqeeq` (ترقيق) | `tarqeeq` | Tafkheem and tarqeeq of raa in Surah Al-Fajr |
| `meem_mokhfah` | هل الميم مخفاة أو مدغمة | `meem` (ميم), `ikhfaa` (إخفاء) | `ikhfaa` | Ikhfaa or idgham of meem in the state of ikhfaa |

---

## Tajweed Rules

Tajweed rules used in error analysis. Imported from `quran_transcript.phonetics.tajweed_rulses`:

| Rule | Arabic | Check Type | Reference Length | Description |
|------|--------|------------|------------------|-------------|
| `Qalqalah` | قلقة | `match` | 0 | Qalqalah - echoing vibration when pronouncing a silent letter |
| `NormalMaddRule` | المد الطبيعي | `count` | 2 | Natural madd that occurs naturally in a word |
| `MonfaselMaddRule` | المد المنفصل | `count` | 4 | Separated madd between two words |
| `MottaselMaddRule` | المد المتصل | `count` | 4 | Connected madd within a word |
| `MottaselMaddPauseRule` | المد المتصل وقفا | `count` | 4 | Connected madd when stopping |
| `LazemMaddRule` | المد اللازم | `count` | 6 | Required madd in specific letters (such as the meem in Alif-Lam-Meem) |
| `AaredMaddRule` | المد العارض للسكون | `count` | 4 | Temporary madd that appears when stopping on a specific word |
| `LeenMaddRule` | مد اللين | `count` | 4 | Leen madd for silent waw and yaa preceded by a letter with fatha |

### Tajweed Rule Types Explained

1. **Qalqalah (قلقة)**: An echoing vibration when pronouncing a silent letter, occurring in the qalqalah letters: ق (qaf), ط (ta), ب (ba), ج (jim), د (dal)
2. **NormalMaddRule (المد الطبيعي)**: The natural madd that occurs in a word naturally, with a length of two counts
3. **MonfaselMaddRule (المد المنفصل)**: The madd between two words when one word ends with a madd letter
4. **MottaselMaddRule (المد المتصل)**: The madd within a word between madd letters
5. **MottaselMaddPauseRule (المد المتصل وقفا)**: The connected madd when stopping on a specific word
6. **LazemMaddRule (المد اللازم)**: The required madd in specific letters such as the meem in {الم} and the hamza in {ءآل}
7. **AaredMaddRule (المد العارض للسكون)**: The madd that appears when stopping due to sukun (silence)
8. **LeenMaddRule (مد اللين)**: The leen madd for silent waw and yaa preceded by a letter with fatha (opening vowel)

---

## Example: Quran Search

Search the Quran using audio or phonetic text.

### Command (curl)

```bash
curl -X 'POST' \
  'http://localhost:8001/search?error_ratio=0.1' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'file=@WhatsApp Ptt 2026-02-20 at 1.56.35 PM.ogg;type=application/ogg'
```

### Response (JSON)

```json
{
  "phonemes": "ءَلِفلَااممِۦۦم",
  "results": [
    {
      "start": {
        "sura_idx": 2,
        "aya_idx": 1,
        "uthmani_word_idx": 0,
        "uthmani_char_idx": 0,
        "phonemes_idx": 0
      },
      "end": {
        "sura_idx": 2,
        "aya_idx": 1,
        "uthmani_word_idx": 0,
        "uthmani_char_idx": 5,
        "phonemes_idx": 25
      },
      "uthmani_text": "الٓمٓ"
    },
    {
      "start": {
        "sura_idx": 3,
        "aya_idx": 1,
        "uthmani_word_idx": 0,
        "uthmani_char_idx": 0,
        "phonemes_idx": 0
      },
      "end": {
        "sura_idx": 3,
        "aya_idx": 1,
        "uthmani_word_idx": 0,
        "uthmani_char_idx": 5,
        "phonemes_idx": 25
      },
      "uthmani_text": "الٓمٓ"
    },
    {
      "start": {
        "sura_idx": 7,
        "aya_idx": 1,
        "uthmani_word_idx": 0,
        "uthmani_char_idx": 0,
        "phonemes_idx": 0
      },
      "end": {
        "sura_idx": 7,
        "aya_idx": 1,
        "uthmani_word_idx": 0,
        "uthmani_char_idx": 5,
        "phonemes_idx": 25
      },
      "uthmani_text": "الٓمٓصٓ"
    },
    {
      "start": {
        "sura_idx": 13,
        "aya_idx": 1,
        "uthmani_word_idx": 0,
        "uthmani_char_idx": 0,
        "phonemes_idx": 0
      },
      "end": {
        "sura_idx": 13,
        "aya_idx": 1,
        "uthmani_word_idx": 0,
        "uthmani_char_idx": 5,
        "phonemes_idx": 25
      },
      "uthmani_text": "الٓمٓر"
    },
    {
      "start": {
        "sura_idx": 29,
        "aya_idx": 1,
        "uthmani_word_idx": 0,
        "uthmani_char_idx": 0,
        "phonemes_idx": 0
      },
      "end": {
        "sura_idx": 29,
        "aya_idx": 1,
        "uthmani_word_idx": 0,
        "uthmani_char_idx": 5,
        "phonemes_idx": 25
      },
      "uthmani_text": "الٓمٓ"
    },
    {
      "start": {
        "sura_idx": 30,
        "aya_idx": 1,
        "uthmani_word_idx": 0,
        "uthmani_char_idx": 0,
        "phonemes_idx": 0
      },
      "end": {
        "sura_idx": 30,
        "aya_idx": 1,
        "uthmani_word_idx": 0,
        "uthmani_char_idx": 5,
        "phonemes_idx": 25
      },
      "uthmani_text": "الٓمٓ"
    },
    {
      "start": {
        "sura_idx": 31,
        "aya_idx": 1,
        "uthmani_word_idx": 0,
        "uthmani_char_idx": 0,
        "phonemes_idx": 0
      },
      "end": {
        "sura_idx": 31,
        "aya_idx": 1,
        "uthmani_word_idx": 0,
        "uthmani_char_idx": 5,
        "phonemes_idx": 25
      },
      "uthmani_text": "الٓمٓ"
    },
    {
      "start": {
        "sura_idx": 32,
        "aya_idx": 1,
        "uthmani_word_idx": 0,
        "uthmani_char_idx": 0,
        "phonemes_idx": 0
      },
      "end": {
        "sura_idx": 32,
        "aya_idx": 1,
        "uthmani_word_idx": 0,
        "uthmani_char_idx": 5,
        "phonemes_idx": 25
      },
      "uthmani_text": "الٓمٓ"
    }
  ],
  "message": null
}
```

### Response Explanation

- **phonemes**: Phonemes extracted from the input audio
- **results**: List of matching results in the Holy Quran, each result contains:
  - **start**: Start position of the match (surah number, verse number, word position, character position, phoneme position)
  - **end**: End position of the match
  - **uthmani_text**: Matching Uthmani script text
- **message**: Optional message (e.g., if no results found)

### Searching with Phonetic Text Directly

You can also search directly with phonetic text without an audio file:

```bash
curl -X 'POST' \
  'http://localhost:8001/search?phonetic_text=bismi&error_ratio=0.1'
```

---

## Complete Example: Recitation Correction

### Command (curl)

```bash
curl -X 'POST' \
  'http://localhost:8001/correct-recitation' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'error_ratio=0.1' \
  -F 'file=@WhatsApp Ptt 2026-02-20 at 1.56.35 PM.ogg;type=application/ogg'
```

### Response (JSON)

```json
{
  "start": {
    "sura_idx": 2,
    "aya_idx": 1,
    "uthmani_word_idx": 0,
    "uthmani_char_idx": 0,
    "phonemes_idx": 0
  },
  "end": {
    "sura_idx": 2,
    "aya_idx": 1,
    "uthmani_word_idx": 0,
    "uthmani_char_idx": 5,
    "phonemes_idx": 25
  },
  "predicted_phonemes": "ءَلِفلَااممِۦۦم",
  "reference_phonemes": "ءَلِفلَااااااممممِۦۦۦۦۦۦم",
  "uthmani_text": "الٓمٓ",
  "errors": [
    {
      "uthmani_pos": [1, 2],
      "ph_pos": [7, 13],
      "error_type": "tajweed",
      "speech_error_type": "replace",
      "expected_ph": "اااااا",
      "preditected_ph": "اا",
      "expected_len": 6,
      "predicted_len": 2,
      "ref_tajweed_rules": [
        {
          "name": {"ar": "المد اللازم", "en": "Lazem Madd"},
          "golden_len": 6,
          "correctness_type": "count",
          "tag": "alif"
        }
      ],
      "inserted_tajweed_rules": null,
      "replaced_tajweed_rules": null,
      "missing_tajweed_rules": null
    },
    {
      "uthmani_pos": [3, 4],
      "ph_pos": [13, 18],
      "error_type": "tajweed",
      "speech_error_type": "replace",
      "expected_ph": "ممممِ",
      "preditected_ph": "ممِ",
      "expected_len": 6,
      "predicted_len": 2,
      "ref_tajweed_rules": [
        {
          "name": {"ar": "المد اللازم", "en": "Lazem Madd"},
          "golden_len": 6,
          "correctness_type": "count",
          "tag": "yaa"
        }
      ],
      "inserted_tajweed_rules": null,
      "replaced_tajweed_rules": null,
      "missing_tajweed_rules": null
    },
    {
      "uthmani_pos": [3, 4],
      "ph_pos": [18, 24],
      "error_type": "tajweed",
      "speech_error_type": "replace",
      "expected_ph": "ۦۦۦۦۦۦ",
      "preditected_ph": "ۦۦ",
      "expected_len": 6,
      "predicted_len": 2,
      "ref_tajweed_rules": [
        {
          "name": {"ar": "المد اللازم", "en": "Lazem Madd"},
          "golden_len": 6,
          "correctness_type": "count",
          "tag": "yaa"
        }
      ],
      "inserted_tajweed_rules": null,
      "replaced_tajweed_rules": null,
      "missing_tajweed_rules": null
    }
  ]
}
```

### Response Explanation

- **start/end**: Position in the Quran (surah number, verse number, word position, character position, phoneme position)
- **predicted_phonemes**: Phonemes predicted from the audio
- **reference_phonemes**: Reference phonemes from the Quranic text using moshaf attributes
- **uthmani_text**: Matching Uthmani script text
- **errors**: List of detected errors, each error contains:
  - **error_type**: Type of error (`tajweed` = tajweed error, `normal` = normal error, `tashkeel` = diacritics error)
  - **speech_error_type**: Type of speech error (`insert` = insertion, `delete` = deletion, `replace` = replacement)
  - **expected_ph/predicted_ph**: Expected and predicted phonemes
  - **expected_len/predicted_len**: Expected and predicted length (for madd rules like madd lazem)
  - **ref_tajweed_rules**: Reference tajweed rules that should be applied

---

## Interactive OpenAPI Documentation

For complete interactive documentation with examples and detailed descriptions for every parameter, please visit:

- **App**: http://localhost:8001/docs
- **Engine**: http://localhost:8000/docs

This documentation includes:
- All endpoints with complete descriptions
- All parameters with their default values and data types
- Interactive examples for each endpoint
- Complete response schemas
- Direct execution capability from the browser

---

## License

This project is licensed under the **Quran Muaalem License (QML), Version 1.0** — the Apache License 2.0 with additional conditions. See [LICENSE](https://github.com/obadx/quran-muaalem/blob/main/LICENSE) for the full text. SPDX identifier: `LicenseRef-QML-1.0`.

### Additional Conditions

1. **AI usage and disclosure.** The Work uses an AI model that may make mistakes. It is for assistance only and must not be relied upon or presented as a substitute for a proficient, authorized teacher. The AI advisory notice must be shown before first use and remain permanently accessible.
2. **Compliance with Islamic Sharia.** All use, modification, distribution, and deployment must adhere to the rulings of Islamic Sharia. Any use that does not comply is unlicensed.
3. **No advertisements.** The Work and any covered product must never include advertisements, ad-serving code, advertising tracking, or the sale or sharing of user data for advertising. General analytics and crash reporting remain permitted.
4. **Commercial use and free access.** Commercial use is permitted. If you charge any fee for access to the Work or a covered product, you must also provide a clear, no-cost path to the essential functionality: automatically for users in World Bank low-income and lower-middle-income economies, and through a simple in-app path for individuals anywhere who cannot afford the fee. The free path must be visible before any payment is requested.
5. **Breach and cure period.** If You breach this License and do not cure the breach within thirty (30) days after the earlier of (i) written notice from the Licensor or any Contributor, or (ii) the time You become aware of the breach, all licenses granted to You under this License terminate automatically. If You cure the breach within that period, Your licenses remain in force.

Releases published before QML 1.0 remain available under the MIT License to those who received them (Section 3.4).
