import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import (
    Optional,
    Annotated,
)

import httpx
from fastapi import FastAPI, UploadFile, File, Query, Body, Form, Depends, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import HTTPException
from fastapi.responses import JSONResponse
from pydantic import Json


from quran_transcript import quran_phonetizer, explain_error
from quran_transcript.phonetics.moshaf_attributes import MoshafAttributes
from quran_transcript.phonetics.search import (
    PhoneticSearch,
    NoPhonemesSearchResult,
)

from .settings import AppSettings
from .types import (
    SearchResponse,
    SearchResultResponse,
    CorrectRecitationResponse,
    CorrectRecitationNoMatchResponse,
    ReciterErrorResponse,
    PhonemesSearchSpanApp,
    TajweedRuleApp,
    TajweedRuleNameApp,
    TranscriptResponse,
    correct_recitation_form_dependency,
)

# TODO:
"""
* [ ] Add timeout for both PHonmems threadpool and correct
* [ ] For both srearch and correct make the file is optional input with input phonems dirctorly
* [ ] Add transcribe end point as a proxy for the predict one
"""


app_settings = AppSettings()


app = FastAPI(
    title="Quran Muaalem Search API",
    description="""## Overview

Quran Muaalem is an AI-powered Quranic recitation correction system that provides:

- **Phonetic Search**: Search the Quran by audio or phonetic text
- **Recitation Correction**: Analyze audio recordings and detect recitation errors with Tajweed rule violations
- **Transcript Proxy**: Transcribe audio to phonetic script via the engine

## Architecture

The system consists of two components:

1. **Engine** (`quran-muaalem-engine`): Runs the Wav2Vec2-BERT CTC model for audio-to-phoneme transcription
2. **App** (`quran-muaalem-app`): Provides search and correction APIs, connects to the engine

## Moshaf Attributes

The app uses `MoshafAttributes` from `quran-transcript` to generate reference phonetic transcriptions.
These attributes define the recitation rules for the Hafs recitation (حفص).

| Attribute | Arabic | Values | Default | Description |
|-----------|--------|--------|---------|-------------|
| rewaya | الرواية | `hafs` (حفص) | `hafs` | The Quran Rewaya type |
| recitation_speed | سرعة التلاوة | `mujawad` (مجود), `above_murattal` (فويق المرتل), `murattal` (مرتل), `hadr` (حدر) | `murattal` | Recitation speed sorted from slowest to fastest |
| takbeer | التكبير | `no_takbeer` (لا تكبير), `beginning_of_sharh` (التكبير من أول الشرح لأول الناس), `end_of_doha` (التكبير من آخر الضحى لآخر الناس), `general_takbeer` (التكبير أول كل سورة إلا التوبة) | `no_takbeer` | Ways to add Takbeer (الله أكبر) after Istiaatha and between surahs |
| madd_monfasel_len | مد المنفصل | `2`, `3`, `4`, `5` | `4` | Length of Mad Al Monfasel (مد النفصل) for Hafs Rewaya |
| madd_mottasel_len | مقدار المد المتصل | `4`, `5`, `6` | `4` | Length of Mad Al Motasel (مد المتصل) for Hafs |
| madd_mottasel_waqf | مقدار المد المتصل وقفا | `4`, `5`, `6` | `4` | Length of Madd Almotasel at pause for Hafs |
| madd_aared_len | مقدار المد العارض | `2`, `4`, `6` | `4` | Length of Mad Al Aared (مد العارض للسكون) |
| madd_alleen_len | مقدار مد اللين | `2`, `4`, `6` | `None` | Length of Madd al-Leen when stopping (default equals madd_aared_len) |
| ghonna_lam_and_raa | غنة اللام و الراء | `ghonna` (غنة), `no_ghonna` (لا غنة) | `no_ghonna` | Ghonna for merging noon with Lam and Raa for Hafs |
| meem_aal_imran | ميم آل عمران | `waqf` (وقف), `wasl_2` (فتح الميم ومدها حركتين), `wasl_6` (فتح الميم ومدها ستة حركات) | `waqf` | Ways to recite {الم الله} - waqf: pause with 6 harakat, wasl_2: 2 harakat, wasl_6: 6 harakat |
| madd_yaa_alayn_alharfy | مقدار المد اللازم الحرفي للعين | `2`, `4`, `6` | `6` | Length of Lzem Harfy of Yaa in letter Al-Ayen Madd |
| saken_before_hamz | الساكن قبل الهمز | `tahqeek` (تحقيق), `general_sakt` (سكت عام), `local_sakt` (سكت خاص) | `tahqeek` | Ways of Hafs for saken before hamz |
| sakt_iwaja | السكت عند عوجا في الكهف | `sakt` (سكت), `waqf` (وقف), `idraj` (إدراج) | `waqf` | Ways to recite عوجا (Iwaja) in Surah Al-Kahf |
| sakt_marqdena | السكت عند مرقدنا في يس | `sakt` (سكت), `waqf` (وقف), `idraj` (إدراج) | `waqf` | Ways to recite مرقدنا (Marqadena) in Surat Yassen |
| sakt_man_raq | السكت عند من راق في القيامة | `sakt` (سكت), `waqf` (وقف), `idraj` (إدراج) | `sakt` | Ways to recite من راق (Man Raq) in Surat Al Qiyama |
| sakt_bal_ran | السكت عند بل ران في المطففين | `sakt` (سكت), `waqf` (وقف), `idraj` (إدراج) | `sakt` | Ways to recite بل ران (Bal Ran) in Surat Al Motaffin |
| sakt_maleeyah | وجه قوله تعالى {ماليه هلك} بالحاقة | `sakt` (سكت), `waqf` (وقف), `idgham` (إدغام) | `waqf` | Ways to recite ماليه هلك in Surah Al-Ahqaf |
| between_anfal_and_tawba | وجه بين الأنفال والتوبة | `waqf` (وقف), `sakt` (سكت), `wasl` (وصل) | `waqf` | Ways to recite end of Surah Al-Anfal and beginning of Surah At-Tawbah |
| noon_and_yaseen | الإدغام والإظهار في النون | `izhar` (إظهار), `idgham` (إدغام) | `izhar` | Whether to merge noon of {يس} and {ن} with و or not |
| yaa_athan | إثبات الياء وحذفها وقفا | `wasl` (وصل), `hadhf` (حذف), `ithbat` (إثبات) | `wasl` | Affirmation/omission of Yaa in pause of {آتاني} in Surah An-Naml |
| start_with_ism | وجه البدأ بكلمة {الاسم} في الحجرات | `wasl` (وصل), `lism` (لسم), `alism` (ألسم) | `wasl` | Ruling on starting with {الاسم} in Surah Al-Hujurat |
| yabsut | السين والصاد في {والله يقبض ويبسط} | `seen` (سين), `saad` (صاد) | `seen` | Pronunciation in Surah Al-Baqarah verse |
| bastah | السين والصاد في {وزادكم في الخلق بسطة} | `seen` (سين), `saad` (صاد) | `seen` | Pronunciation in Surah Al-A'raf verse |
| almusaytirun | السين والصاد في {أم هم المصيطرون} | `seen` (سين), `saad` (صاد) | `saad` | Pronunciation in Surah At-Tur |
| bimusaytir | السين والصاد في {لست عليهم بمصيطر} | `seen` (سين), `saad` (صاد) | `saad` | Pronunciation in Surah Al-Ghashiyah |
| tasheel_or_madd | همزة الوصل | `tasheel` (تسهيل), `madd` (مد) | `madd` | Tasheel of Madd for {آلذكرين}, {آلآن}, {آلله} |
| yalhath_dhalik | الإدغام في {يلهث ذلك} | `izhar` (إظهار), `idgham` (إدغام), `waqf` (وقف) | `idgham` | Assimilation in Surah Al-A'raf verse |
| irkab_maana | الإدغام في {اركب معنا} | `izhar` (إظهار), `idgham` (إدغام), `waqf` (وقف) | `idgham` | Assimilation in Surah Hud verse |
| noon_tamnna | الإشمام والروم في {لا تأمنا على يوسف} | `ishmam` (إشمام), `rawm` (روم) | `ishmam` | Nasalization in Surah Yusuf verse |
| harakat_daaf | حركة الضاد في {ضعف} | `fath` (فتح), `dam` (ضم) | `fath` | Vowel of Dhad in Surah Ar-Rum |
| alif_salasila | الألف في {سلاسلا} | `hadhf` (حذف), `ithbat` (إثبات), `wasl` (وصل) | `wasl` | Affirmation/omission of Alif in Surah Al-Insan |
| idgham_nakhluqkum | إدغام القاف في الكاف | `idgham_kamil` (إدغام كامل), `idgham_naqis` (إدغام ناقص) | `idgham_kamil` | Assimilation of Qaf into Kaf in Surah Al-Mursalat |
| raa_firq | راء {فرق} في الشعراء وصلا | `waqf` (وقف), `tafkheem` (تفخيم), `tarqeeq` (ترقيق) | `tafkheem` | Emphasis/softening of Ra in Surah Ash-Shu'ara' |
| raa_alqitr | راء {القطر} في سبأ وقفا | `wasl` (وصل), `tafkheem` (تفخيم), `tarqeeq` (ترقيق) | `wasl` | Emphasis/softening of Ra in Surah Saba' |
| raa_misr | راء {مصر} في يونس وقفا | `wasl` (وصل), `tafkheem` (تفخيم), `tarqeeq` (ترقيق) | `wasl` | Emphasis/softening of Ra in Surah Yunus |
| raa_nudhur | راء {نذر} في القمر وقفا | `wasl` (وصل), `tafkheem` (تفخيم), `tarqeeq` (ترقيق) | `tafkheem` | Emphasis/softening of Ra in Surah Al-Qamar |
| raa_yasr | راء {يسر} بالفجر وقفا | `wasl` (وصل), `tafkheem` (تفخيم), `tarqeeq` (ترقيق) | `tarqeeq` | Emphasis/softening of Ra in Surah Al-Fajr |
| meem_mokhfah | هل الميم مخفاة أو مدغمة | `meem` (ميم), `ikhfaa` (إخفاء) | `ikhfaa` | Whether Meem is hidden or merged in Ikhfaa |
""",
    version="0.0.3",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

_search_executor: Optional[ThreadPoolExecutor] = None
_phonetic_search: Optional[PhoneticSearch] = None
_phonetization_executor: Optional[ThreadPoolExecutor] = None


def get_search_executor() -> ThreadPoolExecutor:
    global _search_executor
    if _search_executor is None:
        _search_executor = ThreadPoolExecutor(
            max_workers=app_settings.max_workers_phonetic_search
        )
    return _search_executor


def get_phonetization_executor() -> ThreadPoolExecutor:
    global _phonetization_executor
    if _phonetization_executor is None:
        _phonetization_executor = ThreadPoolExecutor(
            max_workers=app_settings.max_workers_phonetization
        )
    return _phonetization_executor


def get_phonetic_search() -> PhoneticSearch:
    global _phonetic_search
    if _phonetic_search is None:
        _phonetic_search = PhoneticSearch()
    return _phonetic_search


def tajweed_rule_to_app(rule) -> TajweedRuleApp:
    return TajweedRuleApp(
        name=TajweedRuleNameApp(ar=rule.name.ar, en=rule.name.en),
        golden_len=rule.golden_len,
        correctness_type=rule.correctness_type,
        tag=rule.tag,
    )


async def call_engine_predict(audio_file: UploadFile) -> str:
    audio_bytes = await audio_file.read()
    async with httpx.AsyncClient(timeout=30.0) as client:
        files = {"request": ("audio.wav", audio_bytes, "audio/wav")}
        response = await client.post(app_settings.engine_url, files=files)
        response.raise_for_status()
        data = response.json()
        return data["phonemes"]


def run_phonetic_search(
    phonemes: str, error_ratio: float
) -> tuple[list[SearchResultResponse], str | None]:
    ph_search = get_phonetic_search()
    try:
        results = ph_search.search(phonemes, error_ratio=error_ratio)
    except NoPhonemesSearchResult:
        return [], "No results found. Try increasing error_ratio."

    response_results = []
    for r in results:
        uthmani_text = ph_search.get_uthmani_from_result(r)
        response_results.append(
            SearchResultResponse(
                start=PhonemesSearchSpanApp(
                    sura_idx=r.start.sura_idx,
                    aya_idx=r.start.aya_idx,
                    uthmani_word_idx=r.start.uthmani_word_idx,
                    uthmani_char_idx=r.start.uthmani_char_idx,
                    phonemes_idx=r.start.phonemes_idx,
                ),
                end=PhonemesSearchSpanApp(
                    sura_idx=r.end.sura_idx,
                    aya_idx=r.end.aya_idx,
                    uthmani_word_idx=r.end.uthmani_word_idx,
                    uthmani_char_idx=r.end.uthmani_char_idx,
                    phonemes_idx=r.end.phonemes_idx,
                ),
                uthmani_text=uthmani_text,
            )
        )
    return response_results, None


def run_phonetization_and_error(
    uthmani_text: str,
    moshaf: MoshafAttributes,
    predicted_phonemes: str,
) -> tuple[str, list[ReciterErrorResponse]]:
    ref_phonetization = quran_phonetizer(uthmani_text, moshaf, remove_spaces=True)

    errors = explain_error(
        uthmani_text=uthmani_text,
        ref_ph_text=ref_phonetization.phonemes,
        predicted_ph_text=predicted_phonemes,
        mappings=ref_phonetization.mappings,
    )

    error_responses = []
    for err in errors:
        error_responses.append(
            ReciterErrorResponse(
                uthmani_pos=err.uthmani_pos,
                ph_pos=err.ph_pos,
                error_type=err.error_type,
                speech_error_type=err.speech_error_type,
                expected_ph=err.expected_ph,
                preditected_ph=err.preditected_ph,
                expected_len=err.expected_len,
                predicted_len=err.predicted_len,
                ref_tajweed_rules=[
                    tajweed_rule_to_app(r) for r in err.ref_tajweed_rules
                ]
                if err.ref_tajweed_rules
                else None,
                inserted_tajweed_rules=[
                    tajweed_rule_to_app(r) for r in err.inserted_tajweed_rules
                ]
                if err.inserted_tajweed_rules
                else None,
                replaced_tajweed_rules=[
                    tajweed_rule_to_app(r) for r in err.replaced_tajweed_rules
                ]
                if err.replaced_tajweed_rules
                else None,
                missing_tajweed_rules=[
                    tajweed_rule_to_app(r) for r in err.missing_tajweed_rules
                ]
                if err.missing_tajweed_rules
                else None,
            )
        )

    return ref_phonetization.phonemes, error_responses


@app.get(
    "/health",
    tags=["Health"],
    summary="Health Check",
    description="""Check the health status of the Quran Muaalem App and Engine.

Returns:
- **status**: "healthy" if app is running, "unhealthy" if there are issues
- **engine_connected**: Whether the engine is reachable at the configured URL

The endpoint performs a connection check to the engine if it's configured.
""",
)
async def health():
    """Health check endpoint."""
    import httpx

    engine_status = "unknown"
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(
                app_settings.engine_url.replace("/predict", "/health")
            )
            if response.status_code == 200:
                engine_status = "connected"
            else:
                engine_status = "disconnected"
    except Exception:
        engine_status = "disconnected"

    return {
        "status": "healthy",
        "engine_status": engine_status,
    }


@app.post(
    "/search",
    response_model=SearchResponse,
    tags=["Search"],
    summary="Search Quran by Audio or Phonetic Text",
    description="""Search the Quran using audio input or phonetic text.

This endpoint transcribes audio to phonemes (via the engine) and searches the Quran
for matching locations using fuzzy string matching.

## Input Parameters

- **file**: Audio file (WAV recommended) - will be transcribed to phonemes via the engine
- **phonetic_text**: Direct phonetic text input (skip audio transcription)
- **error_ratio**: Maximum allowed Levenshtein distance as a fraction of query length (0.0-1.0)

## Response

Returns the transcribed phonemes and a list of matching locations in the Quran
with their Uthmani text.

## Example

```bash
# Search by audio file
curl -X POST "http://localhost:8001/search?error_ratio=0.1" \\
    -F "file=@audio.wav"

# Search by phonetic text
curl -X POST "http://localhost:8001/search?phonetic_text=bismi&error_ratio=0.1"
```
""",
)
async def search(
    file: UploadFile = File(
        default=None,
        description="Audio file (WAV recommended) to transcribe and search",
    ),
    phonetic_text: str = Query(
        default=None,
        description="Direct phonetic text input (alternative to audio file)",
    ),
    error_ratio: float = Query(
        default=app_settings.error_ratio,
        description="Maximum allowed error ratio (0.0-1.0), defaults to app setting",
    ),
):
    if error_ratio is None:
        error_ratio = app_settings.error_ratio

    if file:
        phonemes = await call_engine_predict(file)
    elif phonetic_text:
        phonemes = phonetic_text
    else:
        raise HTTPException(
            status_code=422, detail="Either 'file' or 'phonetic_text' must be provided"
        )

    loop = asyncio.get_event_loop()
    results, message = await loop.run_in_executor(
        get_search_executor(),
        run_phonetic_search,
        phonemes,
        error_ratio,
    )

    return SearchResponse(phonemes=phonemes, results=results, message=message)


@app.post(
    "/correct-recitation",
    response_model=CorrectRecitationResponse,
    responses={
        404: {
            "model": CorrectRecitationNoMatchResponse,
            "description": "Predicted phonemes generated, but no Quran match found",
        }
    },
    tags=["Recitation"],
    summary="Analyze and Correct Recitation",
    description="""Analyze audio recording and detect recitation errors with Tajweed rule violations.

This endpoint:
1. Transcribes audio to phonemes (via the engine)
2. Searches for the best matching location in the Quran
3. Generates reference phonetic transcription using the specified MoshafAttributes
4. Compares predicted vs reference phonemes to identify errors

## Input Parameters

- **file**: Audio file (WAV recommended) to analyze
- **phonetic_text**: Direct phonetic text input (alternative to audio)
- **moshaf**: MoshafAttributes form fields defining recitation rules (see API docs for full list)
- **error_ratio**: Maximum allowed error ratio for search (0.0-1.0)

## MoshafAttributes (Recitation Rules)

All fields are optional and default to Hafs recitation standard values:

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| rewaya | string | "hafs" | Quran recitation style |
| recitation_speed | string | "murattal" | Speed: mujawad, above_murattal, murattal, hadr |
| takbeer | string | "no_takbeer" | Takbeer placement: no_takbeer, beginning_of_sharh, end_of_doha, general_takbeer |
| madd_monfasel_len | int | 4 | Mad Monfasel length: 2,3,4,5 |
| madd_mottasel_len | int | 4 | Mad Mottasel length: 4,5,6 |
| madd_mottasel_waqf | int | 4 | Mad Mottasel at pause: 4,5,6 |
| madd_aared_len | int | 4 | Mad Aared length: 2,4,6 |
| madd_alleen_len | int | null | Mad Al-Leen length: 2,4,6 (defaults to madd_aared_len) |
| ghonna_lam_and_raa | string | "no_ghonna" | Ghonna: ghonna, no_ghonna |
| meem_aal_imran | string | "waqf" | Meem in {الم الله}: waqf, wasl_2, wasl_6 |
| madd_yaa_alayn_alharfy | int | 6 | Madd Harfy Yaa: 2,4,6 |
| saken_before_hamz | string | "tahqeek" | Saken before Hamz: tahqeek, general_sakt, local_sakt |
| sakt_iwaja | string | "waqf" | Sakt in عوجا: sakt, waqf, idraj |
| sakt_marqdena | string | "waqf" | Sakt in مرقدنا: sakt, waqf, idraj |
| sakt_man_raq | string | "sakt" | Sakt in من راق: sakt, waqf, idraj |
| sakt_bal_ran | string | "sakt" | Sakt in بل ران: sakt, waqf, idraj |
| sakt_maleeyah | string | "waqf" | Sakt in ماليه هلك: sakt, waqf, idgham |
| between_anfal_and_tawba | string | "waqf" | Between Anfal/Tawba: waqf, sakt, wasl |
| noon_and_yaseen | string | "izhar" | Noon in يس/ن: izhar, idgham |
| yaa_athan | string | "wasl" | Yaa in آتاني: wasl, hadhf, ithbat |
| start_with_ism | string | "wasl" | Start with الاسم: wasl, lism, alism |
| yabsut | string | "seen" | Seen/Saad in يبسط: seen, saad |
| bastah | string | "seen" | Seen/Saad in بسطة: seen, saad |
| almusaytirun | string | "saad" | Seen/Saad in المصيطرون: seen, saad |
| bimusaytir | string | "saad" | Seen/Saad in بمصيطر: seen, saad |
| tasheel_or_madd | string | "madd" | Hamzat Al-Wasl: tasheel, madd |
| yalhath_dhalik | string | "idgham" | Idgham in يلهث: izhar, idgham, waqf |
| irkab_maana | string | "idgham" | Idgham in اركب: izhar, idgham, waqf |
| noon_tamnna | string | "ishmam" | Ishkam/Rawm in تأمنا: ishmam, rawm |
| harakat_daaf | string | "fath" | Dhad vowel in ضعف: fath, dam |
| alif_salasila | string | "wasl" | Alif in سلاسلا: hadhf, ithbat, wasl |
| idgham_nakhluqkum | string | "idgham_kamil" | Idgham Qaf/Kaf: idgham_kamil, idgham_naqis |
| raa_firq | string | "tafkheem" | Ra in فرق: waqf, tafkheem, tarqeeq |
| raa_alqitr | string | "wasl" | Ra in القطر: wasl, tafkheem, tarqeeq |
| raa_misr | string | "wasl" | Ra in مصر: wasl, tafkheem, tarqeeq |
| raa_nudhur | string | "tafkheem" | Ra in نذر: wasl, tafkheem, tarqeeq |
| raa_yasr | string | "tarqeeq" | Ra in يسر: wasl, tafkheem, tarqeeq |
| meem_mokhfah | string | "ikhfaa" | Meem: meem, ikhfaa |

## Response

Returns:
- **start/end**: Position in Quran (sura, aya, word, char indices)
- **predicted_phonemes**: Phonemes from audio
- **reference_phonemes**: Generated reference phonemes using MoshafAttributes
- **uthmani_text**: Matched Quranic text
- **errors**: List of detected errors with:
  - Error type (tajweed, normal, tashkeel)
  - Speech error type (insert, delete, replace)
  - Position in both texts
  - Expected vs predicted phonemes
  - Tajweed rules involved

## Example

```bash
curl -X POST "http://localhost:8001/correct-recitation" \\
    -F "file=@recitation.wav" \\
    -F "rewaya=hafs" \\
    -F "recitation_speed=murattal" \\
    -F "madd_monfasel_len=4"
```
""",
)
async def correct_recitation(
    file: Annotated[
        UploadFile, File(description="Audio file (WAV recommended) to analyze")
    ],
    phonetic_text: str = Form(
        default="", description="Direct phonetic text input (alternative to audio)"
    ),
    moshaf: MoshafAttributes = Depends(correct_recitation_form_dependency()),
    error_ratio: Annotated[float, Form(ge=0.0, le=1)] = app_settings.error_ratio,
):
    if file:
        predicted_phonemes = await call_engine_predict(file)
    elif phonetic_text:
        predicted_phonemes = phonetic_text
    else:
        raise HTTPException(
            status_code=422, detail="Either 'file' or 'phonetic_text' must be provided"
        )

    loop = asyncio.get_event_loop()

    search_results, message = await loop.run_in_executor(
        get_search_executor(),
        run_phonetic_search,
        predicted_phonemes,
        error_ratio,
    )

    if not search_results:
        return JSONResponse(
            status_code=404,
            content=CorrectRecitationNoMatchResponse(
                predicted_phonemes=predicted_phonemes,
                message=message or "No results found. Try increasing error_ratio.",
            ).model_dump(),
        )

    best_result = search_results[0]

    reference_phonemes, errors = await loop.run_in_executor(
        get_phonetization_executor(),
        run_phonetization_and_error,
        best_result.uthmani_text,
        moshaf,
        predicted_phonemes,
    )

    return CorrectRecitationResponse(
        start=best_result.start,
        end=best_result.end,
        predicted_phonemes=predicted_phonemes,
        reference_phonemes=reference_phonemes,
        uthmani_text=best_result.uthmani_text,
        errors=errors,
    )


@app.post(
    "/transcript",
    response_model=TranscriptResponse,
    tags=["Transcript"],
    summary="Transcribe Audio to Phonetic Script",
    description="""Transcribe audio recording to Quranic phonetic script.

This is a proxy endpoint that forwards the audio to the engine and returns
the phoneme transcription. Use this when you only need the phonetic output
without search or error analysis.

## Input Parameters

- **file**: Audio file (WAV recommended) to transcribe

## Response

Returns:
- **phonemes**: Phonetic transcription of the audio
- **sifat**: Currently always null (attributes not implemented)

## Example

```bash
curl -X POST "http://localhost:8001/transcript" \\
    -F "file=@recitation.wav"
```
""",
)
async def transcript(
    file: UploadFile = File(
        ..., description="Audio file (WAV recommended) to transcribe"
    ),
):
    """Transcribe audio to phonetic script (proxy to engine)."""
    phonemes = await call_engine_predict(file)
    return TranscriptResponse(phonemes=phonemes, sifat=None)
