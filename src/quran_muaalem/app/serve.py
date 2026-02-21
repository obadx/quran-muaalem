import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import httpx
from fastapi import FastAPI, UploadFile, File, Query, Form, Depends
from fastapi.exceptions import HTTPException
from pydantic import ValidationError

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
    ReciterErrorResponse,
    PhonemesSearchSpanApp,
    TajweedRuleApp,
    TajweedRuleNameApp,
    TranscriptResponse,
)

# TODO:
"""
* [ ] Add timeout for both PHonmems threadpool and correct
* [ ] For both srearch and correct make the file is optional input with input phonems dirctorly
* [ ] Add transcribe end point as a proxy for the predict one
"""


app_settings = AppSettings()

app = FastAPI(title="Quran Muaalem Search API")

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
        files = {"file": ("audio.wav", audio_bytes, "audio/wav")}
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
    print(moshaf.madd_aared_len)
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


def parse_moshaf_from_form(moshaf: str = Form(...)) -> MoshafAttributes:
    try:
        return MoshafAttributes.model_validate_json(moshaf, strict=True)
    except ValidationError as e:
        raise HTTPException(status_code=422, detail=e.errors())


@app.post("/search", response_model=SearchResponse)
async def search(
    file: UploadFile = File(default=None),
    phonetic_text: str = Query(default=None),
    error_ratio: float = Query(default=None),
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


@app.post("/correct-recitation", response_model=CorrectRecitationResponse)
async def correct_recitation(
    file: UploadFile = File(...),
    moshaf: MoshafAttributes = Depends(parse_moshaf_from_form),
    error_ratio: float = Form(default=app_settings.error_ratio),
):
    print(moshaf.madd_aared_len)
    predicted_phonemes = await call_engine_predict(file)

    loop = asyncio.get_event_loop()

    search_results, message = await loop.run_in_executor(
        get_search_executor(),
        run_phonetic_search,
        predicted_phonemes,
        error_ratio,
    )

    if not search_results:
        raise ValueError(message or "No results found. Try increasing error_ratio.")

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


@app.post("/transcript", response_model=TranscriptResponse)
async def transcript(
    file: UploadFile = File(...),
):
    """Transcribe audio to phonetic script (proxy to engine)."""
    phonemes = await call_engine_predict(file)
    return TranscriptResponse(phonemes=phonemes, sifat=None)
