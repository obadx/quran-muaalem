"""`max_audio_seconds` is a limit, not a silent trim.

Needs the `engine` extra (fastapi, litserve); skipped where it is missing.
"""

import io
import wave

import numpy as np
import pytest

pytest.importorskip("litserve")
fastapi = pytest.importorskip("fastapi")

from quran_muaalem.engine.serve import QuranMuaalemAPI

SAMPLING_RATE = 16000
MAX_AUDIO_SECONDS = 5.0


def make_wav_bytes(seconds: float) -> bytes:
    """Silent mono wav of the requested length, in memory."""
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(SAMPLING_RATE)
        wav.writeframes(np.zeros(round(seconds * SAMPLING_RATE), dtype=np.int16).tobytes())
    return buffer.getvalue()


class FakeUpload:
    def __init__(self, data: bytes):
        self.file = io.BytesIO(data)


class FakeAPI:
    """`decode_request` reads only these, so the model is not needed here."""

    max_audio_seconds = MAX_AUDIO_SECONDS
    sampling_rate = SAMPLING_RATE
    max_features = 100

    decode_request = QuranMuaalemAPI.decode_request

    def processor(self, audio_array, **kwargs):
        return {"input_features": audio_array, "attention_mask": None}


@pytest.mark.parametrize("seconds", [1.0, MAX_AUDIO_SECONDS])
def test_audio_within_the_limit_is_kept_whole(seconds):
    out = FakeAPI().decode_request(FakeUpload(make_wav_bytes(seconds)))

    assert len(out["input_features"]) == round(seconds * SAMPLING_RATE)


@pytest.mark.parametrize("seconds", [MAX_AUDIO_SECONDS + 0.5, MAX_AUDIO_SECONDS * 2])
def test_audio_over_the_limit_is_rejected(seconds):
    with pytest.raises(fastapi.HTTPException) as excinfo:
        FakeAPI().decode_request(FakeUpload(make_wav_bytes(seconds)))

    assert excinfo.value.status_code == 413
