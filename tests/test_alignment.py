import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"  

import pytest 
import numpy as np 
import json 
import io 
import soundfile as sf 
from quran_muaalem import Muaalem
from quran_muaalem.forced_alignment import align_sample,extract_audio_segment

class TestExtractAudioSegment:
    def test_correct_sample_positions(self):
        audio = np.zeros(66400)
        segment,start,end = extract_audio_segment(audio,18,19)
        assert start ==18*640
        assert end == (19-1) * 640 + 560
    def test_segment_length(self):
        audio = np.random.randn(66400)
        segment,_,_ = extract_audio_segment(audio,18,19)
        assert  len(segment) ==560

class TestAlignSample:
    @pytest.fixture(scope="session")
    def sample_data(self):
        from datasets import load_dataset,Audio

        ds = load_dataset(
            "obadx/muaalem-annotated-v3",
            "moshaf_0.0",
            split="train",
            streaming=True
,
        )
        ds = ds.cast_column("audio",Audio(decode=False))
        sample = next(iter(ds))

        #Load audio from bytes
        audio_bytes = sample["audio"]["bytes"]
        audio_array,sr = sf.read(io.BytesIO(audio_bytes))
        sample["audio"] = {"array":audio_array,"sampling_rate":sr}

        return sample

    @pytest.fixture(scope="session")
    def muaalem(self):
        # The local model is corrupted, use HuggingFace cache
        return Muaalem(model_name_or_path="obadx/muaalem-model-v3_2")

    def test_align_sample_returns_valid_jsonl(self, sample_data, muaalem):
        result = align_sample(
            muaalem.model, 
            muaalem.processor, 
            muaalem.multi_level_tokenizer.get_tokenizer(), 
            sample_data
        )
        
        parsed = json.loads(result)
        assert "moshaf_id" in parsed
        assert "time_stamps" in parsed
        assert len(parsed["time_stamps"]) > 0
        assert "phoneme" in parsed["time_stamps"][0]
        assert "start_frame" in parsed["time_stamps"][0]
        assert "end_frame" in parsed["time_stamps"][0]
        


