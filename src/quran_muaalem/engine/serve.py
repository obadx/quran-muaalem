import io
import time
from typing import Annotated

import librosa
import torch
import litserve as ls
from transformers import AutoFeatureExtractor
import numpy as np
from fastapi import File, HTTPException, UploadFile

from ..modeling.modeling_multi_level_ctc import Wav2Vec2BertForMultilevelCTC
from ..modeling.multi_level_tokenizer import MultiLevelTokenizer


def simple_ctc_decode(
    batch_arr: list[list[int]], blank_id=0, collapse_consecutive=True
) -> list[list[int]]:
    decoded_list = []
    for seq in batch_arr:
        if collapse_consecutive:
            tokens = []
            prev = blank_id
            for current in seq:
                if current == blank_id:
                    prev = blank_id
                    continue
                if current == prev:
                    continue
                tokens.append(current)
                prev = current
            decoded_list.append(tokens)
        else:
            tokens = seq[seq != blank_id]
            decoded_list.append(tokens)
    return decoded_list


class QuranMuaalemAPI(ls.LitAPI):
    def __init__(
        self,
        model_name_or_path: str = "obadx/muaalem-model-v3_2",
        dtype: torch.dtype = torch.bfloat16,
        max_audio_seconds: float = 15,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.model_name_or_path = model_name_or_path
        self.dtype = dtype
        self.max_audio_seconds = max_audio_seconds
        self.sampling_rate = 16000
        self.max_features = int(
            np.ceil((self.sampling_rate * self.max_audio_seconds - 400) / (160 * 2))
        )
        self.multi_level_tokenizer = MultiLevelTokenizer(self.model_name_or_path)

    def setup(self, device):
        self.device = device
        self.processor = AutoFeatureExtractor.from_pretrained(self.model_name_or_path)
        self.model = Wav2Vec2BertForMultilevelCTC.from_pretrained(
            self.model_name_or_path
        )
        self.model.to(device, dtype=self.dtype)
        self.model.eval()

    def decode_request(self, request: Annotated[UploadFile, File()]):
        # audio_bytes = request  # directly use the bytes
        audio_bytes = request.file.read()

        audio_array, sr = librosa.load(
            io.BytesIO(audio_bytes),
            sr=self.sampling_rate,
            mono=True,
        )

        # Rejecting instead of truncating: a truncated recitation is scored as
        # if the missing tail was never recited, with nothing in the response
        # to say so.
        max_samples = round(self.max_audio_seconds * self.sampling_rate)
        if len(audio_array) > max_samples:
            raise HTTPException(
                status_code=413,
                detail=(
                    f"Input audio is {len(audio_array) / sr:.2f}s long, over the"
                    f" engine limit of {self.max_audio_seconds}s. Split the"
                    " audio and send the parts separately."
                ),
            )

        features = self.processor(
            audio_array,
            sampling_rate=sr,
            return_tensors="pt",
            padding="max_length",
            max_length=self.max_features,
        )

        return {
            "input_features": features["input_features"],
            "attention_mask": features["attention_mask"],
        }

    def batch(self, inputs):
        input_features = torch.cat([inp["input_features"] for inp in inputs]).to(
            self.device, dtype=self.dtype
        )
        attention_mask = torch.cat([inp["attention_mask"] for inp in inputs]).to(
            self.device, dtype=self.dtype
        )
        return (input_features, attention_mask)

    def predict(self, x):
        input_features, attention_mask = x
        with torch.inference_mode():
            level_to_logits = self.model(
                input_features, attention_mask, return_dict=False
            )[0]

        list_of_level_to_logits = []
        for idx in range(level_to_logits["phonemes"].shape[0]):
            d = {}
            for level in level_to_logits:
                d[level] = (
                    level_to_logits[level][idx]
                    .cpu()
                    .to(dtype=torch.float32)
                    .unsqueeze(0)
                )
            list_of_level_to_logits.append(d)

        return list_of_level_to_logits

    def unbatch(self, outputs):
        return outputs

    def encode_response(self, output):
        level_to_logits = output

        level_to_probs = {}
        for level, logits in level_to_logits.items():
            probs = torch.nn.functional.softmax(logits, dim=-1)
            level_to_probs[level] = probs

        phonemes_probs = level_to_probs["phonemes"]
        batch_probs, batch_ids = phonemes_probs.topk(1, dim=-1)

        ph_decoded_ids = simple_ctc_decode(batch_ids)

        # TODO: make it more abstract in  a function
        phonemes_level = ""
        for idx in ph_decoded_ids[0]:
            if idx != 0:
                phonemes_level += self.multi_level_tokenizer.id_to_vocab["phonemes"][
                    int(idx)
                ]

        return {"phonemes": phonemes_level}
