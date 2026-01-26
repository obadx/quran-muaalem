import torch
import torchaudio
import json
from typing import Optional
import polars as pl 

def prepare_waveform_data(audio, timestamps, sample_rate=16000, step=640):
    """Prepare audio and phoneme data for visualization.

    Args:
        audio: Audio array
        timestamps: List of phoneme timestamp dicts
        sample_rate: Audio sample rate
        step: Samples per frame

    Returns:
        audio_df: Polars DataFrame with time and amplitude
        boundaries_df: Polars DataFrame with phoneme boundaries
    """
    # Audio waveform - downsample for plotting
    downsample = 100  # take every 100th sample
    audio_df = pl.DataFrame(
        {
            "sample": range(0, len(audio), downsample),
            "amplitude": audio[::downsample].tolist(),
        }
    ).with_columns((pl.col("sample") / sample_rate).alias("time_sec"))

    # Phoneme boundaries
    boundaries = []
    for ts in timestamps:
        start_time = (ts["start_frame"] * step) / sample_rate
        end_time = (ts["end_frame"] * step) / sample_rate
        boundaries.append(
            {
                "phoneme": ts["phoneme"],
                "start_time": start_time,
                "end_time": end_time,
                "duration": end_time - start_time,
            }
        )

    boundaries_df = pl.DataFrame(boundaries)

    return audio_df, boundaries_df


def get_moshaf(moshaf_id="moshaf_0.0", streaming=True):
    """Get moshaf from moshaf_id."""
    from datasets import load_dataset

    ds = load_dataset(
        "obadx/muaalem-annotated-v3", "moshaf_0.0", split="train", streaming=True
    )
    return ds


def get_phoneme_logits(model, audio, processor, sample_rate: int = 16000):
    """Get phoneme log probabilities from model.

    Args:
        model: Loaded Wav2Vec2BertForMultilevelCTC model
        audio: Audio array (numpy or tensor)
        processor: Feature extractor
        sample_rate: Audio sample rate

    Returns:
        log_probs: Tensor of shape [T, num_phonemes]
        num_frames: Number of time frames
    """
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    inputs = processor(audio, sampling_rate=sample_rate, return_tensors="pt")
    inputs = {
        k: v.to(device=device, dtype=dtype if k == "input_features" else v.dtype)
        for k, v in inputs.items()
    }

    model.eval()
    with torch.no_grad():
        outputs = model(**inputs, return_dict=False)[0]

    phoneme_logits = outputs["phonemes"]
    log_probs = torch.log_softmax(phoneme_logits, dim=-1)
    log_probs = log_probs.squeeze(0).cpu().float()

    return log_probs, log_probs.shape[0]


def run_forced_alignment(log_probs, token_ids, blank: int = 0):
    """Run forced alignment on log probabilities.

    Args:
        log_probs: Log probabilities [T, C]
        token_ids: Target token IDs (list or tensor)
        blank: Blank token ID for CTC

    Returns:
        alignment: Tensor of aligned token IDs per frame
    """
    targets = torch.tensor(token_ids, dtype=torch.int32)

    alignment, scores = torchaudio.functional.forced_align(
        log_probs.unsqueeze(0), targets.unsqueeze(0), blank=blank
    )
    return alignment.squeeze(0)


def extract_segments(alignment):
    """Extract token segments from alignment.

    Args:
        alignment: Tensor of token IDs per frame

    Returns:
        segments: List of (token_id, start_frame, end_frame)
    """
    alignment_list = alignment.tolist()
    segments = []
    current_token = None
    start_frame = None

    for i, token in enumerate(alignment_list):
        if token != 0 and token != current_token:
            if current_token is not None:
                segments.append((current_token, start_frame, i))
            current_token = token
            start_frame = i
        elif token == 0 and current_token is not None:
            segments.append((current_token, start_frame, i))
            current_token = None

    if current_token is not None:
        segments.append((current_token, start_frame, len(alignment_list)))

    return segments


def segments_to_timestamps(segments, id_to_phoneme):
    """Convert frame segments to timestamps.

    Args:
        segments: List of (token_id, start_frame, end_frame)
        id_to_phoneme: Dict mapping token IDs to phoneme strings

    Returns:
        timestamps: List of {"phoneme", "start_frame", "end_frame"} dicts
    """
    timestamps = []
    for token_id, start, end in segments:
        timestamps.append(
            {
                "phoneme": id_to_phoneme.get(token_id, "?"),
                "start_frame": start,
                "end_frame": end,
            }
        )
    return timestamps


def format_alignment_jsonl(sample, timestamps):
    """Format alignment as JSONL record.

    Args:
        sample: Dataset sample with metadata
        timestamps: List of timestamp dicts

    Returns:
        JSONL string
    """
    record = {
        "moshaf_id": sample["moshaf_id"],
        "reciter_id": sample["reciter_id"],
        "segment_index": sample["segment_index"],
        "phonemes_level": sample["phonemes"],
        "time_stamps": timestamps,
    }
    return json.dumps(record, ensure_ascii=False)


def align_sample(model, processor, tokenizer, sample, sample_rate: int = 16000):
    """Complete pipeline: align one sample and return JSONL.

    Args:
        model: Loaded model
        processor: Feature extractor
        tokenizer: Phonemes tokenizer
        sample: Dataset sample
        sample_rate: Audio sample rate

    Returns:
        JSONL string with alignment
    """
    audio = sample["audio"]["array"]

    log_probs, num_frames = get_phoneme_logits(model, audio, processor, sample_rate)

    phoneme_text = sample["phonemes"].replace(" ", "")
    token_ids = tokenizer.encode(phoneme_text)

    alignment = run_forced_alignment(log_probs, token_ids)
    segments = extract_segments(alignment)

    vocab = tokenizer.get_vocab()
    id_to_phoneme = {v: k for k, v in vocab.items()}
    timestamps = segments_to_timestamps(segments, id_to_phoneme)

    return format_alignment_jsonl(sample, timestamps)


def extract_audio_segment(audio, start_frame, end_frame, sample_rate=16000):
    """Extract audio segment for given frames.

    Args:
        audio: Audio array
        start_frame: Starting frame index
        end_frame: Ending frame index
        sample_rate: Audio sample rate

    Returns:
        audio_segment: Extracted audio array
        start_sample: Start sample index
        end_sample: End sample index
    """
    step = 640
    span = 560

    start_sample = start_frame * step
    end_sample = (end_frame - 1) * step + span

    return audio[start_sample:end_sample], start_sample, end_sample


def test_alignment(
    audio,
    timestamps,
    sample_rate=16000,
    output_dir="test_segments",
    group_size=3,
    num_tests=5,
    padding=0,
):
    """Save audio segments for manual verification.

    Args:
        audio: Full audio array
        timestamps: List of {"phoneme", "start_frame", "end_frame"} dicts
        sample_rate: Audio sample rate
        output_dir: Directory to save segments
        group_size: Number of phonemes per file
        num_tests: Number of groups to save
        padding: Extra samples before/after segment
    """
    import soundfile as sf
    from pathlib import Path

    Path(output_dir).mkdir(exist_ok=True)

    for i in range(0, min(num_tests * group_size, len(timestamps)), group_size):
        group = timestamps[i : i + group_size]
        if not group:
            break

        start_frame = group[0]["start_frame"]
        end_frame = group[-1]["end_frame"]
        phonemes = "".join([g["phoneme"] for g in group])

        segment, start_sample, end_sample = extract_audio_segment(
            audio, start_frame, end_frame, sample_rate
        )

        if padding > 0:
            pad_start = max(0, start_sample - padding)
            pad_end = min(len(audio), end_sample + padding)
            segment = audio[pad_start:pad_end]

        filename = f"{output_dir}/{i // group_size:02d}_{phonemes}_f{start_frame}-{end_frame}.wav"
        sf.write(filename, segment, sample_rate)
        print(f"Saved: {filename}")
