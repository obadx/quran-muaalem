from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch

from trt_runtime import build_or_load_trt, DEFAULT_MODEL_ID, DEFAULT_SAMPLING_RATE


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert the Quran Muaalem model to a TensorRT TorchScript engine."
    )
    parser.add_argument(
        "--model-id",
        default=DEFAULT_MODEL_ID,
        help="HF model id or local path.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("engine/muaalem_trt.ts"),
        help="Where to save the compiled TensorRT engine.",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Device to load the model on during compilation.",
    )
    parser.add_argument(
        "--dtype",
        default="float16",
        choices=["float16", "float32", "bfloat16"],
        help="Precision for the TensorRT engine.",
    )
    parser.add_argument(
        "--sampling-rate",
        type=int,
        default=DEFAULT_SAMPLING_RATE,
        help="Audio sampling rate expected by the model.",
    )
    parser.add_argument(
        "--min-seconds",
        type=float,
        default=1.0,
        help="Minimum audio length (seconds) for dynamic shapes.",
    )
    parser.add_argument(
        "--opt-seconds",
        type=float,
        default=3.0,
        help="Optimal audio length (seconds) for dynamic shapes.",
    )
    parser.add_argument(
        "--max-seconds",
        type=float,
        default=6.0,
        help="Maximum audio length (seconds) for dynamic shapes.",
    )
    parser.add_argument(
        "--max-batch-size",
        type=int,
        default=4,
        help="Maximum batch size to support in the TensorRT engine.",
    )
    return parser.parse_args()


def _to_dtype(name: str) -> torch.dtype:
    if name == "float16":
        return torch.float16
    if name == "float32":
        return torch.float32
    if name == "bfloat16":
        return torch.bfloat16
    raise ValueError(f"Unsupported dtype: {name}")


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = _parse_args()
    dtype = _to_dtype(args.dtype)

    resources = build_or_load_trt(
        model_id=args.model_id,
        device=args.device,
        dtype=dtype,
        trt_path=args.output,
        sampling_rate=args.sampling_rate,
        min_seconds=args.min_seconds,
        opt_seconds=args.opt_seconds,
        max_seconds=args.max_seconds,
        max_batch_size=args.max_batch_size,
        force_rebuild=True,
    )

    if resources.using_trt:
        print(f"TensorRT engine saved to: {resources.trt_path}")
    else:
        print(
            "torch_tensorrt was not available; a TorchScript wrapper was prepared for runtime use."
        )


if __name__ == "__main__":
    main()
