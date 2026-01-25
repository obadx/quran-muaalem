from __future__ import annotations

"""
Simple helper to hit the running inference server with a local WAV file.

Usage:
  1) Start the server (in another shell):
       .venv/bin/uvicorn engine.inference_server:app --host 127.0.0.1 --port 8000
  2) Run this script (from repo root):
       .venv/bin/python engine/test_endpoint.py --wav assets/test.wav --endpoint sliding
"""

import argparse
import json
import urllib.request
from typing import List

import librosa


def load_wav_to_floats(path: str) -> tuple[List[float], int]:
    wave, sr = librosa.load(path, sr=16000, mono=True)
    return wave.astype("float32").tolist(), 16000


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Send a WAV file to the running inference server."
    )
    parser.add_argument("--wav", default="assets/fatiha_long_track.wav", help="Path to WAV file.")
    parser.add_argument(
        "--endpoint",
        choices=["sliding", "offline"],
        default="sliding",
        help="Which API endpoint to call.",
    )
    parser.add_argument(
        "--mode",
        choices=["single", "realtime"],
        default="single",
        help="single = one request; realtime = send sequential chunks",
    )
    parser.add_argument(
        "--realtime-chunk-ms",
        type=int,
        default=1000,
        help="Chunk size for realtime mode (ms).",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default="8000")
    parser.add_argument("--window-ms", type=int, default=6000)
    parser.add_argument("--chunk-ms", type=int, default=300)
    args = parser.parse_args()

    audio, sr = load_wav_to_floats(args.wav)
    if sr != 16000:
        raise SystemExit(f"sampling_rate must be 16000, got {sr}")

    if args.endpoint == "sliding":
        url = f"http://{args.host}:{args.port}/inference/sliding-window"
        if args.mode == "single":
            payload = {
                "audio": audio,
                "sampling_rate": sr,
                "window_ms": args.window_ms,
                "chunk_ms": args.chunk_ms,
            }
            body = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                url,
                data=body,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=120) as resp:
                raw = resp.read().decode("utf-8")
                print("Status:", resp.status)
                print("Response:", raw)
        else:
            # realtime: send sequential chunks as separate payloads
            samples_per_chunk = int(sr * args.realtime_chunk_ms / 1000)
            if samples_per_chunk <= 0:
                raise SystemExit("chunk-ms too small.")
            all_ids = []
            all_text = []
            start = 0
            idx = 0
            while start < len(audio):
                end = min(len(audio), start + samples_per_chunk)
                chunk = audio[start:end]
                payload = {
                    "audio": chunk,
                    "sampling_rate": sr,
                    "window_ms": args.window_ms,
                    "chunk_ms": args.chunk_ms,
                }
                body = json.dumps(payload).encode("utf-8")
                req = urllib.request.Request(
                    url,
                    data=body,
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=120) as resp:
                    raw = resp.read().decode("utf-8")
                    print(f"Chunk {idx} [{start}:{end}] -> {resp.status}")
                    print("Response snippet:", raw[:200])
                    try:
                        data = json.loads(raw)
                        if data.get("phonemes_ids"):
                            all_ids.extend(data["phonemes_ids"])
                        if data.get("decoded"):
                            all_text.append(data["decoded"])
                    except Exception:
                        pass
                start = end
                idx += 1
            print("\n=== Realtime combined ===")
            print("Total phoneme ids:", len(all_ids))
            print("Concatenated decoded:", "".join(all_text))
    else:
        url = f"http://{args.host}:{args.port}/inference/offline"
        payload = {"audio": audio, "sampling_rate": sr}
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            raw = resp.read().decode("utf-8")
            print("Status:", resp.status)
            print("Response:", raw)


if __name__ == "__main__":
    main()
