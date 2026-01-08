from __future__ import annotations

import argparse
import asyncio
from time import perf_counter
from typing import List

import httpx
import numpy as np


def _generate_wave(seconds: float, sampling_rate: int) -> List[float]:
    t = np.linspace(0, seconds, int(seconds * sampling_rate), endpoint=False)
    wave = 0.1 * np.sin(2 * np.pi * 220 * t)
    return wave.astype(np.float32).tolist()


async def _post_request(client: httpx.AsyncClient, url: str, payload: dict) -> float:
    start = perf_counter()
    resp = await client.post(url, json=payload)
    latency_ms = (perf_counter() - start) * 1000
    resp.raise_for_status()
    data = resp.json()
    print(f"{url} -> {latency_ms:.2f} ms, decoded preview: {data.get('decoded')}")
    return latency_ms


async def main(args: argparse.Namespace) -> None:
    base = args.base_url.rstrip("/")
    if args.endpoint == "sliding":
        url = f"{base}/inference/sliding-window"
        payload = {
            "audio": _generate_wave(args.seconds, args.sampling_rate),
            "sampling_rate": args.sampling_rate,
            "window_ms": args.window_ms,
            "chunk_ms": args.chunk_ms,
        }
    else:
        url = f"{base}/inference/offline"
        audio = _generate_wave(args.seconds, args.sampling_rate)
        payload = {
            "audios": [audio for _ in range(args.batch_size)],
            "sampling_rate": args.sampling_rate,
        }

    async with httpx.AsyncClient(timeout=args.timeout) as client:
        tasks = [
            asyncio.create_task(_post_request(client, url, payload))
            for _ in range(args.requests)
        ]
        results = await asyncio.gather(*tasks)

    print(
        f"Finished {len(results)} requests -> "
        f"p50={np.percentile(results, 50):.2f} ms, "
        f"p90={np.percentile(results, 90):.2f} ms, "
        f"max={max(results):.2f} ms"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fire concurrent requests against the inference server to observe batching/latency."
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--requests", type=int, default=4)
    parser.add_argument("--endpoint", choices=["sliding", "offline"], default="offline")
    parser.add_argument("--seconds", type=float, default=2.0)
    parser.add_argument("--sampling-rate", type=int, default=16000)
    parser.add_argument("--window-ms", type=int, default=6000)
    parser.add_argument("--chunk-ms", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=120.0)
    args = parser.parse_args()
    asyncio.run(main(args))

