import asyncio
import time
import argparse
import json

import httpx

DEFAULT_URL = "http://localhost:8001"
DEFAULT_FILE = "assets/002282_15.wav"
DEFAULT_REQUESTS = 100
DEFAULT_CONCURRENCY = 100
DEFAULT_ERROR_RATIO = 0.1

DEFAULT_MOSHAF = {
    "rewaya": "hafs",
    "madd_monfasel_len": 4,
    "madd_mottasel_len": 4,
    "madd_mottasel_waqf": 4,
    "madd_aared_len": 4,
}


async def send_search_request(
    client: httpx.AsyncClient,
    url: str,
    file_bytes: bytes | None,
    phonetic_text: str | None,
    error_ratio: float,
    idx: int,
):
    """Send a single /search request and return its latency in seconds."""
    if file_bytes is not None:
        files = {"file": ("test.wav", file_bytes, "audio/wav")}
        params = {"error_ratio": error_ratio}
        start = time.perf_counter()
        try:
            resp = await client.post(f"{url}/search", files=files, params=params)
            resp.raise_for_status()
            if idx == 0:
                print("Sample search (file) response:", resp.json())
        except Exception as e:
            print(f"Request {idx} failed: {e}")
            return None
        end = time.perf_counter()
        return end - start
    elif phonetic_text is not None:
        params = {"phonetic_text": phonetic_text, "error_ratio": error_ratio}
        start = time.perf_counter()
        try:
            resp = await client.post(f"{url}/search", params=params)
            resp.raise_for_status()
            if idx == 0:
                print("Sample search (text) response:", resp.json())
        except Exception as e:
            print(f"Request {idx} failed: {e}")
            return None
        end = time.perf_counter()
        return end - start
    else:
        print("Request {idx} failed: No file or phonetic_text provided")
        return None


async def send_correct_recitation_request(
    client: httpx.AsyncClient,
    url: str,
    file_bytes: bytes,
    error_ratio: float,
    moshaf: dict,
    idx: int,
):
    """Send a single /correct-recitation request and return its latency in seconds."""
    files = {"file": ("test.wav", file_bytes, "audio/wav")}
    data = {"moshaf": json.dumps(moshaf), "error_ratio": str(error_ratio)}
    start = time.perf_counter()
    try:
        resp = await client.post(
            f"{url}/correct-recitation", files=files, data=data
        )
        resp.raise_for_status()
        if idx == 0:
            print("Sample correct-recitation response:", resp.json())
    except Exception as e:
        print(f"Request {idx} failed: {e}")
        return None
    end = time.perf_counter()
    return end - start


async def send_transcript_request(
    client: httpx.AsyncClient,
    url: str,
    file_bytes: bytes,
    idx: int,
):
    """Send a single /transcript request and return its latency in seconds."""
    files = {"file": ("test.wav", file_bytes, "audio/wav")}
    start = time.perf_counter()
    try:
        resp = await client.post(f"{url}/transcript", files=files)
        resp.raise_for_status()
        if idx == 0:
            print("Sample transcript response:", resp.json())
    except Exception as e:
        print(f"Request {idx} failed: {e}")
        return None
    end = time.perf_counter()
    return end - start


async def load_test(
    url: str,
    file_path: str,
    total_requests: int,
    concurrency: int,
    error_ratio: float,
    endpoint: str,
    moshaf: dict,
    phonetic_text: str | None,
):
    file_bytes = None
    if file_path:
        try:
            with open(file_path, "rb") as f:
                file_bytes = f.read()
        except FileNotFoundError:
            print(f"Error: File {file_path} not found.")
            return

    limits = httpx.Limits(
        max_keepalive_connections=concurrency, max_connections=concurrency
    )

    async def run_search():
        async with httpx.AsyncClient(limits=limits, timeout=30.0) as client:
            semaphore = asyncio.Semaphore(concurrency)

            async def bounded_send(idx):
                async with semaphore:
                    return await send_search_request(
                        client, url, file_bytes, phonetic_text, error_ratio, idx
                    )

            tasks = [bounded_send(i) for i in range(total_requests)]
            return await asyncio.gather(*tasks)

    async def run_correct_recitation():
        if file_bytes is None:
            print("Error: file is required for correct-recitation endpoint")
            return [None] * total_requests
        fb: bytes = file_bytes
        async with httpx.AsyncClient(limits=limits, timeout=30.0) as client:
            semaphore = asyncio.Semaphore(concurrency)

            async def bounded_send(idx):
                async with semaphore:
                    return await send_correct_recitation_request(
                        client, url, fb, error_ratio, moshaf, idx
                    )

            tasks = [bounded_send(i) for i in range(total_requests)]
            return await asyncio.gather(*tasks)

    async def run_transcript():
        if file_bytes is None:
            print("Error: file is required for transcript endpoint")
            return [None] * total_requests
        fb: bytes = file_bytes
        async with httpx.AsyncClient(limits=limits, timeout=30.0) as client:
            semaphore = asyncio.Semaphore(concurrency)

            async def bounded_send(idx):
                async with semaphore:
                    return await send_transcript_request(client, url, fb, idx)

            tasks = [bounded_send(i) for i in range(total_requests)]
            return await asyncio.gather(*tasks)

    async def run_all():
        search_task = asyncio.create_task(run_search())
        correct_task = asyncio.create_task(run_correct_recitation())
        transcript_task = asyncio.create_task(run_transcript())
        (
            search_latencies,
            correct_latencies,
            transcript_latencies,
        ) = await asyncio.gather(search_task, correct_task, transcript_task)
        return {
            "search": search_latencies,
            "correct-recitation": correct_latencies,
            "transcript": transcript_latencies,
        }

    print(
        f"Starting load test: {total_requests} requests, concurrency {concurrency}, endpoint: {endpoint}"
    )
    start_time = time.perf_counter()

    if endpoint == "all":
        results = await run_all()
    elif endpoint == "search":
        latencies = await run_search()
        results = {"search": latencies}
    elif endpoint == "correct-recitation":
        latencies = await run_correct_recitation()
        results = {"correct-recitation": latencies}
    else:
        latencies = await run_transcript()
        results = {"transcript": latencies}

    end_time = time.perf_counter()

    total_success = 0
    total_failed = 0

    for ep_name, latencies in results.items():
        successful_latencies = [lat for lat in latencies if lat is not None]
        success_count = len(successful_latencies)
        failed_count = total_requests - success_count

        total_success += success_count
        total_failed += failed_count

        if success_count == 0:
            print(f"\n{ep_name}: No successful requests.")
            continue

        min_lat = min(successful_latencies)
        max_lat = max(successful_latencies)
        avg_lat = sum(successful_latencies) / success_count

        print(f"\n--- {ep_name} Results ---")
        print(f"Successful requests: {success_count}")
        print(f"Failed requests: {failed_count}")
        print(
            f"Latency (seconds) - min: {min_lat:.4f}, max: {max_lat:.4f}, avg: {avg_lat:.4f}"
        )

    total_time = end_time - start_time
    throughput = total_success / total_time if total_time > 0 else 0

    print(f"\n--- Combined Results ---")
    print(f"Total successful: {total_success}")
    print(f"Total failed: {total_failed}")
    print(f"Total time: {total_time:.2f} s")
    print(f"Throughput: {throughput:.2f} req/s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Load test for QuranMuaalem Search API"
    )
    parser.add_argument("--url", type=str, default=DEFAULT_URL, help="API base URL")
    parser.add_argument(
        "--file", type=str, default=DEFAULT_FILE, help="Path to audio file"
    )
    parser.add_argument(
        "--requests",
        "-n",
        type=int,
        default=DEFAULT_REQUESTS,
        help="Number of requests",
    )
    parser.add_argument(
        "--concurrency",
        "-c",
        type=int,
        default=DEFAULT_CONCURRENCY,
        help="Max concurrent requests",
    )
    parser.add_argument(
        "--error-ratio",
        "-e",
        type=float,
        default=DEFAULT_ERROR_RATIO,
        help="Error ratio for phonetic search",
    )
    parser.add_argument(
        "--endpoint",
        "-E",
        choices=["search", "correct-recitation", "transcript", "all"],
        default="all",
        help="API endpoint to test",
    )
    parser.add_argument(
        "--phonetic-text",
        "-t",
        type=str,
        default=None,
        help="Phonetic text for search endpoint (instead of file)",
    )
    args = parser.parse_args()

    asyncio.run(
        load_test(
            args.url,
            args.file,
            args.requests,
            args.concurrency,
            args.error_ratio,
            args.endpoint,
            DEFAULT_MOSHAF,
            args.phonetic_text,
        )
    )
