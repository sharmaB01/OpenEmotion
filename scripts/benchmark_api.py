import argparse
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import httpx


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = round((len(ordered) - 1) * q)
    return ordered[index]


def post_frame(client: httpx.Client, url: str, api_key: str, image: bytes, frame_id: str):
    started = time.perf_counter()
    response = client.post(
        f"{url.rstrip('/')}/v1/analyze/frame",
        headers={"Authorization": f"Bearer {api_key}"},
        data={"frame_id": frame_id, "timestamp": "2026-10-06T10:00:00Z"},
        files={"frame": ("frame.jpg", image, "image/jpeg")},
    )
    elapsed = time.perf_counter() - started
    response.raise_for_status()
    return elapsed, response.json()


def main():
    parser = argparse.ArgumentParser(description="Benchmark OpenEmotion HTTP latency")
    parser.add_argument("--url", default="http://127.0.0.1:9000")
    parser.add_argument("--api-key", required=True)
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--requests", type=int, default=100)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=20)
    args = parser.parse_args()

    image = args.image.read_bytes()
    latencies = []
    failures = 0
    started = time.perf_counter()
    with httpx.Client(timeout=args.timeout) as client:
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            futures = [
                pool.submit(
                    post_frame,
                    client,
                    args.url,
                    args.api_key,
                    image,
                    f"bench-{idx}",
                )
                for idx in range(args.requests)
            ]
            for future in as_completed(futures):
                try:
                    elapsed, _ = future.result()
                    latencies.append(elapsed)
                except Exception as exc:  # noqa: BLE001 - count benchmark failures
                    failures += 1
                    print(f"request failed: {type(exc).__name__}: {exc}")

    total = time.perf_counter() - started
    success = len(latencies)
    print(f"requests={args.requests} success={success} failures={failures}")
    print(f"concurrency={args.concurrency} total_seconds={total:.3f}")
    print(f"throughput_fps={success / total if total else 0:.2f}")
    if latencies:
        print(f"latency_avg_ms={statistics.mean(latencies) * 1000:.2f}")
        print(f"latency_p50_ms={percentile(latencies, 0.50) * 1000:.2f}")
        print(f"latency_p95_ms={percentile(latencies, 0.95) * 1000:.2f}")
        print(f"latency_max_ms={max(latencies) * 1000:.2f}")


if __name__ == "__main__":
    main()
