#!/usr/bin/env python3
"""
Smoke test: 12 runs (4 models × 3 strategies).
Submit sequentially via the API, poll until done, collect metrics.
"""
import json
import sys
import time
import httpx

API = "http://localhost:8000"

SCRIPT = (
    "A futuristic golden-hour street bathed in warm amber light. Dense crowds of "
    "humans fill the sidewalks — faces illuminated by neon reflections from towering "
    "holographic billboards. Street performers and dancers move through the crowd, "
    "their movements fluid and expressive. Close-ups reveal micro-expressions of "
    "wonder and joy. The camera glides smoothly through the scene, then rises above "
    "the rooftops where fireworks burst against the twilight sky. Cinematic pacing "
    "with narration and ambient music."
)

RUNS = [
    ("WAN13B_FAST",      "wan2_1b",    "fast"),
    ("WAN13B_BALANCED",  "wan2_1b",    "balanced"),
    ("WAN13B_QUALITY",   "wan2_1b",    "quality"),
    ("WAN14B_FAST",      "wan2_14b",   "fast"),
    ("WAN14B_BALANCED",  "wan2_14b",   "balanced"),
    ("WAN14B_QUALITY",   "wan2_14b",   "quality"),
    ("HUNYUAN_FAST",     "hunyuan",    "fast"),
    ("HUNYUAN_BALANCED", "hunyuan",    "balanced"),
    ("HUNYUAN_QUALITY",  "hunyuan",    "quality"),
    ("COGVID_FAST",      "cogvideox",  "fast"),
    ("COGVID_BALANCED",  "cogvideox",  "balanced"),
    ("COGVID_QUALITY",   "cogvideox",  "quality"),
]

client = httpx.Client(base_url=API, timeout=30.0)
results = []


def submit(name, model, strategy):
    payload = {
        "script": SCRIPT,
        "strategy": strategy,
        "style": "cinematic",
        "quality": "high",
        "preferred_model": model,
        "api_fallback": False,
        "transition": "crossfade",
        "min_clip": 5.0,
        "max_clip": 15.0,
        "upscale_4k": False,
        "use_dag": True,
    }
    r = client.post("/generate_video", json=payload)
    r.raise_for_status()
    data = r.json()
    return data["job_id"]


def poll(job_id, timeout=1800):
    start = time.time()
    while time.time() - start < timeout:
        r = client.get(f"/status/{job_id}")
        data = r.json()
        status = data["status"]
        progress = data.get("progress", 0)
        if status in ("done", "completed", "finished"):
            return data, time.time() - start
        if status in ("failed", "error"):
            return data, time.time() - start
        print(f"  [{job_id[:8]}] {status} {progress:.0%}", end="\r", flush=True)
        time.sleep(15)
    return {"status": "timeout"}, time.time() - start


def main():
    print(f"=== Smoke Test: {len(RUNS)} runs ===\n")

    for name, model, strategy in RUNS:
        print(f"▶ {name} (model={model}, strategy={strategy})")
        try:
            job_id = submit(name, model, strategy)
            print(f"  Submitted: {job_id}")
            data, elapsed = poll(job_id)
            status = data.get("status", "unknown")
            error = data.get("error")
            output = data.get("output_path")
            results.append({
                "name": name,
                "model": model,
                "strategy": strategy,
                "job_id": job_id,
                "status": status,
                "elapsed_s": round(elapsed, 1),
                "output": output,
                "error": error,
            })
            print(f"  → {status} in {elapsed:.0f}s | output={output}")
            if error:
                print(f"  ✗ Error: {error[:200]}")
        except Exception as e:
            results.append({
                "name": name,
                "model": model,
                "strategy": strategy,
                "job_id": None,
                "status": "submit_failed",
                "elapsed_s": 0,
                "output": None,
                "error": str(e)[:300],
            })
            print(f"  ✗ Submit failed: {e}")
        print()

    print("\n" + "=" * 70)
    print("SMOKE TEST RESULTS")
    print("=" * 70)
    for r in results:
        flag = "✓" if r["status"] in ("done", "completed", "finished") else "✗"
        print(f"  {flag} {r['name']:20s} | {r['status']:10s} | {r['elapsed_s']:>7.0f}s | {r['output'] or r['error'] or '-'}")
    print("=" * 70)

    # Dump JSON for post-processing
    with open("/app/outputs/smoke_test_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to /app/outputs/smoke_test_results.json")


if __name__ == "__main__":
    main()
