#!/usr/bin/env python3
"""
Smoke test runner — submits 12 model×strategy combinations via the API.
Run inside API container or anywhere with httpx and access to localhost:8000.
"""
import json
import sys
import time
import urllib.request
import urllib.error

API = "http://localhost:8000"

SCRIPT = (
    "Futuristic golden-hour street scene with dense crowds of humans walking. "
    "Neon reflections shimmer on wet pavement. Street performers and dancers "
    "move through the crowd. Holographic advertisements float above. "
    "Close-up shots capture micro-expressions on faces. Camera glides smoothly "
    "through the scene, rises above rooftops. Fireworks burst over the "
    "cityscape with cinematic pacing."
)

RUNS = [
    ("WAN13B_FAST",      "wan2_1b",   "fast"),
    ("WAN13B_BALANCED",  "wan2_1b",   "balanced"),
    ("WAN13B_QUALITY",   "wan2_1b",   "quality"),
    ("WAN14B_FAST",      "wan2_14b",  "fast"),
    ("WAN14B_BALANCED",  "wan2_14b",  "balanced"),
    ("WAN14B_QUALITY",   "wan2_14b",  "quality"),
    ("HUNYUAN_FAST",     "hunyuan",   "fast"),
    ("HUNYUAN_BALANCED", "hunyuan",   "balanced"),
    ("HUNYUAN_QUALITY",  "hunyuan",   "quality"),
    ("COGVID_FAST",      "cogvideox", "fast"),
    ("COGVID_BALANCED",  "cogvideox", "balanced"),
    ("COGVID_QUALITY",   "cogvideox", "quality"),
]


def submit(name, model, strategy):
    payload = json.dumps({
        "script": SCRIPT,
        "strategy": strategy,
        "style": "cinematic",
        "quality": "high",
        "preferred_model": model,
        "min_clip": 10,
        "max_clip": 15,
        "transition": "crossfade",
        "use_dag": True,
    }).encode()
    req = urllib.request.Request(
        f"{API}/generate_video",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read())
            return data.get("job_id")
    except urllib.error.HTTPError as e:
        body = e.read().decode()
        print(f"  ERROR submitting {name}: {e.code} {body}")
        return None
    except Exception as e:
        print(f"  ERROR submitting {name}: {e}")
        return None


def poll(job_id, timeout=1800):
    start = time.time()
    while time.time() - start < timeout:
        try:
            req = urllib.request.Request(f"{API}/status/{job_id}")
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read())
                status = data.get("status", "unknown")
                progress = data.get("progress", 0)
                if status in ("done", "completed", "finished"):
                    return data
                if status in ("failed", "error"):
                    return data
                elapsed = int(time.time() - start)
                print(f"    [{elapsed}s] status={status} progress={progress:.0%}", flush=True)
        except Exception:
            pass
        time.sleep(15)
    return {"status": "timeout", "elapsed": timeout}


def main():
    # If a specific run index is given, only run that one
    indices = range(len(RUNS))
    if len(sys.argv) > 1:
        indices = [int(x) for x in sys.argv[1:]]

    results = {}
    for i in indices:
        name, model, strategy = RUNS[i]
        print(f"\n{'='*60}")
        print(f"RUN {i+1}: {name} (model={model}, strategy={strategy})")
        print(f"{'='*60}", flush=True)

        t0 = time.time()
        job_id = submit(name, model, strategy)
        if not job_id:
            results[name] = {"status": "submit_failed", "latency": 0}
            continue

        print(f"  job_id={job_id}", flush=True)
        result = poll(job_id)
        elapsed = time.time() - t0
        result["latency_s"] = round(elapsed, 1)
        result["name"] = name
        results[name] = result
        print(f"  RESULT: status={result.get('status')} latency={elapsed:.0f}s")

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for name, r in results.items():
        print(f"  {name}: status={r.get('status')} latency={r.get('latency_s', 0)}s")


if __name__ == "__main__":
    main()
