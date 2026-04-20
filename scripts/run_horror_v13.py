#!/usr/bin/env python3
"""
Horror v13 test launcher.
Deploys script to container, runs orchestrator.py via docker exec.
Uses Python subprocess.Popen (not shell) to avoid API key special-char issues.
"""
import os
import subprocess
import sys
import time
from pathlib import Path

CONTAINER = "text_video_multiai_worker_gpu_1"
LOG_PATH = "/tmp/horror_v13.log"
SCRIPT_PATH = "/tmp/horror_v13_script.txt"
JOB_ID = "horror_v13"

HORROR_SCRIPT = """An old Victorian mansion stands alone on a fog-covered hill under a pale full moon.
Twisted iron gates creak open slowly, revealing an overgrown path to the front door.
A shadow moves behind a second-floor window, though the house has been empty for decades.
Inside the dark entrance hall, a grandfather clock ticks backwards, its pendulum swinging in silence.
Dusty portraits line the walls, their painted eyes shifting to follow every movement.
In the master bedroom, a rocking chair sways on its own beside a cold fireplace.
Candles ignite themselves one by one along the hallway, casting long trembling shadows.
A child's laughter echoes from the basement, where the door is sealed with iron chains.
The mirror at the end of the corridor reflects a different room — one that burned long ago.
Footsteps descend the staircase from above, but no one appears on the stairs.
Outside, the fog thickens until the iron gates vanish completely into the white darkness.
The mansion's front door slams shut, and every light inside dies at once.
"""


def load_env(env_file: str) -> dict:
    env = dict(os.environ)
    p = Path(env_file)
    if not p.exists():
        print(f"ERROR: .env not found at {env_file}", file=sys.stderr)
        sys.exit(1)
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        env[key.strip()] = val.strip().strip('"').strip("'")
    return env


def write_script_to_container(env: dict) -> None:
    r = subprocess.run(
        ["docker", "exec", "-i", CONTAINER, "tee", SCRIPT_PATH],
        input=HORROR_SCRIPT.encode(),
        capture_output=True,
        env=env,
    )
    if r.returncode != 0:
        print(f"ERROR writing script to container: {r.stderr.decode()}", file=sys.stderr)
        sys.exit(1)
    print(f"Script written to container: {SCRIPT_PATH}")


def main() -> None:
    env_file = os.path.expanduser("~/Text_Video_MultiAI/.env")
    if not Path(env_file).exists():
        env_file = os.path.expanduser("~/project/.env")

    env = load_env(env_file)

    gemini_key = env.get("Gemini_API_KEY", "")
    hf_token = env.get("HF_TOKEN", "")
    if not gemini_key or not hf_token:
        print("ERROR: Gemini_API_KEY or HF_TOKEN missing in .env", file=sys.stderr)
        sys.exit(1)

    write_script_to_container(env)

    cmd = [
        "docker", "exec",
        "-e", f"Gemini_API_KEY={gemini_key}",
        "-e", f"HF_TOKEN={hf_token}",
        CONTAINER,
        "python3", "/app/orchestrator.py",
        "--script", SCRIPT_PATH,
        "--style", "horror",
        "--strategy", "balanced",
        "--job-id", JOB_ID,
        "--min-clip", "5.0",
        "--max-clip", "5.1",
    ]

    print(f"Launching horror_v13... log → {LOG_PATH}")
    print(f"Strategy: balanced (WAN 14B 848x480 + upscale, 30 steps)")
    print(f"Started: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("-" * 60)

    with open(LOG_PATH, "w") as log_f:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=env,
        )
        for line in proc.stdout:
            decoded = line.decode(errors="replace")
            sys.stdout.write(decoded)
            sys.stdout.flush()
            log_f.write(decoded)
            log_f.flush()

    rc = proc.wait()
    print("-" * 60)
    print(f"Done: {time.strftime('%Y-%m-%d %H:%M:%S')} | exit={rc}")
    print(f"Log saved: {LOG_PATH}")
    sys.exit(rc)


if __name__ == "__main__":
    main()
