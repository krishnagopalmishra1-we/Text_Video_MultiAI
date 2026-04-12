"""
End-to-end pipeline orchestrator — standalone CLI entry point.
Use this to run the full pipeline without the FastAPI/Celery stack.
Ideal for single-machine batch runs on the A100.

Usage:
  python orchestrator.py --script my_script.txt --style cinematic --quality high
  python orchestrator.py --script my_script.txt --upscale-4k --resume
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import time
from pathlib import Path

import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("outputs/pipeline.log"),
    ],
)
logger = logging.getLogger("orchestrator")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Long Video Generator")
    p.add_argument("--script", required=True, help="Script file or inline text")
    p.add_argument("--style", default="cinematic",
                   choices=["cinematic", "documentary", "commercial", "sci_fi", "nature", "dramatic"])
    p.add_argument("--quality", default="high", choices=["high", "fast", "preview"])
    p.add_argument("--pacing", default="normal", choices=["slow", "normal", "fast"])
    p.add_argument("--transition", default="crossfade",
                   choices=["crossfade", "fade_black", "fade_white", "dissolve", "cut"])
    p.add_argument("--min-clip", type=float, default=5.0)
    p.add_argument("--max-clip", type=float, default=20.0)
    p.add_argument("--output", default=None, help="Output video path")
    p.add_argument("--upscale-4k", action="store_true")
    p.add_argument("--resume", action="store_true", default=True,
                   help="Skip already-generated scene clips")
    p.add_argument("--no-audio", action="store_true")
    p.add_argument("--tts-backend", default="kokoro", choices=["kokoro", "elevenlabs"])
    p.add_argument("--preferred-model", default=None,
                   choices=["wan2_14b", "wan2_1b", "hunyuan", "cogvideox"])
    p.add_argument("--strategy", default="balanced",
                   choices=["fast", "balanced", "quality"],
                   help="Generation strategy: fast (1.3B), balanced (14B+upscale), quality (14B+Hunyuan hero)")
    p.add_argument("--job-id", default=None, help="Unique job ID for output organization")
    return p.parse_args()


def load_script(path_or_text: str) -> str:
    p = Path(path_or_text)
    if p.exists():
        return p.read_text(encoding="utf-8")
    return path_or_text


def main() -> None:
    args = parse_args()

    import uuid
    job_id = args.job_id or uuid.uuid4().hex[:12]
    out_dir = Path("outputs") / job_id
    (out_dir / "scenes").mkdir(parents=True, exist_ok=True)
    (out_dir / "audio").mkdir(parents=True, exist_ok=True)
    (out_dir / "final").mkdir(parents=True, exist_ok=True)

    final_output = Path(args.output) if args.output else out_dir / "final" / f"{job_id}.mp4"

    script_text = load_script(args.script)
    logger.info(f"Job: {job_id} | Style: {args.style} | Quality: {args.quality}")
    logger.info(f"Script: {len(script_text)} chars")

    t_start = time.time()

    # ── 1. Scene splitting ──────────────────────────────────────────
    logger.info("Step 1/5: Splitting script into scenes…")
    from scene_splitter import SceneSplitter
    splitter = SceneSplitter(
        pacing=args.pacing,
        min_duration=args.min_clip,
        max_duration=args.max_clip,
    )
    scenes = splitter.split(script_text)
    splitter.to_json(scenes, out_dir / "scenes.json")
    logger.info(f"  → {len(scenes)} scenes | total: {sum(s.duration for s in scenes):.0f}s")

    # ── 2. Prompt generation ────────────────────────────────────────
    logger.info("Step 2/5: Generating video prompts…")
    from prompt_engine import PromptEngine
    pe = PromptEngine(style=args.style, use_llm=True)
    scenes = asyncio.run(pe.generate_batch(scenes, concurrency=4))
    logger.info("  → Prompts generated.")

    # ── 3. Video generation ─────────────────────────────────────────
    logger.info("Step 3/5: Generating scene clips…")
    from video_engine import VideoRouter
    router = VideoRouter(
        output_dir=str(out_dir / "scenes"),
        quality=args.quality,
        api_fallback=True,
        strategy=args.strategy,
    )

    # Determine hero scene IDs for quality strategy
    hero_scene_ids: set[str] = set()
    if args.strategy == "quality" and len(scenes) >= 2:
        hero_scene_ids = {scenes[0].scene_id, scenes[-1].scene_id}

    clip_paths = []
    for i, scene in enumerate(scenes, 1):
        t = time.time()
        is_hero = scene.scene_id in hero_scene_ids
        logger.info(f"  Scene {i}/{len(scenes)} (id={scene.scene_id}, {scene.duration:.1f}s, hero={is_hero})…")
        path = router.generate_scene(
            scene,
            preferred_model=args.preferred_model,
            is_hero=is_hero,
        )
        clip_paths.append(path)
        logger.info(f"    ✓ {path.name} [{time.time()-t:.1f}s]")
    router.cleanup()

    # ── 4. Audio ────────────────────────────────────────────────────
    audio_path = None
    if not args.no_audio:
        logger.info("Step 4/5: Generating audio (TTS + music)…")
        from audio import TTSEngine, MusicEngine, AudioSync
        total_dur = sum(s.duration for s in scenes)

        tts = TTSEngine()
        narration = tts.synthesize_full(scenes, out_dir / "audio" / "narration.wav")
        logger.info("  → Narration done.")

        music_engine = MusicEngine()
        music = music_engine.generate_for_video(
            args.style, total_dur, out_dir / "audio" / "music.wav"
        )
        music_engine.unload()
        logger.info("  → Music done.")

        syncer = AudioSync()
        audio_path = syncer.mix(
            narration, music, out_dir / "audio" / "mixed.aac", total_dur, scenes
        )
        logger.info("  → Audio mixed.")
    else:
        logger.info("Step 4/5: Audio skipped (--no-audio).")

    # ── 5. Stitch ───────────────────────────────────────────────────
    logger.info("Step 5/5: Stitching final video…")
    from stitcher import Stitcher
    stitcher = Stitcher()
    final = stitcher.stitch(
        clip_paths=clip_paths,
        output_path=final_output,
        audio_path=audio_path,
        transition=args.transition,
        resolution="1080p",
        target_fps=24,
        upscale_to_4k=args.upscale_4k,
    )

    elapsed = time.time() - t_start
    logger.info(f"\n{'='*60}")
    logger.info(f"✓ Done in {elapsed/60:.1f} min")
    logger.info(f"  Output: {final}")
    logger.info(f"  Scenes: {len(scenes)} | Duration: {sum(s.duration for s in scenes):.0f}s")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()
