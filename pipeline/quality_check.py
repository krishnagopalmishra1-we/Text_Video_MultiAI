"""
Automated clip quality assessment.
Checks for black frames, frozen/static video, and abnormal brightness.
"""
from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def check_clip_quality(clip_path: Path) -> dict:
    """
    Quick quality assessment of a generated video clip.
    Returns dict with quality metrics and pass/fail flags.
    """
    try:
        import cv2
    except ImportError:
        logger.warning("opencv not available — skipping quality check")
        return {"skipped": True}

    cap = cv2.VideoCapture(str(clip_path))
    scores = {
        "path": str(clip_path),
        "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "is_black": False,
        "is_static": False,
        "avg_brightness": 0.0,
        "passed": True,
    }

    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    if not frames:
        scores["is_black"] = True
        scores["passed"] = False
        return scores

    # Average brightness
    brightness = [f.mean() for f in frames]
    scores["avg_brightness"] = round(sum(brightness) / len(brightness), 2)
    scores["is_black"] = scores["avg_brightness"] < 10

    # Frozen/static detection — compare consecutive frames
    if len(frames) > 2:
        diffs = [
            cv2.absdiff(frames[i], frames[i + 1]).mean()
            for i in range(len(frames) - 1)
        ]
        avg_diff = sum(diffs) / len(diffs)
        scores["avg_frame_diff"] = round(avg_diff, 2)
        scores["is_static"] = avg_diff < 1.0

    scores["passed"] = not scores["is_black"] and not scores["is_static"]
    return scores


def check_batch(clips: list[Path]) -> list[dict]:
    """Check all clips, return list of results."""
    results = []
    for clip in clips:
        result = check_clip_quality(clip)
        if not result.get("passed", True):
            logger.warning(f"Quality check FAILED for {clip.name}: {result}")
        results.append(result)
    return results
