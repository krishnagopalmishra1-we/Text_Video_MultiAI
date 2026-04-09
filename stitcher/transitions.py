"""
FFmpeg transition filter builder.
Each function returns an ffmpeg filter_complex fragment string.
"""
from __future__ import annotations


def crossfade(duration: float = 0.8, idx: int = 0) -> str:
    """xfade crossfade between two inputs."""
    offset = 0  # calculated externally based on cumulative duration
    return f"xfade=transition=fade:duration={duration}:offset={{offset}}"


def fade_black(duration: float = 0.5) -> str:
    return f"xfade=transition=fadeblack:duration={duration}:offset={{offset}}"


def fade_white(duration: float = 0.5) -> str:
    return f"xfade=transition=fadewhite:duration={duration}:offset={{offset}}"


def wipe_left(duration: float = 0.4) -> str:
    return f"xfade=transition=wipeleft:duration={duration}:offset={{offset}}"


def dissolve(duration: float = 1.0) -> str:
    return f"xfade=transition=dissolve:duration={duration}:offset={{offset}}"


def cut() -> str:
    return ""  # no transition — direct cut (handled by concat)


TRANSITION_MAP = {
    "crossfade": crossfade,
    "fade_black": fade_black,
    "fade_white": fade_white,
    "wipe_left": wipe_left,
    "dissolve": dissolve,
    "cut": cut,
}


_XFADE_NAME_MAP = {
    "crossfade": "fade",
    "fade_black": "fadeblack",
    "fade_white": "fadewhite",
    "wipe_left": "wipeleft",
    "dissolve": "dissolve",
}


def build_xfade_chain(
    clip_durations: list[float],
    transition_type: str = "crossfade",
    transition_duration: float = 0.8,
) -> str:
    """
    Build a complete xfade filter_complex for N clips.
    Returns the full filter_complex string for use in FFmpeg.
    """
    n = len(clip_durations)
    if n == 1:
        return "[0:v]copy[outv]"
    if transition_type == "cut":
        inputs = "".join(f"[{i}:v]" for i in range(n))
        return f"{inputs}concat=n={n}:v=1:a=0[outv]"

    # Map user-facing names to valid FFmpeg xfade transition identifiers
    xfade_name = _XFADE_NAME_MAP.get(transition_type, transition_type)
    td = transition_duration
    lines: list[str] = []
    cumulative = 0.0

    for i in range(n - 1):
        cumulative += clip_durations[i] - td
        in_a = f"[v{i}]" if i > 0 else f"[{i}:v]"
        in_b = f"[{i + 1}:v]"
        out = f"[v{i + 1}]" if i < n - 2 else "[outv]"
        lines.append(
            f"{in_a}{in_b}xfade=transition={xfade_name}:"
            f"duration={td}:offset={cumulative:.3f}{out}"
        )

    return "; ".join(lines)
