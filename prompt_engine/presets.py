"""Style and camera preset helpers."""
from __future__ import annotations


STYLE_PRESETS: dict[str, dict] = {
    "cinematic": {
        "prefix": "cinematic film shot, 35mm, anamorphic lens, shallow depth of field, golden hour lighting, vibrant warm colors, high contrast, sharp details, professional color grading,",
        "suffix": ", film grain, ultra realistic, 8K, highly detailed, vivid",
        "negative": "cartoon, animation, blurry, low quality, watermark, text, CGI, dark, black, dull, dim, desaturated, monochrome, underexposed, flat lighting, washed out, murky, shadow-heavy, low contrast",
    },
    "documentary": {
        "prefix": "documentary style, natural lighting, handheld camera, realistic,",
        "suffix": ", authentic, raw, photojournalistic, high detail",
        "negative": "staged, artificial, CGI, cartoon, fantasy",
    },
    "commercial": {
        "prefix": "commercial advertisement style, professional studio lighting, 4K, clean, polished,",
        "suffix": ", product quality, vibrant colors, sharp focus",
        "negative": "amateur, dark, grainy, low budget, shaky",
    },
    "sci_fi": {
        "prefix": "sci-fi futuristic, neon accent lights, advanced technology, cinematic, volumetric atmosphere,",
        "suffix": ", cyberpunk aesthetic, holographic elements, ultra detailed",
        "negative": "medieval, historical, cartoonish, low tech",
    },
    "nature": {
        "prefix": "nature documentary, BBC Planet Earth style, golden hour light, photorealistic,",
        "suffix": ", stunning wildlife visuals, ultra HD, pristine",
        "negative": "urban, indoor, artificial, CGI",
    },
    "dramatic": {
        "prefix": "dramatic cinematic, high contrast Rembrandt lighting, moody atmosphere, powerful composition,",
        "suffix": ", epic scale, emotional impact, IMAX quality",
        "negative": "flat, boring, low contrast, comedy, casual",
    },
    "horror": {
        "prefix": "horror film, dramatic amber candlelight, visible detailed subject, deep shadows with bright highlights, gothic atmosphere, sharp textures, cinematic composition, 35mm,",
        "suffix": ", film grain, high detail, vivid colors, visible subject, dramatic lighting",
        "negative": "cartoon, animation, blurry, low quality, watermark, text, CGI, pitch black, completely dark, flat black, black screen, invisible, underexposed, flat lighting, washed out, bright comedy lighting, cheerful, blue tint, cold lighting, overcast",
    },
}

CAMERA_PRESETS: dict[str, str] = {
    "static": "static locked-off camera,",
    "pan_left": "slow smooth pan left,",
    "pan_right": "slow smooth pan right,",
    "tilt_up": "slow tilt up revealing,",
    "tilt_down": "slow tilt down,",
    "zoom_in": "slow push-in zoom,",
    "zoom_out": "slow pull-back zoom,",
    "dolly_forward": "smooth dolly forward tracking shot,",
    "dolly_backward": "smooth dolly backward,",
    "orbit": "smooth orbital camera move,",
    "handheld": "handheld camera with slight natural movement,",
    "crane_up": "crane shot rising upward,",
    "crane_down": "crane shot descending,",
    "aerial": "aerial drone shot bird's eye view,",
}

_DEFAULT_STYLE = "cinematic"
_DEFAULT_CAMERA = "dolly_forward"


def build_prompt(
    core: str,
    style: str,
    camera: str,
    presets: dict | None = None,
) -> str:
    """Assemble final prompt from core text + style + camera presets."""
    if presets and "styles" in presets:
        s = presets["styles"].get(style, presets["styles"].get(_DEFAULT_STYLE, {}))
        c = presets.get("camera_motions", {}).get(camera, "")
    else:
        s = STYLE_PRESETS.get(style, STYLE_PRESETS[_DEFAULT_STYLE])
        c = CAMERA_PRESETS.get(camera, CAMERA_PRESETS[_DEFAULT_CAMERA])

    prefix = s.get("prefix", "")
    suffix = s.get("suffix", "")

    parts = [prefix, c, core, suffix]
    return " ".join(p.strip().rstrip(",") for p in parts if p.strip())
