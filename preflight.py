"""
Pre-flight checker — run before any pipeline launch to catch missing deps early.
Usage: python preflight.py
"""
from __future__ import annotations
import subprocess
import sys
from pathlib import Path


def check() -> list[str]:
    errors = []

    # Apply torchvision compat shim (basicsr/realesrgan need functional_tensor removed in newer torchvision)
    try:
        from video_engine.upscaler import _torchvision_shim
        _torchvision_shim()
    except Exception:
        pass

    # Python packages
    for pkg in ['cv2', 'torch', 'diffusers', 'realesrgan', 'basicsr',
                'sageattention', 'nltk', 'yaml', 'google.genai']:
        try:
            __import__(pkg)
        except ImportError:
            errors.append(f"MISSING package: {pkg}")

    # NLTK data
    try:
        import nltk
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        errors.append("MISSING nltk data: punkt_tab — run: python -c \"import nltk; nltk.download('punkt_tab')\"")
    except Exception:
        pass

    # Real-ESRGAN weights
    weights = Path('/root/.cache/realesrgan/RealESRGAN_x4plus.pth')
    if not weights.exists():
        errors.append(f"MISSING weights: {weights}")

    # CUDA
    try:
        import torch
        if not torch.cuda.is_available():
            errors.append("CUDA not available")
    except Exception:
        pass

    # ffmpeg
    r = subprocess.run(['ffmpeg', '-version'], capture_output=True)
    if r.returncode != 0:
        errors.append("ffmpeg not installed")

    # Environment variables
    import os
    for key in ['Gemini_API_KEY', 'HF_TOKEN']:
        if not os.environ.get(key, '').strip():
            errors.append(f"ENV not set: {key}")

    # sageattn signature compat
    try:
        from sageattention import sageattn
        import inspect
        params = list(inspect.signature(sageattn).parameters.keys())
        if params[:3] != ['q', 'k', 'v']:
            errors.append(f"sageattn signature mismatch: {params[:3]} (expected q,k,v)")
    except ImportError:
        pass  # already caught above

    # basicsr torchvision patch
    try:
        import realesrgan  # noqa — tests the full import chain
    except Exception as e:
        errors.append(f"realesrgan import broken: {e}")

    return errors


if __name__ == '__main__':
    errors = check()
    if errors:
        print("PREFLIGHT FAILED:")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)
    else:
        print("PREFLIGHT OK — all checks passed")
