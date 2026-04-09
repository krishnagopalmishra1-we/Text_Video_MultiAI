#!/usr/bin/env bash
# ComfyUI startup script
# First run: pulls Wan2.1 model files from HF hub into the shared hf_cache volume,
# then symlinks them into ComfyUI's model directories.
# Subsequent runs: skips downloads, starts immediately.
set -euo pipefail

COMFYUI_DIR="/opt/comfyui"
MODELS_DIR="$COMFYUI_DIR/models"

mkdir -p "$MODELS_DIR/diffusion_models" \
         "$MODELS_DIR/text_encoders" \
         "$MODELS_DIR/vae"

# ── wire_first_available: try multiple filenames and link first that exists ──
wire_first_available() {
    local repo="$1" target_dir="$2"
    shift 2

    local remote_file target cached
    for remote_file in "$@"; do
        target="$target_dir/$(basename "$remote_file")"
        if [ -L "$target" ] || [ -f "$target" ]; then
            echo "[comfyui] already present: $(basename "$remote_file")"
            return 0
        fi

        echo "[comfyui] trying $(basename "$remote_file") from $repo …"
        if cached=$(python3 - <<PYEOF
from huggingface_hub import hf_hub_download
import os
try:
    p = hf_hub_download(repo_id="$repo", filename="$remote_file", repo_type="model")
    print(os.path.realpath(p))
except Exception:
    raise SystemExit(1)
PYEOF
); then
            ln -s "$cached" "$target"
            echo "[comfyui] linked: $target"
            return 0
        fi
    done

    echo "[comfyui] ERROR: no candidate file found in $repo for target dir $target_dir"
    return 1
}

# ── Model files (Kijai/WanVideo_comfy — pre-converted safetensors) ─────────
# Disk budget: ~28 GB transformer + ~10 GB text encoder + ~0.4 GB VAE ≈ 38 GB
# All files land in the shared hf_cache volume; only symlinks added here.

wire_first_available "Kijai/WanVideo_comfy" \
                    "$MODELS_DIR/diffusion_models" \
                    "Wan2_1-T2V-14B_fp8_e4m3fn.safetensors" \
                    "Wan2_1-T2V-14B_fp8_e5m2.safetensors" \
                    "SCAIL/Wan21-14B-SCAIL-preview_comfy_bf16.safetensors"

wire_first_available "Kijai/WanVideo_comfy" \
                    "$MODELS_DIR/text_encoders" \
                    "umt5-xxl-enc-bf16.safetensors"

wire_first_available "Kijai/WanVideo_comfy" \
                    "$MODELS_DIR/vae" \
                    "Wan2_1_VAE_bf16.safetensors" \
                    "Wan2_1_VAE_fp32.safetensors"

echo "[comfyui] All models ready. Starting server on :8188 …"
cd "$COMFYUI_DIR"
exec python3 main.py \
    --listen 0.0.0.0 \
    --port 8188 \
    --cuda-device 0 \
    --preview-method auto
