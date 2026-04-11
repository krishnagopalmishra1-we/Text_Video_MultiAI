FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    CUDA_HOME=/usr/local/cuda \
    PATH="/usr/local/cuda/bin:$PATH" \
    LD_LIBRARY_PATH="/usr/local/cuda/lib64:$LD_LIBRARY_PATH"

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-dev python3-pip \
    build-essential \
    ffmpeg \
    git wget curl \
    libsndfile1 libsndfile1-dev \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

WORKDIR /app

# PyTorch with CUDA 12.4
RUN python -m pip install --no-cache-dir \
    torch==2.4.1 torchvision==0.19.1 \
    --index-url https://download.pytorch.org/whl/cu124

# Build helpers needed by flash-attn setup
RUN python -m pip install --no-cache-dir packaging wheel setuptools ninja

# Flash Attention (compiled for A100 sm_80)
RUN python -m pip install --no-cache-dir flash-attn --no-build-isolation

# App requirements
COPY requirements.txt .
RUN python -m pip install --no-cache-dir -r requirements.txt

# Patch diffusers attention_dispatch.py: wrap flash_attn_3 custom_op registration in
# try/except to avoid infer_schema failure with flash-attn>=2.8 string annotations.
COPY scripts/patch_diffusers.py /tmp/patch_diffusers.py
RUN python /tmp/patch_diffusers.py && rm /tmp/patch_diffusers.py

# App code
COPY . .

# Create output dirs
RUN mkdir -p outputs/scenes outputs/final outputs/audio outputs/upscaled

# Expose FastAPI port
EXPOSE 8000

CMD ["uvicorn", "server.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
