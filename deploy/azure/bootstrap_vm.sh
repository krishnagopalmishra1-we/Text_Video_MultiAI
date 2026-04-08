#!/usr/bin/env bash
set -euo pipefail

echo "[1/5] Installing base packages..."
sudo apt-get update
sudo apt-get install -y \
  ca-certificates \
  curl \
  git \
  ffmpeg \
  docker.io \
  docker-compose-plugin

echo "[2/5] Ensuring NVIDIA driver is available..."
if ! command -v nvidia-smi >/dev/null 2>&1; then
  sudo apt-get install -y ubuntu-drivers-common
  sudo ubuntu-drivers autoinstall
  echo "NVIDIA drivers installed. Reboot is required before continuing."
  exit 2
fi

echo "[3/5] Installing NVIDIA Container Toolkit..."
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
  sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list >/dev/null

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

echo "[4/5] Configuring Docker GPU runtime..."
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl enable docker
sudo systemctl restart docker

if id -nG "$USER" | grep -qw docker; then
  :
else
  sudo usermod -aG docker "$USER"
  echo "Added $USER to docker group. Re-login required for group changes."
fi

echo "[5/5] Verifying GPU access inside container..."
sudo docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi

echo "Bootstrap completed successfully."
