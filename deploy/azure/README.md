# Azure Deployment (GPU VM)

This project is best deployed to an Azure GPU VM because it needs CUDA, NVIDIA runtime, and long-running local model inference.

## Recommended target

- VM SKU: `Standard_NC24ads_A100_v4` (1x A100)
- OS image: Ubuntu 22.04
- Runtime: Docker Compose (API + Redis + Celery workers)

If the SKU is unavailable in your region, run:

```bash
az vm list-skus --location <region> --resource-type virtualMachines --output table | grep -i A100
```

## Prerequisites

- Azure subscription with GPU quota
- Azure CLI installed and logged in (`az login`)
- SSH key at `~/.ssh/id_rsa.pub`

## 1) Create the VM

From project root (PowerShell):

```powershell
./deploy/azure/create_vm.ps1 \
  -ResourceGroup video-gen-rg \
  -Location eastus \
  -VmName video-gen-a100 \
  -VmSize Standard_NC24ads_A100_v4
```

This script creates:

- Resource Group
- GPU VM
- NVIDIA GPU driver extension
- NSG inbound rules for ports 8000 and 5555

## 2) SSH and bootstrap the host

```bash
ssh azureuser@<PUBLIC_IP>
```

On the VM:

```bash
git clone <your-repo-url> project
cd project
bash deploy/azure/bootstrap_vm.sh
```

If bootstrap says a reboot is required:

```bash
sudo reboot
# Then SSH in again and re-run bootstrap
```

## 3) Configure secrets and start services

```bash
cp .env.example .env
nano .env
docker compose up -d --build
```

## 4) Verify deployment

```bash
curl http://localhost:8000/health
curl http://<PUBLIC_IP>:8000/health
```

Expected response:

```json
{"status":"ok"}
```

## Endpoints

- API: `http://<PUBLIC_IP>:8000`
- Flower: `http://<PUBLIC_IP>:5555`

## Operational notes

- Outputs persist under `./outputs` on the VM disk.
- Hugging Face cache is in Docker volume `hf_cache`.
- For production-grade resilience, use Azure Managed Disks for data volume and Azure Container Registry for prebuilt images.
- Lock down NSG source IPs for ports 8000/5555 before exposing publicly.
