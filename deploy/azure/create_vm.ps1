param(
    [string]$ResourceGroup = "video-gen-rg",
    [string]$Location = "eastus",
    [string]$VmName = "video-gen-a100",
    [string]$VmSize = "Standard_NC24ads_A100_v4",
    [string]$Image = "Ubuntu2204",
    [string]$AdminUsername = "azureuser",
    [string]$SshPublicKeyPath = "$HOME/.ssh/id_rsa.pub"
)

$ErrorActionPreference = "Stop"

if (-not (Get-Command az -ErrorAction SilentlyContinue)) {
    throw "Azure CLI is required. Install from https://aka.ms/installazurecliwindows"
}

if (-not (Test-Path $SshPublicKeyPath)) {
    throw "SSH public key not found at: $SshPublicKeyPath"
}

Write-Host "Checking Azure login..."
az account show | Out-Null

Write-Host "Creating resource group: $ResourceGroup ($Location)"
az group create --name $ResourceGroup --location $Location | Out-Null

Write-Host "Creating GPU VM: $VmName ($VmSize)"
az vm create `
    --resource-group $ResourceGroup `
    --name $VmName `
    --image $Image `
    --size $VmSize `
    --admin-username $AdminUsername `
    --ssh-key-values $SshPublicKeyPath `
    --public-ip-sku Standard `
    --authentication-type ssh `
    --output none

Write-Host "Installing NVIDIA GPU driver extension..."
az vm extension set `
    --resource-group $ResourceGroup `
    --vm-name $VmName `
    --publisher Microsoft.HpcCompute `
    --name NvidiaGpuDriverLinux `
    --output none

Write-Host "Opening API and Flower ports..."
az vm open-port --resource-group $ResourceGroup --name $VmName --port 8000 --priority 1010 | Out-Null
az vm open-port --resource-group $ResourceGroup --name $VmName --port 5555 --priority 1020 | Out-Null

$ip = az vm show -d --resource-group $ResourceGroup --name $VmName --query publicIps -o tsv

Write-Host ""
Write-Host "VM created successfully."
Write-Host "Public IP: $ip"
Write-Host ""
Write-Host "Next steps:"
Write-Host "1) SSH into VM:"
Write-Host "   ssh $AdminUsername@$ip"
Write-Host "2) Clone your repo and run bootstrap script:"
Write-Host "   bash deploy/azure/bootstrap_vm.sh"
Write-Host "3) Add .env and start stack:"
Write-Host "   docker compose up -d --build"
