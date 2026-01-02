#!/bin/bash
# GCP VM setup for TinyStories training
#
# Run this script once after creating a new GCP VM with GPU.
# Assumes using Deep Learning VM with PyTorch 2.0 image.
#
# Usage:
#   ./gcp/setup_vm.sh
#
# VM Specifications (recommended):
#   Machine Type: n1-standard-4 (4 vCPU, 15GB RAM)
#   GPU: NVIDIA T4 (16GB) or V100 (16GB)
#   Disk: 100GB SSD
#   Region: us-central1 (cheapest)
#   Image: Deep Learning VM with PyTorch 2.0

set -e

echo "=========================================="
echo "TinyStories GCP VM Setup"
echo "=========================================="

# 1. Update system
echo ""
echo "[1/5] Updating system packages..."
sudo apt-get update && sudo apt-get upgrade -y

# 2. Install additional Python dependencies
echo ""
echo "[2/5] Installing Python dependencies..."
pip install --upgrade pip
pip install tiktoken datasets wandb tqdm

# 3. Create necessary directories
echo ""
echo "[3/5] Creating project directories..."
mkdir -p ~/llm-model/logs
mkdir -p ~/llm-model/checkpoints
mkdir -p ~/llm-model/data/cache

# 4. Verify GPU
echo ""
echo "[4/5] Verifying GPU setup..."
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
else:
    echo 'WARNING: No GPU detected!'
    exit 1
"

# 5. Login to wandb
echo ""
echo "[5/5] Weights & Biases setup..."
echo "Please login to wandb for experiment tracking."
echo "(You can skip this with Ctrl+C and use --no-wandb during training)"
echo ""
wandb login

echo ""
echo "=========================================="
echo "Setup complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. Upload code: gcloud compute scp --recurse ./llm-model VM_NAME:~/ --zone=ZONE"
echo "  2. SSH into VM: gcloud compute ssh VM_NAME --zone=ZONE"
echo "  3. Run training: cd ~/llm-model && ./gcp/run_training.sh"
echo ""
