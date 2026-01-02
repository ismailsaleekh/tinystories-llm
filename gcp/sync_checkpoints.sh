#!/bin/bash
# Sync checkpoints from GCP VM to local machine
#
# Usage:
#   ./gcp/sync_checkpoints.sh [VM_NAME] [ZONE]
#
# Examples:
#   ./gcp/sync_checkpoints.sh                           # Default VM and zone
#   ./gcp/sync_checkpoints.sh my-vm                     # Custom VM name
#   ./gcp/sync_checkpoints.sh my-vm us-west1-b          # Custom VM and zone

set -e

# Configuration (with defaults)
VM_NAME=${1:-"tinystories-vm"}
ZONE=${2:-"us-central1-a"}

echo "=========================================="
echo "Syncing from GCP VM"
echo "=========================================="
echo "  VM Name: $VM_NAME"
echo "  Zone:    $ZONE"
echo "=========================================="
echo ""

# Create local directories
echo "Creating local directories..."
mkdir -p checkpoints
mkdir -p logs

# Download checkpoints
echo ""
echo "Downloading checkpoints..."
gcloud compute scp --recurse \
    ${VM_NAME}:~/llm-model/checkpoints/* \
    ./checkpoints/ \
    --zone=${ZONE} \
    || echo "No checkpoints found or download failed"

# Download logs
echo ""
echo "Downloading logs..."
gcloud compute scp --recurse \
    ${VM_NAME}:~/llm-model/logs/* \
    ./logs/ \
    --zone=${ZONE} \
    || echo "No logs found or download failed"

echo ""
echo "=========================================="
echo "Sync complete!"
echo "=========================================="
echo ""
echo "Downloaded checkpoints:"
ls -la checkpoints/ 2>/dev/null || echo "  (none)"
echo ""
echo "Downloaded logs:"
ls -la logs/*.log 2>/dev/null || echo "  (none)"
echo ""
