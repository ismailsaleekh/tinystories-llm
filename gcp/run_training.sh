#!/bin/bash
# Run training on GCP VM
#
# Usage:
#   ./gcp/run_training.sh [MAX_ITERS] [BATCH_SIZE] [RUN_NAME]
#
# Examples:
#   ./gcp/run_training.sh                      # Default: 5000 iters, batch 64
#   ./gcp/run_training.sh 1000                 # Quick run: 1000 iters
#   ./gcp/run_training.sh 5000 64 my-run       # Custom run name
#   ./gcp/run_training.sh 100 32 test --no-wandb  # Test without wandb

set -e

# Configuration (with defaults)
MAX_ITERS=${1:-5000}
BATCH_SIZE=${2:-64}
RUN_NAME=${3:-"tinystories-run-$(date +%Y%m%d-%H%M%S)"}

# Check for additional flags (--no-wandb, etc.)
EXTRA_ARGS=""
for arg in "${@:4}"; do
    EXTRA_ARGS="$EXTRA_ARGS $arg"
done

echo "=========================================="
echo "TinyStories Training"
echo "=========================================="
echo "  Max iters:  $MAX_ITERS"
echo "  Batch size: $BATCH_SIZE"
echo "  Run name:   $RUN_NAME"
if [ -n "$EXTRA_ARGS" ]; then
    echo "  Extra args: $EXTRA_ARGS"
fi
echo "=========================================="
echo ""

# Navigate to project directory
cd ~/llm-model

# Create logs directory if not exists
mkdir -p logs

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
elif [ -d ".venv" ]; then
    echo "Activating virtual environment..."
    source .venv/bin/activate
fi

# Show GPU info
echo ""
echo "GPU Status:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
echo ""

# Run training
echo "Starting training..."
echo ""

python scripts/train.py \
    --max-iters $MAX_ITERS \
    --batch-size $BATCH_SIZE \
    --wandb-run-name $RUN_NAME \
    $EXTRA_ARGS \
    2>&1 | tee logs/training_${RUN_NAME}.log

echo ""
echo "=========================================="
echo "Training complete!"
echo "=========================================="
echo ""
echo "Checkpoints saved to: checkpoints/"
ls -la checkpoints/
echo ""
echo "Logs saved to: logs/training_${RUN_NAME}.log"
echo ""
echo "To download results locally, run:"
echo "  ./gcp/sync_checkpoints.sh"
echo ""
