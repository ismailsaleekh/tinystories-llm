# Phase 3: Training Loop & GCP Training

## Overview

Build the training infrastructure and run training on Google Cloud Platform.

**Important**: All training runs on GCP - no local GPU training.

**Estimated Time**: 5-6 hours
**Dependencies**: Phase 0, 1, 2 complete, all tests passing

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        LOCAL MACHINE                             │
│  - Code development                                              │
│  - Unit tests (CPU only)                                         │
│  - Push to GCP                                                   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     GOOGLE CLOUD PLATFORM                        │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  │
│  │ Cloud Storage   │  │ Compute Engine  │  │ Weights & Biases│  │
│  │ - Code bundle   │  │ - GPU training  │  │ - Metrics       │  │
│  │ - Checkpoints   │  │ - T4/V100/A100  │  │ - Samples       │  │
│  │ - Logs          │  │                 │  │ - Monitoring    │  │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Files to Create

| File | Purpose |
|------|---------|
| `config/training_config.py` | Training hyperparameters |
| `training/utils.py` | LR scheduler + optimizer config |
| `training/trainer.py` | Main Trainer class |
| `training/__init__.py` | Exports |
| `scripts/train.py` | Training entry point |
| `scripts/evaluate.py` | Evaluation script |
| `gcp/setup_vm.sh` | GCP VM setup script |
| `gcp/run_training.sh` | Training launch script |
| `gcp/sync_checkpoints.sh` | Download checkpoints from GCP |

---

## Step 1: Training Configuration (`config/training_config.py`)

### Purpose
Centralized training hyperparameters as a dataclass.

### Parameters

```python
@dataclass
class TrainingConfig:
    # Optimization
    learning_rate: float = 3e-4      # Peak learning rate
    min_lr: float = 3e-5             # Minimum LR (10% of max)
    weight_decay: float = 0.1        # AdamW weight decay
    beta1: float = 0.9               # Adam beta1
    beta2: float = 0.95              # Adam beta2
    grad_clip: float = 1.0           # Gradient clipping threshold

    # Schedule
    max_iters: int = 5000            # Total training iterations
    warmup_iters: int = 100          # Linear warmup steps
    lr_decay_iters: int = 5000       # Cosine decay length

    # Evaluation
    eval_interval: int = 500         # Evaluate every N iters
    eval_iters: int = 200            # Batches for evaluation

    # Checkpointing
    checkpoint_interval: int = 1000  # Save every N iters
    checkpoint_dir: str = "checkpoints"

    # Logging
    log_interval: int = 10           # Log every N iters
    wandb_project: str = "tinystories-gpt"
    wandb_run_name: str = None

    # Data
    batch_size: int = 64
    block_size: int = 256
    max_train_stories: int = None    # None = use all

    # Device (GCP)
    device: str = "cuda"             # Always CUDA on GCP
    compile_model: bool = True       # torch.compile for speed
    mixed_precision: bool = True     # FP16 training
```

---

## Step 2: Training Utilities (`training/utils.py`)

### Learning Rate Schedule

```
LR
 │    ╱╲
 │   ╱  ╲
 │  ╱    ╲____
 │ ╱          ╲____
 │╱                 ╲____
 └──────────────────────────→ iterations
   │     │                │
   0   warmup          decay_end
```

**Formula**:
1. **Warmup** (iter < warmup_iters): `lr = max_lr * iter / warmup_iters`
2. **Decay** (warmup <= iter <= decay_iters): Cosine decay from max_lr to min_lr
3. **Constant** (iter > decay_iters): `lr = min_lr`

### Optimizer Configuration

AdamW with selective weight decay:

| Parameter Type | Weight Decay |
|---------------|--------------|
| Linear weights | 0.1 |
| Biases | 0.0 |
| LayerNorm | 0.0 |
| Embeddings | 0.0 |

---

## Step 3: Trainer Class (`training/trainer.py`)

### Responsibilities

| Feature | Implementation |
|---------|----------------|
| Mixed Precision | `torch.cuda.amp.GradScaler` + `autocast` |
| Gradient Clipping | `clip_grad_norm_(params, 1.0)` |
| LR Scheduling | Manual update each iteration |
| Checkpointing | Save model + optimizer + state |
| Evaluation | Periodic loss estimation |
| Sample Generation | Generate text during training |
| Wandb Logging | Loss, LR, perplexity, samples |

### Training Loop Structure

```python
for iter_num in range(max_iters):
    # 1. Update learning rate
    lr = get_lr(iter_num, ...)
    for pg in optimizer.param_groups:
        pg['lr'] = lr

    # 2. Get batch
    x, y = get_batch('train')

    # 3. Forward pass (mixed precision)
    with autocast():
        logits, loss = model(x, y)

    # 4. Backward pass
    optimizer.zero_grad()
    scaler.scale(loss).backward()

    # 5. Gradient clipping
    scaler.unscale_(optimizer)
    clip_grad_norm_(model.parameters(), grad_clip)

    # 6. Optimizer step
    scaler.step(optimizer)
    scaler.update()

    # 7. Logging (every log_interval)
    # 8. Evaluation (every eval_interval)
    # 9. Checkpointing (every checkpoint_interval)
```

### Checkpoint Contents

```python
checkpoint = {
    'model': model.state_dict(),
    'optimizer': optimizer.state_dict(),
    'iter_num': iter_num,
    'best_val_loss': best_val_loss,
    'config': model.config,
    'training_config': training_config,
}
```

---

## Step 4: Training Script (`scripts/train.py`)

### CLI Arguments

```bash
python scripts/train.py \
    --seed 42 \
    --max-iters 5000 \
    --batch-size 64 \
    --max-stories 100000 \    # Optional: limit data for faster testing
    --resume checkpoints/checkpoint_1000.pt \  # Optional: resume
    --no-wandb                 # Optional: disable wandb
```

### Script Flow

```
1. Parse arguments
2. Set random seed
3. Create model config (SMALL_CONFIG)
4. Create training config
5. Load dataset + create dataloaders
6. Create model
7. Create trainer
8. Run training
9. Save final checkpoint
```

---

## Step 5: Evaluation Script (`scripts/evaluate.py`)

### Purpose
Evaluate trained model on validation set.

### Usage

```bash
python scripts/evaluate.py checkpoints/best_model.pt
```

### Output

```
=== Evaluation Results ===
Validation Loss: 1.2345
Validation Perplexity: 3.44
```

---

## Step 6: GCP Setup (`gcp/setup_vm.sh`)

### VM Specifications

| Component | Recommended |
|-----------|-------------|
| Machine Type | n1-standard-4 (4 vCPU, 15GB RAM) |
| GPU | NVIDIA T4 (16GB) or V100 (16GB) |
| Disk | 100GB SSD |
| Region | us-central1 (cheapest) |
| Image | Deep Learning VM with PyTorch 2.0 |

### Setup Script Contents

```bash
#!/bin/bash
# GCP VM setup for TinyStories training

# 1. Update system
sudo apt-get update && sudo apt-get upgrade -y

# 2. Install additional dependencies
pip install --upgrade pip
pip install tiktoken datasets wandb tqdm

# 3. Clone/copy project code
# (handled by sync script)

# 4. Verify GPU
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}')"

# 5. Login to wandb
wandb login
```

---

## Step 7: Training Launch (`gcp/run_training.sh`)

### Script Contents

```bash
#!/bin/bash
# Run training on GCP VM

set -e

# Configuration
MAX_ITERS=${1:-5000}
BATCH_SIZE=${2:-64}
RUN_NAME=${3:-"tinystories-run-$(date +%Y%m%d-%H%M%S)"}

echo "Starting training..."
echo "  Max iters: $MAX_ITERS"
echo "  Batch size: $BATCH_SIZE"
echo "  Run name: $RUN_NAME"

# Activate environment (if using venv)
source ~/venv/bin/activate

# Run training
cd ~/llm-model
python scripts/train.py \
    --max-iters $MAX_ITERS \
    --batch-size $BATCH_SIZE \
    --wandb-run-name $RUN_NAME \
    2>&1 | tee logs/training_$RUN_NAME.log

echo "Training complete!"
echo "Checkpoints saved to: checkpoints/"
```

---

## Step 8: Checkpoint Sync (`gcp/sync_checkpoints.sh`)

### Download from GCP

```bash
#!/bin/bash
# Sync checkpoints from GCP VM to local machine

VM_NAME=${1:-"tinystories-vm"}
ZONE=${2:-"us-central1-a"}

echo "Syncing checkpoints from $VM_NAME..."

# Create local directory
mkdir -p checkpoints

# Download checkpoints
gcloud compute scp --recurse \
    $VM_NAME:~/llm-model/checkpoints/* \
    ./checkpoints/ \
    --zone=$ZONE

# Download logs
mkdir -p logs
gcloud compute scp --recurse \
    $VM_NAME:~/llm-model/logs/* \
    ./logs/ \
    --zone=$ZONE

echo "Sync complete!"
ls -la checkpoints/
```

---

## GCP Commands Reference

### Create VM with GPU

```bash
# Create VM
gcloud compute instances create tinystories-vm \
    --zone=us-central1-a \
    --machine-type=n1-standard-4 \
    --accelerator=type=nvidia-tesla-t4,count=1 \
    --image-family=pytorch-latest-gpu \
    --image-project=deeplearning-platform-release \
    --boot-disk-size=100GB \
    --maintenance-policy=TERMINATE

# SSH into VM
gcloud compute ssh tinystories-vm --zone=us-central1-a

# Copy code to VM
gcloud compute scp --recurse ./llm-model tinystories-vm:~/ --zone=us-central1-a

# Stop VM (to save costs)
gcloud compute instances stop tinystories-vm --zone=us-central1-a

# Start VM
gcloud compute instances start tinystories-vm --zone=us-central1-a

# Delete VM
gcloud compute instances delete tinystories-vm --zone=us-central1-a
```

### Monitor Training

```bash
# SSH and watch logs
gcloud compute ssh tinystories-vm --zone=us-central1-a -- tail -f ~/llm-model/logs/training.log

# Check GPU usage
gcloud compute ssh tinystories-vm --zone=us-central1-a -- nvidia-smi
```

---

## Training Configurations

### Quick Test (verify setup)

```bash
python scripts/train.py --max-iters 100 --batch-size 32 --max-stories 10000 --no-wandb
```

### Small Run (~30 min on T4)

```bash
python scripts/train.py --max-iters 1000 --batch-size 64 --max-stories 100000
```

### Full Training (~2-3 hours on T4)

```bash
python scripts/train.py --max-iters 5000 --batch-size 64
```

### Large Model (~6-8 hours on T4)

```bash
# Modify script to use MEDIUM_CONFIG
python scripts/train.py --max-iters 10000 --batch-size 32
```

---

## Expected Metrics

### SMALL_CONFIG (~10M params)

| Iteration | Train Loss | Val Loss | Val Perplexity |
|-----------|------------|----------|----------------|
| 0 | ~10.8 | ~10.8 | ~49000 |
| 500 | ~2.5 | ~2.6 | ~13 |
| 1000 | ~2.0 | ~2.1 | ~8 |
| 2000 | ~1.6 | ~1.7 | ~5.5 |
| 5000 | ~1.3 | ~1.5 | ~4.5 |

### Signs of Good Training

- Loss decreases smoothly
- Train/val loss stay close (no overfitting)
- Perplexity drops to 4-6 range
- Generated samples become coherent

### Signs of Problems

| Symptom | Likely Cause | Fix |
|---------|--------------|-----|
| Loss NaN | LR too high | Reduce learning_rate |
| Loss stuck | LR too low | Increase learning_rate |
| Val loss increases | Overfitting | Add dropout, reduce iters |
| OOM error | Batch too large | Reduce batch_size |

---

## File Checklist

### Training Code
- [ ] `config/training_config.py` - TrainingConfig dataclass
- [ ] `training/utils.py` - get_lr() + configure_optimizer()
- [ ] `training/trainer.py` - Trainer class
- [ ] `training/__init__.py` - Exports
- [ ] `scripts/train.py` - Training CLI
- [ ] `scripts/evaluate.py` - Evaluation CLI

### GCP Scripts
- [ ] `gcp/setup_vm.sh` - VM setup script
- [ ] `gcp/run_training.sh` - Training launch script
- [ ] `gcp/sync_checkpoints.sh` - Checkpoint download script

---

## Implementation Checklist

### TrainingConfig (`config/training_config.py`)
- [ ] All optimization params (lr, weight_decay, betas, grad_clip)
- [ ] Schedule params (max_iters, warmup_iters, lr_decay_iters)
- [ ] Eval params (eval_interval, eval_iters)
- [ ] Checkpoint params (checkpoint_interval, checkpoint_dir)
- [ ] Logging params (log_interval, wandb_project, wandb_run_name)
- [ ] Data params (batch_size, block_size, max_train_stories)
- [ ] Device params (device="cuda", compile_model, mixed_precision)

### Training Utils (`training/utils.py`)
- [ ] get_lr() with warmup + cosine decay
- [ ] configure_optimizer() with weight decay separation
- [ ] Fused AdamW detection

### Trainer (`training/trainer.py`)
- [ ] __init__: setup device, model, optimizer, scaler
- [ ] Mixed precision context manager
- [ ] _get_batch(): iterate through dataloader
- [ ] estimate_loss(): eval on train/val
- [ ] generate_sample(): text generation
- [ ] save_checkpoint(): model + optimizer + state
- [ ] load_checkpoint(): resume training
- [ ] train(): main loop with all features

### Train Script (`scripts/train.py`)
- [ ] Argument parsing
- [ ] Seed setting
- [ ] Config creation
- [ ] Dataloader creation
- [ ] Model creation
- [ ] Trainer creation
- [ ] Training execution
- [ ] Logging setup

### Evaluate Script (`scripts/evaluate.py`)
- [ ] Load checkpoint
- [ ] Create model from config
- [ ] Calculate perplexity
- [ ] Print results

### GCP Scripts
- [ ] setup_vm.sh - dependencies, GPU verify, wandb login
- [ ] run_training.sh - parameterized training launch
- [ ] sync_checkpoints.sh - download from GCP

---

## Verification (on GCP)

```bash
# 1. SSH into VM
gcloud compute ssh tinystories-vm --zone=us-central1-a

# 2. Quick training test
cd ~/llm-model
python scripts/train.py --max-iters 50 --batch-size 16 --no-wandb

# 3. Check checkpoint created
ls -la checkpoints/

# 4. Run evaluation
python scripts/evaluate.py checkpoints/best_model.pt

# 5. Full training with wandb
python scripts/train.py --max-iters 5000 --batch-size 64
```

---

## Cost Estimation (GCP)

| GPU | $/hour | 5000 iters | Full dataset |
|-----|--------|------------|--------------|
| T4 | ~$0.35 | ~$1.00 | ~$1.50 |
| V100 | ~$2.50 | ~$2.50 | ~$4.00 |
| A100 | ~$4.00 | ~$2.00 | ~$3.00 |

**Tips to save costs**:
- Use preemptible/spot VMs (70% cheaper)
- Stop VM when not training
- Start with T4, upgrade only if needed

---

## Success Criteria

Phase 3 is complete when:

1. All training code files created
2. GCP scripts created
3. Training runs successfully on GCP VM
4. Wandb shows loss decreasing
5. Checkpoints saved to GCP VM
6. Can download checkpoints locally
7. Evaluation script works
8. Val perplexity < 10 after 1000 iters

---

## Next Phase

After Phase 3, proceed to **Phase 4: Inference & Deployment** which covers:
- Text generation API
- Flask/FastAPI server
- Docker containerization
- Vertex AI deployment
