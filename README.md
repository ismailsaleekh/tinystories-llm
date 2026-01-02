# TinyStories LLM

A GPT-style language model trained on the TinyStories dataset. This project implements a transformer-based model from scratch using PyTorch, designed to generate simple children's stories.

## Project Overview

- **Model**: GPT-style transformer (~30M parameters)
- **Dataset**: [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) (~2.1M short stories)
- **Training**: GCP with NVIDIA T4 GPU
- **Tokenizer**: GPT-2 (tiktoken)

## Project Structure

```
llm-model/
├── config/                 # Model and training configurations
│   ├── model_config.py     # GPT model configs (SMALL/MEDIUM/LARGE)
│   └── training_config.py  # Training hyperparameters
├── data/                   # Data loading and processing
│   ├── dataset.py          # TinyStories dataset class
│   ├── download.py         # Dataset download utilities
│   └── tokenizer.py        # Tokenizer wrapper
├── model/                  # Model architecture
│   └── gpt.py              # GPT implementation (attention, blocks, etc.)
├── training/               # Training loop
│   ├── trainer.py          # Main trainer class
│   └── utils.py            # Training utilities
├── scripts/                # Entry points
│   ├── train.py            # Training script
│   └── evaluate.py         # Evaluation script
├── gcp/                    # GCP deployment scripts
│   ├── setup_vm.sh         # VM initialization
│   ├── run_training.sh     # Launch training
│   └── sync_checkpoints.sh # Download checkpoints
├── tests/                  # Unit tests
└── docs/                   # Documentation
    └── phases/             # Development phase plans
```

## Quick Start

### Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run tests
pytest tests/ -v

# Quick training test (CPU/MPS)
python scripts/train.py --max-iters 100 --batch-size 8 --model-size small --no-wandb
```

### GCP Training

```bash
# 1. Create VM with T4 GPU
gcloud compute instances create tinystories-vm \
  --zone=us-central1-a \
  --machine-type=n1-standard-4 \
  --accelerator=type=nvidia-tesla-t4,count=1 \
  --image-family=pytorch-2-7-cu128-ubuntu-2204-nvidia-570 \
  --image-project=deeplearning-platform-release \
  --boot-disk-size=100GB

# 2. Upload code
gcloud compute scp --recurse ./ tinystories-vm:~/llm-model/

# 3. SSH and run setup
gcloud compute ssh tinystories-vm
cd ~/llm-model && bash gcp/setup_vm.sh

# 4. Start training
bash gcp/run_training.sh 5000 64  # 5000 iters, batch size 64

# 5. Download checkpoints (from local machine)
bash gcp/sync_checkpoints.sh tinystories-vm us-central1-a
```

## Model Configurations

| Config | Params | Layers | Heads | Embed Dim | Use Case |
|--------|--------|--------|-------|-----------|----------|
| SMALL  | ~30M   | 6      | 6     | 384       | Quick experiments |
| MEDIUM | ~85M   | 8      | 8     | 512       | Balanced |
| LARGE  | ~300M  | 12     | 12    | 768       | Best quality |

## Training

### Hyperparameters

- **Optimizer**: AdamW (lr=3e-4, weight_decay=0.1)
- **LR Schedule**: Cosine with warmup (100 steps)
- **Batch Size**: 64
- **Block Size**: 256 tokens
- **Mixed Precision**: FP16
- **Gradient Clipping**: 1.0

### Expected Results

| Iterations | Loss  | Perplexity | Time (T4) |
|------------|-------|------------|-----------|
| 1000       | ~3.5  | ~33        | ~30 min   |
| 3000       | ~2.5  | ~12        | ~1.5 hr   |
| 5000       | ~2.0  | ~7         | ~2.5 hr   |

## Generation

```python
from model.gpt import GPT
from config.model_config import SMALL_CONFIG
import torch

# Load model
model = GPT(SMALL_CONFIG)
model.load_state_dict(torch.load('checkpoints/model_5000.pt'))
model.eval()

# Generate
prompt = "Once upon a time"
output = model.generate(prompt, max_tokens=100, temperature=0.8)
print(output)
```

## Development Phases

- [x] **Phase 0-1**: Project setup + data pipeline
- [x] **Phase 2**: Transformer model architecture
- [x] **Phase 3**: Training loop + GCP deployment
- [ ] **Phase 4**: Inference optimization
- [ ] **Phase 5**: Deployment (API/Demo)

## Requirements

- Python 3.10+
- PyTorch 2.0+
- CUDA 12.x (for GPU training)
- tiktoken
- datasets
- tqdm

## License

MIT
