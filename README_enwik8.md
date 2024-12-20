# Training NanoGPT on Enwik8 Dataset

This document provides instructions for training and evaluating NanoGPT on the Enwik8 dataset.

## Training Configuration Overview

The training configuration (`config/train_enwik8.py`) sets up a character-level model with the following key parameters:

### Model Architecture
- 12 layers (`n_layer=12`)
- 12 attention heads (`n_head=12`)
- 768 embedding dimensions (`n_embd=768`)
- 30% dropout (`dropout=0.3`)

### Training Parameters
- Batch size: 16
- Block size (context length): 512
- Learning rate: 5e-4 → 5e-5 (with decay)
- Maximum iterations: 150,000
- Gradient accumulation steps: 4
- Beta2: 0.99

### Logging and Evaluation
- Evaluation interval: every 5,000 iterations
- Logging interval: every 5,000 iterations
- Weights & Biases logging enabled
- Checkpoints saved only on validation improvement

### Progressive Loss (Optional)
- Can be enabled via `use_prog=True`
- Progressive duration ratio: 0.2
- Supports multiple weighting schemes: exp, power, step, sharp_cutoff

All these parameters can be adjusted in `config/train_enwik8.py` or overridden via command line arguments.

## Prerequisites

- CUDA-compatible GPU
- Conda environment with PyTorch and required dependencies
- The Enwik8 dataset (will be prepared in the setup step)

## Environment Setup

1. Activate the conda environment:
```bash
source activate nanogpt
```

2. Make sure your GPU is properly configured:
```bash
export CUDA_VISIBLE_DEVICES=0  # Use first GPU
```

## Data Preparation

Prepare the Enwik8 dataset by running:
```bash
python data/enwik8/prepare.py
```

## Training

### Multi-GPU Training
To train the model using multiple GPUs:
```bash
torchrun --standalone --nproc_per_node=N train.py config/train_enwik8.py
```
Replace `N` with the number of GPUs you want to use.

### Single-GPU Training
For training on a single GPU:
```bash
torchrun --standalone --nproc_per_node=1 train.py config/train_enwik8.py
```

## Model Evaluation

To evaluate a trained model:
```bash
python eval.py \
    --model_dir path_to_model/ \
    --test_data data/enwik8/test.bin \
    --batch_size 32 \
    --block_size 512 \
    --plot
```

Parameters:
- `--model_dir`: Directory containing the trained model checkpoints
- `--test_data`: Path to the test dataset
- `--batch_size`: Batch size for evaluation (default: 32)
- `--block_size`: Context size (default: 512)
- `--plot`: Enable plotting of position-wise loss (optional)
