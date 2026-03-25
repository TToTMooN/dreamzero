# DreamZero - Development Guide

## Project Overview
DreamZero is NVIDIA GEAR Lab's World Action Model that jointly predicts actions and videos for zero-shot robot policy. Built on top of Wan2.1-I2V-14B-480P video generation backbone.

Paper: https://arxiv.org/abs/2602.15922

## Repository Structure
```
groot/vla/
├── configs/                 # Hydra YAML configs (data, model, deepspeed)
├── data/dataset/            # Dataset classes (lerobot.py, lerobot_sharded.py)
├── model/dreamzero/         # DreamZero model implementation
├── experiment/experiment.py # Main training entry point (VLATrainer)
└── utils/
scripts/
├── train/                   # Training launch scripts (droid, agibot, yam)
│   ├── download_and_train.sh  # One-command: download + train (NEW)
│   └── run_droid_training.sh  # Train with auto-download (NEW)
├── data/                    # Dataset conversion scripts
└── inference/               # TRT engine building
docs/                        # Guides for new embodiments, DROID conversion, Wan2.2
```

## VM Setup (GCP 8xH100)

### Environment
- **Python**: 3.11 via uv venv at `.venv/`
- **Activate**: `source .venv/bin/activate`
- **CUDA_HOME**: `.venv/cuda_home` (symlinks to pip-installed CUDA tools)
- **uv**: `/home/sunlingfeng/.local/bin/uv`

### Storage Layout
- `/mnt/localssd` — 5.9TB NVMe RAID-0 (FAST but EPHEMERAL — lost on VM stop!)
- `/mnt/gcs` — GCS bucket `physical-ai-intern-workspace` (persistent)
- `data/` → symlink to `/mnt/localssd/dreamzero/data/`
- `checkpoints/` → symlink to `/mnt/localssd/dreamzero/checkpoints/`

### What's Already Installed
- uv + Python 3.11 venv with all dependencies
- torch 2.8.0+cu129, flash-attn 2.8.3, deepspeed 0.18.8, transformers 4.51.3
- Wan2.1-I2V-14B-480P weights (77GB) at `checkpoints/Wan2.1-I2V-14B-480P/`
- umt5-xxl tokenizer (49GB) at `checkpoints/umt5-xxl/`
- git-lfs at `~/.local/bin/git-lfs`

### What's MISSING (Needs HF Token)
- DROID dataset (~131GB) — download blocked by HF rate limit without auth token

## Quick Start: Reproduce DROID Training

```bash
# 1. Set HF token (REQUIRED for dataset download)
export HF_TOKEN=hf_xxxxx

# 2. One-command download + train
cd ~/dreamzero
bash scripts/train/download_and_train.sh

# Or step by step:
source .venv/bin/activate
export CUDA_HOME=.venv/cuda_home
hf download GEAR-Dreams/DreamZero-DROID-Data --repo-type dataset --local-dir data/droid_lerobot
bash scripts/train/run_droid_training.sh
```

## Training Variants
| Script | Backbone | Mode | Notes |
|--------|----------|------|-------|
| `droid_training_lora.sh` | Wan2.1 14B | LoRA | Default, LR=1e-4 |
| `droid_training_full_finetune.sh` | Wan2.1 14B | Full | LR=1e-5, ZeRO-2 offload |
| `droid_training_wan22.sh` | Wan2.2 5B | LoRA | Lower VRAM, 320x160 |
| `droid_training_full_finetune_wan22.sh` | Wan2.2 5B | Full | Lower VRAM |

### Key Training Parameters
- `max_steps=100` in scripts is a sanity check — increase for full training
- `num_frames=33`, `action_horizon=24`, `num_views=3`
- `image_resolution`: 320x176 (Wan2.1) or 320x160 (Wan2.2)
- DeepSpeed ZeRO-2 for distributed training

## Config System
Uses Hydra. Main config: `groot/vla/configs/conf.yaml`
- Data configs: `groot/vla/configs/data/dreamzero/`
- Model configs: `groot/vla/configs/model/dreamzero/`
- DeepSpeed: `groot/vla/configs/deepspeed/`

## Development Notes
- Branch `dev/logging-and-fork` is for development, logging improvements, and future fork
- `evdev` removed from pyproject.toml (not needed for training, requires kernel headers)
- Entry point: `groot/vla/experiment/experiment.py` → `VLATrainer`
- Dataset: DROID uses `ShardedLeRobotSubLangSingleActionChunkDatasetDROID` class
- 3 cameras: exterior_image_1_left, exterior_image_2_left, wrist_image_left
- Relative actions enabled for joint_position

## IMPORTANT: Back Up Before Stopping VM!
```bash
# Local SSD data is LOST when VM stops!
cp -r /mnt/localssd/dreamzero/checkpoints/dreamzero_droid_lora /mnt/gcs/dreamzero_checkpoints/
```
