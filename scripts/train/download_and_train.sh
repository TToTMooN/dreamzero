#!/bin/bash
# DreamZero: Download data + Start training
#
# PREREQUISITE: Set HF_TOKEN for fast downloads (unauthenticated is rate-limited)
#   export HF_TOKEN=hf_xxxxx
#
# Usage:
#   bash scripts/train/download_and_train.sh
#
# Everything is stored on /mnt/localssd for speed (ephemeral!)
# Remember to back up checkpoints to GCS before stopping the VM.

set -euo pipefail

export HYDRA_FULL_ERROR=1
export CUDA_HOME=/home/sunlingfeng/dreamzero/.venv/cuda_home
export PATH="/home/sunlingfeng/.local/bin:$PATH"
source /home/sunlingfeng/dreamzero/.venv/bin/activate

DATA_DIR="/mnt/localssd/dreamzero/data/droid_lerobot"
CKPT_DIR="/mnt/localssd/dreamzero/checkpoints"
WAN_CKPT_DIR="$CKPT_DIR/Wan2.1-I2V-14B-480P"
TOKENIZER_DIR="$CKPT_DIR/umt5-xxl"
OUTPUT_DIR="$CKPT_DIR/dreamzero_droid_lora"
NUM_GPUS=8

echo "============================================"
echo "  DreamZero DROID Training Pipeline"
echo "============================================"
echo ""

# Step 1: Download dataset
if [ ! -d "$DATA_DIR" ] || [ "$(du -sm "$DATA_DIR" 2>/dev/null | cut -f1)" -lt 100000 ]; then
    echo "[$(date)] Step 1/4: Downloading DROID dataset (~131GB)..."
    if [ -z "${HF_TOKEN:-}" ]; then
        echo "WARNING: HF_TOKEN not set. Download will be SLOW (rate-limited)."
        echo "Set it with: export HF_TOKEN=hf_xxxxx"
    fi
    hf download GEAR-Dreams/DreamZero-DROID-Data --repo-type dataset --local-dir "$DATA_DIR"
else
    echo "[$(date)] Step 1/4: DROID dataset already downloaded ($(du -sh $DATA_DIR | cut -f1))"
fi

# Step 2: Verify model weights (already downloaded)
echo "[$(date)] Step 2/4: Verifying model weights..."
if [ ! -d "$WAN_CKPT_DIR" ] || [ -z "$(ls -A "$WAN_CKPT_DIR" 2>/dev/null)" ]; then
    echo "Downloading Wan2.1-I2V-14B-480P..."
    hf download Wan-AI/Wan2.1-I2V-14B-480P --local-dir "$WAN_CKPT_DIR"
fi
if [ ! -d "$TOKENIZER_DIR" ] || [ -z "$(ls -A "$TOKENIZER_DIR" 2>/dev/null)" ]; then
    echo "Downloading umt5-xxl tokenizer..."
    hf download google/umt5-xxl --local-dir "$TOKENIZER_DIR"
fi
echo "  Wan2.1: $(du -sh $WAN_CKPT_DIR | cut -f1)"
echo "  umt5-xxl: $(du -sh $TOKENIZER_DIR | cut -f1)"

# Step 3: Verify environment
echo "[$(date)] Step 3/4: Verifying environment..."
python -c "
import torch, flash_attn, transformers, deepspeed
assert torch.cuda.is_available(), 'CUDA not available'
assert torch.cuda.device_count() >= $NUM_GPUS, f'Need $NUM_GPUS GPUs, got {torch.cuda.device_count()}'
print(f'  torch={torch.__version__}, GPUs={torch.cuda.device_count()}, flash_attn={flash_attn.__version__}')
"

# Step 4: Launch training
echo "[$(date)] Step 4/4: Launching training on $NUM_GPUS GPUs..."
echo "  Output: $OUTPUT_DIR"
echo ""

cd /home/sunlingfeng/dreamzero

torchrun --nproc_per_node $NUM_GPUS --standalone groot/vla/experiment/experiment.py \
    report_to=none \
    data=dreamzero/droid_relative \
    wandb_project=dreamzero \
    train_architecture=lora \
    num_frames=33 \
    action_horizon=24 \
    num_views=3 \
    model=dreamzero/vla \
    model/dreamzero/action_head=wan_flow_matching_action_tf \
    model/dreamzero/transform=dreamzero_cotrain \
    num_frame_per_block=2 \
    num_action_per_block=24 \
    num_state_per_block=1 \
    seed=42 \
    training_args.learning_rate=1e-4 \
    training_args.deepspeed="groot/vla/configs/deepspeed/zero2.json" \
    save_steps=1000 \
    training_args.warmup_ratio=0.05 \
    output_dir=$OUTPUT_DIR \
    per_device_train_batch_size=1 \
    max_steps=100 \
    weight_decay=1e-5 \
    save_total_limit=10 \
    upload_checkpoints=false \
    bf16=true \
    tf32=true \
    eval_bf16=true \
    dataloader_pin_memory=false \
    dataloader_num_workers=1 \
    image_resolution_width=320 \
    image_resolution_height=176 \
    save_lora_only=true \
    max_chunk_size=4 \
    frame_seqlen=880 \
    save_strategy=steps \
    droid_data_root=$DATA_DIR \
    dit_version=$WAN_CKPT_DIR \
    text_encoder_pretrained_path=$WAN_CKPT_DIR/models_t5_umt5-xxl-enc-bf16.pth \
    image_encoder_pretrained_path=$WAN_CKPT_DIR/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth \
    vae_pretrained_path=$WAN_CKPT_DIR/Wan2.1_VAE.pth \
    tokenizer_path=$TOKENIZER_DIR

echo ""
echo "[$(date)] Training complete!"
echo "Checkpoints saved to: $OUTPUT_DIR"
echo ""
echo "REMINDER: Back up checkpoints to GCS before stopping the VM!"
echo "  cp -r $OUTPUT_DIR /mnt/gcs/dreamzero_checkpoints/"
