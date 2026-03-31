#!/bin/bash
# DreamZero DROID Training - Auto-launch script
# Waits for data download to complete, then starts training
# Usage: bash scripts/train/run_droid_training.sh

set -euo pipefail

export HYDRA_FULL_ERROR=1
export CUDA_HOME=/home/sunlingfeng/dreamzero/.venv/cuda_home
export PATH="/home/sunlingfeng/.local/bin:$PATH"

# Activate venv
source /home/sunlingfeng/dreamzero/.venv/bin/activate

# ============ PATHS ============
DROID_DATA_ROOT="${DROID_DATA_ROOT:-/mnt/localssd/dreamzero/data/droid_lerobot}"
OUTPUT_DIR="${OUTPUT_DIR:-/mnt/localssd/dreamzero/checkpoints/dreamzero_droid_lora}"
WAN_CKPT_DIR="${WAN_CKPT_DIR:-/mnt/localssd/dreamzero/checkpoints/Wan2.1-I2V-14B-480P}"
TOKENIZER_DIR="${TOKENIZER_DIR:-/mnt/localssd/dreamzero/checkpoints/umt5-xxl}"
NUM_GPUS="${NUM_GPUS:-8}"
# ================================

echo "[$(date)] Checking prerequisites..."

# Download dataset if not present
if [ ! -d "$DROID_DATA_ROOT" ] || [ -z "$(find "$DROID_DATA_ROOT" -name '*.mp4' -size +1k 2>/dev/null | head -1)" ]; then
    echo "[$(date)] DROID dataset not found or incomplete at $DROID_DATA_ROOT"
    echo "[$(date)] Downloading DROID dataset (~131GB)..."
    echo "[$(date)] NOTE: Set HF_TOKEN env var for faster downloads"
    hf download GEAR-Dreams/DreamZero-DROID-Data --repo-type dataset --local-dir "$DROID_DATA_ROOT"
fi

# Verify checkpoints
for path in "$WAN_CKPT_DIR" "$TOKENIZER_DIR"; do
    if [ ! -d "$path" ] || [ -z "$(ls -A "$path" 2>/dev/null)" ]; then
        echo "ERROR: Checkpoint not found at $path"
        exit 1
    fi
done

echo "[$(date)] All prerequisites ready. Starting training..."
echo "  Dataset: $DROID_DATA_ROOT ($(du -sh $DROID_DATA_ROOT | cut -f1))"
echo "  Output: $OUTPUT_DIR"
echo "  GPUs: $NUM_GPUS"
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
    droid_data_root=$DROID_DATA_ROOT \
    dit_version=$WAN_CKPT_DIR \
    text_encoder_pretrained_path=$WAN_CKPT_DIR/models_t5_umt5-xxl-enc-bf16.pth \
    image_encoder_pretrained_path=$WAN_CKPT_DIR/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth \
    vae_pretrained_path=$WAN_CKPT_DIR/Wan2.1_VAE.pth \
    tokenizer_path=$TOKENIZER_DIR

echo ""
echo "[$(date)] Training complete!"
echo "Checkpoints saved to: $OUTPUT_DIR"
