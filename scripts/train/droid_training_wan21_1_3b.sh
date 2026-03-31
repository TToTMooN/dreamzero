#!/bin/bash
# DreamZero DROID Training with Wan2.1-T2V-1.3B backbone
# ~10x fewer DiT params than 14B, significantly faster training
# First-frame conditioning via CLIP (like 5B), same VAE as 14B
#
# Prerequisites:
#   - Wan2.1-T2V-1.3B weights at WAN13B_CKPT_DIR
#   - CLIP image encoder from Wan2.1-I2V-14B-480P
#   - DROID dataset in LeRobot format
set -euo pipefail

export HYDRA_FULL_ERROR=1
export CUDA_HOME=${CUDA_HOME:-/home/sunlingfeng/dreamzero/.venv/cuda_home}

DROID_DATA_ROOT=${DROID_DATA_ROOT:-"/mnt/localssd/dreamzero/data/droid_lerobot"}
OUTPUT_DIR=${OUTPUT_DIR:-"/mnt/localssd/dreamzero/checkpoints/dreamzero_droid_wan21_1_3b"}
WAN13B_CKPT_DIR=${WAN13B_CKPT_DIR:-"/mnt/localssd/dreamzero/checkpoints/Wan2.1-T2V-1.3B"}
WAN14B_CKPT_DIR=${WAN14B_CKPT_DIR:-"/mnt/localssd/dreamzero/checkpoints/Wan2.1-I2V-14B-480P"}
TOKENIZER_DIR=${TOKENIZER_DIR:-"/mnt/localssd/dreamzero/checkpoints/umt5-xxl"}
NUM_GPUS=${NUM_GPUS:-8}

torchrun --nproc_per_node $NUM_GPUS --standalone groot/vla/experiment/experiment.py \
    report_to=none \
    data=dreamzero/droid_relative \
    wandb_project=dreamzero \
    train_architecture=lora \
    num_frames=33 \
    action_horizon=24 \
    num_views=3 \
    model=dreamzero/vla \
    model/dreamzero/action_head=wan_flow_matching_action_tf_wan21_1_3b \
    model/dreamzero/transform=dreamzero_cotrain \
    num_frame_per_block=2 \
    num_action_per_block=24 \
    num_state_per_block=1 \
    seed=42 \
    training_args.learning_rate=1e-4 \
    "training_args.deepspeed=groot/vla/configs/deepspeed/zero2.json" \
    save_steps=1000 \
    training_args.warmup_ratio=0.05 \
    output_dir=$OUTPUT_DIR \
    per_device_train_batch_size=1 \
    max_steps=10000 \
    weight_decay=1e-5 \
    save_total_limit=10 \
    upload_checkpoints=false \
    bf16=true \
    tf32=true \
    eval_bf16=true \
    dataloader_pin_memory=true \
    dataloader_num_workers=4 \
    image_resolution_width=320 \
    image_resolution_height=176 \
    save_lora_only=true \
    max_chunk_size=4 \
    frame_seqlen=880 \
    save_strategy=steps \
    droid_data_root=$DROID_DATA_ROOT \
    dit_version=$WAN13B_CKPT_DIR \
    text_encoder_pretrained_path=$WAN13B_CKPT_DIR/models_t5_umt5-xxl-enc-bf16.pth \
    image_encoder_pretrained_path=$WAN14B_CKPT_DIR/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth \
    vae_pretrained_path=$WAN13B_CKPT_DIR/Wan2.1_VAE.pth \
    tokenizer_path=$TOKENIZER_DIR
