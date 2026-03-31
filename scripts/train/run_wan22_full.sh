#!/bin/bash
# DreamZero Wan2.2-TI2V-5B Full Fine-tune on DROID
# 200K steps, bs=1/GPU × 8 GPUs = 8 global batch
# (bs>1 not supported due to VRAM — 5B is larger than 1.3B)
# ETA: ~78 hours on 8xH100 (1.4s/step)
set -euo pipefail

export CUDA_HOME=/mnt/localssd/dreamzero/venv/cuda_home
export HYDRA_FULL_ERROR=1
source /home/sunlingfeng/dreamzero/.venv/bin/activate
cd /home/sunlingfeng/dreamzero

echo "[$(date)] Starting Wan2.2-TI2V-5B Full Fine-tune..."
echo "  Auto-restarts on crash (NVLink errors etc)"
echo ""

MAX_RETRIES=50
for ATTEMPT in $(seq 1 $MAX_RETRIES); do
    echo "[$(date)] Training attempt $ATTEMPT/$MAX_RETRIES"
    torchrun --nproc_per_node 8 --standalone groot/vla/experiment/experiment.py \
    report_to=wandb \
    data=dreamzero/droid_relative_wan22 \
    wandb_project=dreamzero-lite \
    train_architecture=full \
    num_frames=33 action_horizon=24 num_views=3 \
    model=dreamzero/vla \
    model/dreamzero/action_head=wan_flow_matching_action_tf_wan22 \
    model/dreamzero/transform=dreamzero_cotrain \
    num_frame_per_block=2 num_action_per_block=24 num_state_per_block=1 \
    seed=42 training_args.learning_rate=1e-5 \
    "training_args.deepspeed=groot/vla/configs/deepspeed/zero2.json" \
    save_steps=1000 training_args.warmup_ratio=0.05 \
    output_dir=/mnt/localssd/dreamzero/checkpoints/wan22_full_200k \
    per_device_train_batch_size=1 \
    max_steps=200000 \
    weight_decay=1e-5 save_total_limit=10 upload_checkpoints=false \
    bf16=true tf32=true eval_bf16=true \
    dataloader_pin_memory=true dataloader_num_workers=4 \
    image_resolution_width=320 image_resolution_height=160 \
    save_lora_only=false max_chunk_size=4 \
    log_video_every_n_steps=1000 \
    save_strategy=steps \
    droid_data_root=/mnt/localssd/dreamzero/data/droid_lerobot \
    dit_version=/mnt/localssd/dreamzero/checkpoints/Wan2.2-TI2V-5B \
    text_encoder_pretrained_path=/mnt/localssd/dreamzero/checkpoints/Wan2.2-TI2V-5B/models_t5_umt5-xxl-enc-bf16.pth \
    image_encoder_pretrained_path=/mnt/localssd/dreamzero/checkpoints/Wan2.1-I2V-14B-480P/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth \
    vae_pretrained_path=/mnt/localssd/dreamzero/checkpoints/Wan2.2-TI2V-5B/Wan2.2_VAE.pth \
    tokenizer_path=/mnt/localssd/dreamzero/checkpoints/umt5-xxl

    EXIT_CODE=$?
    if [ $EXIT_CODE -eq 0 ]; then
        echo "[$(date)] Training completed successfully!"
        break
    fi
    echo "[$(date)] Training crashed (exit $EXIT_CODE). Resuming in 30s..."
    sleep 30
done
