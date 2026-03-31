#!/bin/bash
# Full pipeline: 1.3B training → eval → 5B training
set -euo pipefail

export CUDA_HOME=/mnt/localssd/dreamzero/venv/cuda_home
export HYDRA_FULL_ERROR=1
source /home/sunlingfeng/dreamzero/.venv/bin/activate
cd /home/sunlingfeng/dreamzero

DATA_ROOT=/mnt/localssd/dreamzero/data/droid_lerobot
CKPT_1_3B=/mnt/localssd/dreamzero/checkpoints/wan21_1_3b_full_200k

echo "[$(date)] === Phase 1: Wan2.1 1.3B Full Fine-tune ==="
bash scripts/train/run_1_3b_full.sh

echo "[$(date)] === Phase 2: Eval 1.3B checkpoint ==="
# Find latest checkpoint
LATEST_CKPT=$(ls -d ${CKPT_1_3B}/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1)
if [ -n "$LATEST_CKPT" ]; then
    echo "Evaluating: $LATEST_CKPT"
    torchrun --nproc_per_node=1 --standalone eval_utils/offline_eval.py \
        --model_path "$LATEST_CKPT" \
        --data_root "$DATA_ROOT" \
        --num_episodes 20 \
        --save_videos \
        --output_dir "${CKPT_1_3B}/eval_results"
    echo "[$(date)] Eval complete. Results: ${CKPT_1_3B}/eval_results/"
else
    echo "WARNING: No checkpoint found for eval"
fi

echo ""
echo "[$(date)] === Phase 3: Wan2.2 5B Full Fine-tune ==="
bash scripts/train/run_wan22_full.sh

echo "[$(date)] === ALL PHASES COMPLETE ==="
