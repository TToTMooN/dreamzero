#!/bin/bash
# Run 1.3B full fine-tune, then 5B full fine-tune sequentially
set -euo pipefail

echo "[$(date)] === Starting Wan2.1 1.3B Full Fine-tune ==="
bash /home/sunlingfeng/dreamzero/scripts/train/run_1_3b_full.sh
echo "[$(date)] === Wan2.1 1.3B Complete ==="

echo ""
echo "[$(date)] === Starting Wan2.2 5B Full Fine-tune ==="
bash /home/sunlingfeng/dreamzero/scripts/train/run_wan22_full.sh
echo "[$(date)] === Wan2.2 5B Complete ==="

echo ""
echo "[$(date)] === ALL TRAINING COMPLETE ==="
