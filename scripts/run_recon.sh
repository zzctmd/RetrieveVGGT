#!/bin/bash
# ============================================
# RetrieveVGGT 3D Reconstruction Evaluation
# Datasets: 7-Scenes, Neural RGBD
# ============================================
#
# Usage:
#   bash scripts/run_recon.sh
#   GPU_ID=1 DATASET=nrgbd bash scripts/run_recon.sh
#   GPU_ID=0 FRAMES="200 300 500" THRESHOLD_MODE="mean" bash scripts/run_recon.sh
# ============================================

set -e

# Configuration
GPU_ID=${GPU_ID:-0}
DATASET="${DATASET:-7scenes}"              # 7scenes or nrgbd
THRESHOLD_MODE="${THRESHOLD_MODE:-mean+0.3std}"
FRAME_LIST="${FRAMES:-200}"
CKPT_PATH="${CKPT_PATH:-ckpt/checkpoints.pth}"
RESULT_DIR="${RESULT_DIR:-eval_results}"

# Defaults
TOP_K="${TOP_K:-47}"
ANCHOR="${ANCHOR:-1}"
SIZE="${SIZE:-518}"

# Check checkpoint
if [ ! -f "$CKPT_PATH" ]; then
    echo "Error: Checkpoint not found at $CKPT_PATH"
    echo "Download from: https://huggingface.co/lch01/StreamVGGT"
    exit 1
fi

echo "========================================"
echo "RetrieveVGGT 3D Reconstruction"
echo "========================================"
echo "GPU: $GPU_ID"
echo "Dataset: $DATASET"
echo "Threshold: $THRESHOLD_MODE"
echo "Frame List: $FRAME_LIST"
echo "Top-K: $TOP_K, Anchor: $ANCHOR"
echo "========================================"

cd src/

for frames in ${FRAME_LIST}; do
    OUTPUT_DIR="../${RESULT_DIR}/${DATASET}_${frames}"

    echo ""
    echo ">>> ${DATASET} ${frames} frames - $(date)"

    CUDA_VISIBLE_DEVICES=$GPU_ID python eval/mv_recon/launch_retrieve.py \
        --weights "../$CKPT_PATH" \
        --dataset "$DATASET" \
        --size $SIZE \
        --output_dir "$OUTPUT_DIR" \
        --max_frames ${frames} \
        --top_k $TOP_K \
        --anchor $ANCHOR \
        --use_segment_sampling \
        --segment_threshold_mode "$THRESHOLD_MODE"

    echo "<<< ${DATASET} ${frames} frames done - $(date)"
done

echo ""
echo "========================================"
echo "Done! Results in: ${RESULT_DIR}/"
echo "========================================"
