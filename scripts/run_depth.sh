#!/bin/bash
# ============================================
# RetrieveVGGT Video Depth Estimation Evaluation
# Dataset: Bonn RGB-D Dynamic
# ============================================
#
# Usage:
#   bash scripts/run_depth.sh
#   GPU_ID=1 DATASETS="bonn_s1_200 bonn_s1_500" bash scripts/run_depth.sh
# ============================================

set -e

# Configuration
GPU_ID=${GPU_ID:-4}
THRESHOLD_MODE="${THRESHOLD_MODE:-mean+0.3std}"
CKPT_PATH="${CKPT_PATH:-ckpt/checkpoints.pth}"
RESULT_DIR="${RESULT_DIR:-eval_results/depth}"
DATASET_LIST="${DATASETS:-bonn_s1_50 bonn_s1_200}"
SIZE="${SIZE:-518}"

# Check checkpoint
if [ ! -f "$CKPT_PATH" ]; then
    echo "Error: Checkpoint not found at $CKPT_PATH"
    exit 1
fi

echo "========================================"
echo "RetrieveVGGT Video Depth Estimation"
echo "========================================"
echo "GPU: $GPU_ID"
echo "Threshold: $THRESHOLD_MODE"
echo "Datasets: $DATASET_LIST"
echo "========================================"

cd src/

for dataset in ${DATASET_LIST}; do
    OUTPUT_DIR="../${RESULT_DIR}/${dataset}"

    echo ""
    echo ">>> Depth ${dataset} - $(date)"

    CUDA_VISIBLE_DEVICES=$GPU_ID python eval/video_depth/launch_retrieve.py \
        --weights "../$CKPT_PATH" \
        --eval_dataset ${dataset} \
        --output_dir "$OUTPUT_DIR" \
        --size $SIZE \
        --use_segment_sampling \
        --segment_threshold_mode "$THRESHOLD_MODE"

    # Compute depth metrics (metric alignment & scale alignment)
    for align in metric scale; do
        python eval/video_depth/eval_depth.py \
            --output_dir "$OUTPUT_DIR" \
            --eval_dataset ${dataset} \
            --align "${align}" || true
    done

    echo "<<< Depth ${dataset} done - $(date)"
done

echo ""
echo "========================================"
echo "Done! Results in: ${RESULT_DIR}/"
echo "========================================"
