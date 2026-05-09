#!/bin/bash
# ============================================
# RetrieveVGGT Camera Pose Estimation Evaluation
# Dataset: TUM RGB-D Dynamic
# ============================================
#
# Usage:
#   bash scripts/run_pose.sh
#   GPU_ID=1 DATASETS="tum_dynamic_s1_200 tum_dynamic_s1_500" bash scripts/run_pose.sh
# ============================================

set -e

# Configuration
GPU_ID=${GPU_ID:-1}
THRESHOLD_MODE="${THRESHOLD_MODE:-mean+0.3std}"
CKPT_PATH="${CKPT_PATH:-ckpt/checkpoints.pth}"
RESULT_DIR="${RESULT_DIR:-eval_results/pose}"
DATASET_LIST="${DATASETS:-tum_dynamic_s1_50 tum_dynamic_s1_200}"
SIZE="${SIZE:-512}"

# Check checkpoint
if [ ! -f "$CKPT_PATH" ]; then
    echo "Error: Checkpoint not found at $CKPT_PATH"
    exit 1
fi

echo "========================================"
echo "RetrieveVGGT Camera Pose Estimation"
echo "========================================"
echo "GPU: $GPU_ID"
echo "Threshold: $THRESHOLD_MODE"
echo "Datasets: $DATASET_LIST"
echo "========================================"

cd src/

for dataset in ${DATASET_LIST}; do
    OUTPUT_DIR="../${RESULT_DIR}/${dataset}"

    echo ""
    echo ">>> Pose ${dataset} - $(date)"

    CUDA_VISIBLE_DEVICES=$GPU_ID python eval/pose_evaluation/launch.py \
        --weights "../$CKPT_PATH" \
        --eval_dataset ${dataset} \
        --output_dir "$OUTPUT_DIR" \
        --size $SIZE \
        --use_segment_sampling \
        --segment_threshold_mode "$THRESHOLD_MODE"

    echo "<<< Pose ${dataset} done - $(date)"
done

echo ""
echo "========================================"
echo "Done! Results in: ${RESULT_DIR}/"
echo "========================================"
