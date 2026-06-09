#!/usr/bin/env bash
set -euo pipefail

TASK_DIR="/home/dataset-local/chenzixu/PepSEF/03_experiments/fusion_method_comparison"
DOWNSTREAM_CODE="/home/dataset-local/chenzixu/PepSEF/02_downstream_multilabel_gptesm2/code/fusion2_use_esm2.py"
RUN_DIR="${PEPSEF_RUN_DIR:-$TASK_DIR/runs/$(date '+%Y%m%d_%H%M%S')}"
GPU_IDS="${PEPSEF_GPU_IDS:-0}"
cd "$TASK_DIR"
mkdir -p "$RUN_DIR/model" "$RUN_DIR/results"
export PYTHONPATH="$TASK_DIR:${PYTHONPATH:-}"

OUT_FILE="$RUN_DIR/results/fusemethod3.out"
mkdir -p "$(dirname "$OUT_FILE")"
: > "$OUT_FILE"

COMMON_ARGS=(
  --gpu_ids "$GPU_IDS"
  --epochs 20
  --loss_type focal_dice
  --opt_metric balanced
  --f1_mode macro
  --threshold_mode per_class
  --threshold_search_start 4
  --unfreeze_esm_layers 30
  --learning_rate 2.5e-5
  --head_lr_mult 12
  --fusion_lr_mult 30
  --fusion_alpha_init -1.0
  --pssm_dropout 0.05
  --early_stop_patience 50
  --pssm_hmm both
  --hhm_root data/hhm
)

for METHOD in cross_attention aff concat; do
  {
    echo "============================================================"
    echo "[FUSE_COMPARE] fusion_method=${METHOD} | pssm_hmm=both | started at $(date '+%F %T')"
    echo "============================================================"
  } >> "$OUT_FILE"

  python -u "$DOWNSTREAM_CODE" "${COMMON_ARGS[@]}" \
    --fusion_method "$METHOD" \
    --save_path "$RUN_DIR/model/model_${METHOD}.bin" \
    --csv_save_path "$RUN_DIR/results/training_result_${METHOD}.csv" \
    >> "$OUT_FILE" 2>&1

  {
    echo "[FUSE_COMPARE] fusion_method=${METHOD} | finished at $(date '+%F %T')"
    echo
  } >> "$OUT_FILE"
done

echo "All fusion method comparison runs completed at $(date '+%F %T')" >> "$OUT_FILE"
