#!/usr/bin/env bash
set -euo pipefail

TASK_DIR="/home/dataset-local/chenzixu/PepSEF/03_experiments/feature_ablation"
DOWNSTREAM_CODE="/home/dataset-local/chenzixu/PepSEF/02_downstream_multilabel_gptesm2/code/fusion2_use_esm2.py"
RUN_DIR="${PEPSEF_RUN_DIR:-$TASK_DIR/runs/$(date '+%Y%m%d_%H%M%S')}"
cd "$TASK_DIR"
mkdir -p "$RUN_DIR/model" "$RUN_DIR/results"
export PYTHONPATH="$TASK_DIR:${PYTHONPATH:-}"

OUT_FILE="$RUN_DIR/results/feature_ablation.out"
mkdir -p "$(dirname "$OUT_FILE")"
: > "$OUT_FILE"

COMMON_ARGS=(
  --gpu_ids 0
  --model_dir /home/dataset-local/chenzixu/PepSEF/01_pretraining/esm2_650m_pretraining/outputs/finetuned_stage2_best
  --epochs 20
  --loss_type focal_dice
  --opt_metric balanced
  --f1_mode macro
  --threshold_mode per_class
  --threshold_search_start 4
  --unfreeze_esm_layers 30
  --learning_rate 3e-5
  --head_lr_mult 12
  --fusion_lr_mult 30
  --fusion_alpha_init -1.0
  --pssm_dropout 0.05
  --early_stop_patience 50
  --hhm_root data/hhm
)

for MODE in none pssm hmm both; do
  {
    echo "============================================================"
    echo "[ABALATION] pssm_hmm=${MODE} | started at $(date '+%F %T')"
    echo "============================================================"
  } >> "$OUT_FILE"

  python -u "$DOWNSTREAM_CODE" "${COMMON_ARGS[@]}" \
    --pssm_hmm "$MODE" \
    --save_path "$RUN_DIR/model/model_${MODE}.bin" \
    --csv_save_path "$RUN_DIR/results/training_result_${MODE}.csv" \
    >> "$OUT_FILE" 2>&1

  {
    echo "[ABALATION] pssm_hmm=${MODE} | finished at $(date '+%F %T')"
    echo
  } >> "$OUT_FILE"
done

echo "All feature ablation runs completed at $(date '+%F %T')" >> "$OUT_FILE"
