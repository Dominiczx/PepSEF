#!/usr/bin/env bash
set -euo pipefail

TASK_DIR="/home/dataset-local/chenzixu/PepSEF/02_downstream_multilabel_gptesm2"
RUN_DIR="${PEPSEF_RUN_DIR:-$TASK_DIR/runs/$(date '+%Y%m%d_%H%M%S')}"
cd "$TASK_DIR"
mkdir -p "$RUN_DIR/model" "$RUN_DIR/results"
export PYTHONPATH="$TASK_DIR:${PYTHONPATH:-}"

# Full downstream training entry. Outputs are isolated per run so copied
# reference results in results/ are not overwritten during verification.
python -u code/fusion2_use_esm2.py \
    --gpu_ids 1 \
    --epochs 100 \
    --loss_type focal_dice \
    --opt_metric balanced \
    --f1_mode macro \
    --threshold_mode per_class \
    --threshold_search_start 4 \
    --unfreeze_esm_layers 30 \
    --learning_rate 2e-5 \
    --head_lr_mult 12 \
    --fusion_lr_mult 30 \
    --fusion_alpha_init -1.0 \
    --pssm_dropout 0.05 \
    --early_stop_patience 50 \
    --pssm_hmm both \
    --hhm_root data/hhm \
    --fusion_method cross_attention \
    --save_path "$RUN_DIR/model/model_esm2.bin" \
    --csv_save_path "$RUN_DIR/results/training_result_esm2.csv" \
    > "$RUN_DIR/results/33_ca.out"
