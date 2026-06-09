# PepSEF Code Organization

This directory is a reorganized copy of the relevant PSM and tpmlc work. Original
files outside this directory were not modified.

## Layout

- `01_pretraining/bert_pretraining/`
  - BERT pretraining code copied from `PSM/src`.
  - Local inputs: `data/merge_80.fasta`, `data/uniprot_80.fasta`.
  - Local model files and copied smaller outputs are under `models/` and `outputs/`.

- `01_pretraining/esm2_650m_pretraining/`
  - ESM2-650M MLM/LoRA pretraining code copied from `PSM/esm2_finetune`.
  - Local inputs: peptide/protein FASTA files used by `ins.sh`.
  - Local base model: `models/esm2-650M`.
  - Local copied best adapter: `outputs/finetuned_stage2_best`.
  - The full historical `/home/dataset-local/chenzixu/test_stage2_reg_1000ep`
    directory is about 555G and was not copied in full; only `best/` was copied.

- `02_downstream_multilabel_gptesm2/`
  - Main downstream multilabel training task from `gptesm2.sh` and
    `fusion2_use_esm2.py`.
  - Outputs and copied results are under `outputs/` and `results/`.

- `03_experiments/`
  - `feature_ablation/`: feature ablation scripts and logs.
  - `fusion_method_comparison/`: fusion method comparison scripts and logs.
  - `single_classifier_comparison/`: single-classifier comparison code and CSVs.
  - `encoder_comparison/`: encoder/PLM comparison code, JSON/CSV/log outputs, and
    comparison figure.

- `04_interpretability/`
  - `motif_comparison/`: motif plotting and attention/key-subsequence scripts,
    copied category data, and figures.
  - `alanine_scan/`: Alanine Scan code, inferred PSSM/HHM inputs, PSI-BLAST helper
    scripts, and copied figures.

- `common/tpmlc_runtime/`
  - Shared runtime copied from `tpmlc`: `utils`, `esm2_finetune`, model configs,
    `ft_data`, `hhm`, and `ascan`.
  - Task directories link to this internal copy for shared imports and data.

## Running

Run scripts from their task directory. For example:

```bash
cd /home/dataset-local/chenzixu/PepSEF/02_downstream_multilabel_gptesm2
conda activate tp
bash code/gptesm2.sh
```

Feature ablation:

```bash
cd /home/dataset-local/chenzixu/PepSEF/03_experiments/feature_ablation
conda activate tp
bash code/run_feature_ablation.sh
```

Fusion method comparison:

```bash
cd /home/dataset-local/chenzixu/PepSEF/03_experiments/fusion_method_comparison
conda activate tp
bash code/run_fusemethod_comparison.sh
```

Alanine Scan:

```bash
cd /home/dataset-local/chenzixu/PepSEF/04_interpretability/alanine_scan
conda activate tp
python code/AScan.py
```

## Verification

Performed a lightweight syntax test:

```bash
python -m py_compile $(find PepSEF -path '*/code/*' -name '*.py' -type f)
```

Also checked that copied code and shell scripts no longer reference the old
`/home/dataset-local/chenzixu/PSM`, `/home/dataset-local/chenzixu/tpmlc`, or
`/data0/chenzixu` paths.

## GitHub Upload

This repository is configured for code upload. Large generated artifacts are
ignored by `.gitignore`, including experiment `runs/`, `outputs/`, model
checkpoints, and copied large data/features.

After the first remote is configured, use:

```bash
cd /home/dataset-local/chenzixu/PepSEF
bash scripts/git_quick_push.sh "describe your change"
```

If model weights or full generated results need to be published, use Git LFS or
an external release/download location instead of committing them to the normal
Git history.
