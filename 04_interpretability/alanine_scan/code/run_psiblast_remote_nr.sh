#!/usr/bin/env bash
set -euo pipefail

# Run PSI-BLAST remotely against NCBI nr for short peptides and export ASCII PSSM + CSV.
# Prerequisite: psiblast from NCBI BLAST+ must be available in PATH.
# Recommended install on conda: conda install -c bioconda blast

QUERY_FASTA="${1:-peptides.fasta}"
OUTDIR="${2:-pssm_out_remote_nr}"
mkdir -p "$OUTDIR/single_queries"

python3 - <<'PY' "$QUERY_FASTA" "$OUTDIR/single_queries"
import sys
from pathlib import Path
fasta = Path(sys.argv[1])
outdir = Path(sys.argv[2])
outdir.mkdir(parents=True, exist_ok=True)
name = None
seqs = []
for line in fasta.read_text().splitlines():
    line = line.strip()
    if not line:
        continue
    if line.startswith('>'):
        if name is not None:
            (outdir / f"{name}.fa").write_text(f">{name}\n{''.join(seqs)}\n")
        name = line[1:].split()[0].replace('/', '_').replace('|', '_')
        seqs = []
    else:
        seqs.append(line)
if name is not None:
    (outdir / f"{name}.fa").write_text(f">{name}\n{''.join(seqs)}\n")
PY

for FA in "$OUTDIR"/single_queries/*.fa; do
  ID=$(basename "$FA" .fa)
  echo "[INFO] Running PSI-BLAST remote nr for $ID"
  psiblast \
    -query "$FA" \
    -db nr \
    -remote \
    -evalue 20000 \
    -inclusion_ethresh 0.01 \
    -word_size 2 \
    -matrix PAM30 \
    -gapopen 9 \
    -gapextend 1 \
    -comp_based_stats 0 \
    -seg no \
    -max_target_seqs 500 \
    -out "$OUTDIR/${ID}.blast.txt" \
    -out_ascii_pssm "$OUTDIR/${ID}.pssm"

  python3 pssm_to_csv.py "$OUTDIR/${ID}.pssm" "$OUTDIR/${ID}.pssm.csv"

  # NCBI asks users not to overload the shared remote service; keep jobs serial and spaced.
  sleep 10
done

echo "Done. Outputs are in: $OUTDIR"
