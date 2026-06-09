#!/usr/bin/env bash
set -euo pipefail

# Run PSI-BLAST against a local nr database and export ASCII PSSM + CSV.
# Usage: bash run_psiblast_local_nr.sh peptides.fasta /path/to/nr 8 pssm_out_local_nr
# Here /path/to/nr is the BLAST database prefix, not a FASTA file.

QUERY_FASTA="${1:-peptides.fasta}"
NR_DB_PREFIX="${2:-/path/to/nr}"
THREADS="${3:-8}"
OUTDIR="${4:-pssm_out_local_nr}"
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
  echo "[INFO] Running PSI-BLAST local nr for $ID"
  psiblast \
    -query "$FA" \
    -db "$NR_DB_PREFIX" \
    -num_threads "$THREADS" \
    -num_iterations 3 \
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
done

echo "Done. Outputs are in: $OUTDIR"
