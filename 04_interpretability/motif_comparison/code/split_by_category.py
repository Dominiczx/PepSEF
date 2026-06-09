#!/usr/bin/env python3
"""
Split sequences into per-category FASTA files using labels.csv mapping.
For each split (train/val/test), reads labels.csv and seqs.fasta in
/home/dataset-local/chenzixu/PepSEF/common/tpmlc_runtime/data/ft_data/<split>/ and writes per-category
FASTA files to /home/dataset-local/chenzixu/PepSEF/common/tpmlc_runtime/data/ft_data/by_category/<Split>/<CATEGORY>.fasta

Usage: python3 split_by_category.py
"""
import argparse
import csv
import os
from pathlib import Path

TASK_DIR = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_BASE = Path('/home/dataset-local/chenzixu/PepSEF/common/tpmlc_runtime/data/ft_data')
DEFAULT_OUTPUT_BASE = TASK_DIR / 'runs' / 'by_category'
SPLITS = [('train','Train'), ('val','Val'), ('test','Test')]


def read_fasta(path):
    records = []
    header = None
    seq_lines = []
    with open(path, 'r') as f:
        for line in f:
            line = line.rstrip('\n')
            if not line:
                continue
            if line.startswith('>'):
                if header is not None:
                    records.append((header, ''.join(seq_lines)))
                header = line
                seq_lines = []
            else:
                seq_lines.append(line.strip())
        if header is not None:
            records.append((header, ''.join(seq_lines)))
    return records


def safe_fname(s):
    # make a safe filename from category
    return ''.join(c if c.isalnum() or c in ('-', '_') else '_' for c in s)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_base', type=Path, default=DEFAULT_INPUT_BASE)
    parser.add_argument('--output_base', type=Path, default=DEFAULT_OUTPUT_BASE)
    args = parser.parse_args()

    input_base = args.input_base
    output_base = args.output_base
    output_base.mkdir(parents=True, exist_ok=True)
    summary = {}

    for split_dirname, split_display in SPLITS:
        split_path = input_base / split_dirname
        labels_path = split_path / 'labels.csv'
        fasta_path = split_path / 'seqs.fasta'

        if not labels_path.exists() or not fasta_path.exists():
            print(f"Skipping {split_display}: missing {labels_path} or {fasta_path}")
            continue

        print(f"Processing {split_display}...")
        # read labels
        with open(labels_path, newline='') as csvfile:
            reader = csv.reader(csvfile)
            header = next(reader)
            labels = [row for row in reader]

        categories = header
        n_cat = len(categories)

        # parse fasta
        records = read_fasta(fasta_path)

        n_labels = len(labels)
        n_seqs = len(records)
        if n_labels != n_seqs:
            print(f"Warning: {split_display} labels rows ({n_labels}) != fasta records ({n_seqs}). Aligning to min.")
        n = min(n_labels, n_seqs)

        # prepare output dirs and open file handles
        # Keep generated category FASTA files inside this motif task.
        out_dir = output_base / split_display
        out_dir.mkdir(parents=True, exist_ok=True)

        out_files = {}
        counts = {cat:0 for cat in categories}
        for idx, cat in enumerate(categories):
            fname = safe_fname(cat) + '.fasta'
            fpath = out_dir / fname
            out_files[idx] = open(fpath, 'w')

        # iterate and write
        for i in range(n):
            hdr, seq = records[i]
            row = labels[i]
            # ensure row length matches categories
            if len(row) != n_cat:
                # try to pad or trim
                if len(row) < n_cat:
                    row = row + ['0'] * (n_cat - len(row))
                else:
                    row = row[:n_cat]
            for j, val in enumerate(row):
                if val.strip() == '1':
                    f = out_files[j]
                    f.write(hdr + '\n')
                    # wrap sequence to 80 chars per line
                    for k in range(0, len(seq), 80):
                        f.write(seq[k:k+80] + '\n')
                    counts[categories[j]] += 1

        # close files
        for f in out_files.values():
            f.close()

        summary[split_display] = {'labels': n_labels, 'seqs': n_seqs, 'written_counts': counts}
        print(f"Done {split_display}: labels={n_labels}, seqs={n_seqs}, written per-category: \n" + '\n'.join([f"  {k}: {v}" for k,v in counts.items()]))

    # final summary
    print("\nAll splits processed. Summary:")
    for s, info in summary.items():
        print(f"{s}: labels={info['labels']}, seqs={info['seqs']}")
        for k,v in info['written_counts'].items():
            if v>0:
                print(f"  {k}: {v}")


if __name__ == '__main__':
    main()
