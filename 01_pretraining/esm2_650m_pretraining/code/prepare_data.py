#!/usr/bin/env python3
"""
prepare_data.py

Utilities to read FASTA files, slice proteins with sliding window,
upsample peptide sequences, and write combined FASTA for stage-2 training.

Usage examples:
  python prepare_data.py --proteins protein_merge_80.fasta --peptides peptides_merge_80.fasta \
      --out stage2_input.fasta --window 50 --stride 25
"""
import argparse
import os
import random
import textwrap


def read_fasta(path):
    sequences = []
    header = None
    seq_lines = []
    with open(path, "r") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    sequences.append((header, ''.join(seq_lines)))
                header = line[1:].strip()
                seq_lines = []
            else:
                seq_lines.append(line.strip())
        if header is not None:
            sequences.append((header, ''.join(seq_lines)))
    return sequences


def write_fasta(seq_tuples, out_path, line_width=80):
    with open(out_path, "w") as f:
        for i, (header, seq) in enumerate(seq_tuples):
            f.write(f">{header}\n")
            for chunk in textwrap.wrap(seq, line_width):
                f.write(chunk + "\n")


def sliding_window(seq, window=50, stride=25, min_len=None):
    """Yield slices of `seq` with given window and stride.
    If min_len provided, only yield slices with >= min_len.
    """
    if min_len is None:
        min_len = window
    slices = []
    L = len(seq)
    if L <= window:
        if L >= min_len:
            slices.append(seq)
        return slices

    i = 0
    while i < L:
        end = i + window
        fragment = seq[i:end] if end <= L else seq[i:L]
        if len(fragment) >= min_len:
            slices.append(fragment)
        if end >= L:
            break
        i += stride
    return slices


def upsample_list(src_list, target_count, seed=42):
    """Upsample `src_list` (list) randomly with replacement to reach target_count."""
    random.seed(seed)
    if len(src_list) == 0:
        return []
    if len(src_list) >= target_count:
        return random.sample(src_list, target_count)
    out = []
    while len(out) < target_count:
        remaining = target_count - len(out)
        take = min(remaining, len(src_list))
        out.extend(random.sample(src_list, take))
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--proteins", required=True, help="Protein FASTA (protein_merge_80.fasta)")
    parser.add_argument("--peptides", required=True, help="Peptide FASTA (peptides_merge_80.fasta)")
    parser.add_argument("--out", required=True, help="Output combined FASTA for stage 2")
    parser.add_argument("--window", type=int, default=50, help="Sliding window length")
    parser.add_argument("--stride", type=int, default=25, help="Sliding window stride")
    parser.add_argument("--min_len", type=int, default=20, help="Minimum slice length to keep")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    proteins = read_fasta(args.proteins)
    peptides = read_fasta(args.peptides)

    # generate protein slices
    protein_slices = []
    for header, seq in proteins:
        slices = sliding_window(seq, window=args.window, stride=args.stride, min_len=args.min_len)
        for j, s in enumerate(slices):
            protein_slices.append((f"{header}_slice_{j}", s))

    # convert peptides to tuples (keep headers unique)
    peptide_list = [(p_header if p_header else f"pep_{i}", seq) for i, (p_header, seq) in enumerate(peptides)]

    # upsample peptides to match number of protein slices
    target = len(protein_slices)
    if target == 0:
        raise RuntimeError("No protein slices generated; check protein FASTA and sliding window params")

    peptide_seqs_only = [s for (_, s) in peptide_list]
    upsampled = upsample_list(peptide_seqs_only, target, seed=args.seed)
    upsampled_tuples = [(f"upsampled_pep_{i}", s) for i, s in enumerate(upsampled)]

    combined = peptide_list + protein_slices + upsampled_tuples
    # shuffle to mix peptides and protein slices
    random.seed(args.seed)
    random.shuffle(combined)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    write_fasta(combined, args.out)
    print(f"Wrote {len(combined)} sequences to {args.out} (peptides: {len(peptide_list)}, protein_slices: {len(protein_slices)}, upsampled_peptides: {len(upsampled_tuples)})")


if __name__ == '__main__':
    main()
