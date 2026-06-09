#!/usr/bin/env python3
"""
Convert NCBI PSI-BLAST ASCII PSSM (*.pssm from -out_ascii_pssm) to CSV.
The CSV columns are: pos, aa, A, R, N, D, C, Q, E, G, H, I, L, K, M, F, P, S, T, W, Y, V
"""
import argparse
import csv
import re
from pathlib import Path

AA_ORDER = ["A", "R", "N", "D", "C", "Q", "E", "G", "H", "I", "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V"]
ROW_RE = re.compile(r"^\s*(\d+)\s+([A-Z*])\s+(.+)$")


def parse_pssm(path: Path):
    rows = []
    for line in path.read_text(errors="replace").splitlines():
        m = ROW_RE.match(line)
        if not m:
            continue
        pos = int(m.group(1))
        aa = m.group(2)
        nums = re.findall(r"[-+]?\d+", m.group(3))
        # NCBI ASCII PSSM rows normally start with 20 position-specific scores,
        # followed by frequency/probability columns. Keep only the first 20 scores.
        if len(nums) >= 20:
            rows.append([pos, aa] + [int(x) for x in nums[:20]])
    if not rows:
        raise ValueError(f"No PSSM rows parsed from {path}. Check whether PSI-BLAST produced a valid ASCII PSSM.")
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_pssm", type=Path)
    parser.add_argument("output_csv", type=Path)
    args = parser.parse_args()

    rows = parse_pssm(args.input_pssm)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["pos", "aa"] + AA_ORDER)
        writer.writerows(rows)
    print(f"Wrote {args.output_csv} with {len(rows)} positions")


if __name__ == "__main__":
    main()
