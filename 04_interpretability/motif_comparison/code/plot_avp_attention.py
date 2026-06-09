#!/usr/bin/env python3
import json
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

task_dir = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
base_dir = os.environ.get('PEPSEF_MOTIF_CATEGORY_DIR', os.path.join(task_dir, 'data', 'by_category_combined_test_run4'))
jsonl = os.path.join(base_dir, 'AVP.fasta.key_subsequences.jsonl')
out_png = os.environ.get('PEPSEF_AVP_ATTENTION_PNG', os.path.join(task_dir, 'figures', 'AVP_attention_map.png'))

# true labels requested for AVP (one sequence per true subsequence)
true_labels = ['EES', 'LLE', 'QEK']

records = []
with open(jsonl, 'r', encoding='utf-8') as f:
    for line in f:
        rec = json.loads(line)
        records.append(rec)

# For each true label, pick the first sequence whose key_subsequences contains that label (case-insensitive).
# If none found for a label, skip it.
selected = []
found_for_label = {}
for label in true_labels:
    found_for_label[label] = None

for rec in records:
    keys = rec.get('key_subsequences', []) or []
    up_keys = [k.upper() for k in keys if k]
    for label in true_labels:
        if found_for_label[label] is None and label in up_keys:
            found_for_label[label] = rec
            selected.append(rec)
    # stop early if we've found all labels
    if all(found_for_label[l] is not None for l in true_labels):
        break

if len(selected) == 0:
    raise RuntimeError('No records found for requested true key_subsequences')

# Prepare figure with one row per selected sequence
n = len(selected)
per_row = 0.9
fig_h = 1 + n * per_row
fig_w = 12
fig, axes = plt.subplots(nrows=n, ncols=1, figsize=(fig_w, fig_h))
if n == 1:
    axes = [axes]

# collect global min/max for color normalization across selected records
all_vals = []
for rec in selected:
    vals = rec.get('attention_importance', []) or []
    all_vals.extend(vals)
if len(all_vals) == 0:
    raise RuntimeError('Selected records do not contain attention_importance values')
vmin = float(min(all_vals))
vmax = float(max(all_vals))

for ax, rec in zip(axes, selected):
    seq = rec['sequence']
    imp = rec.get('attention_importance', []) or []
    L = len(imp)
    # build 2D array 1xL
    arr = np.array(imp, dtype=float)[None, :]
    im = ax.imshow(arr, aspect='auto', cmap='Blues', vmin=vmin, vmax=vmax)
    ax.set_yticks([])
    # set xticks only for actual attention positions
    ax.set_xticks(np.arange(L))
    # label only up to the valid length (attention vector length)
    seq_labels = list(seq.upper()[:L])
    # pad labels if shorter than attention vector to avoid mismatch
    if len(seq_labels) < L:
        seq_labels = seq_labels + [''] * (L - len(seq_labels))
    ax.set_xticklabels(seq_labels, rotation=90, fontsize=8)
    ax.set_xlim(-0.5, L - 0.5)
    # draw dashed rectangles for selected_indices (if present)
    sel = rec.get('selected_indices', []) or []
    for w in sel:
        if not w:
            continue
        start = min(w)
        end = max(w)
        width = end - start + 1
        rect = Rectangle((start - 0.5, -0.5), width, 1, linewidth=1.5, edgecolor='black', facecolor='none', linestyle='--')
        ax.add_patch(rect)
    # title with first key subseq if available
    keys = rec.get('key_subsequences', []) or []
    title = seq if not keys else f"{seq}   keys: {','.join(keys)}"
    ax.set_title(title, fontsize=10)

# Place a single colorbar to the right without overlapping the axes
from matplotlib.transforms import Bbox
fig.subplots_adjust(right=0.88)
# cax: [left, bottom, width, height] in figure coordinates
cax = fig.add_axes([0.90, 0.15, 0.02, 0.7])
cb = fig.colorbar(im, cax=cax, orientation='vertical')
cb.set_label('attention importance')

plt.tight_layout(rect=[0, 0, 0.9, 1.0])
os.makedirs(os.path.dirname(out_png), exist_ok=True)
plt.savefig(out_png, dpi=300)
print('Wrote', out_png)
