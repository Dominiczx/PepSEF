#!/usr/bin/env python3
import json
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

task_dir = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
base_dir = os.environ.get('PEPSEF_MOTIF_CATEGORY_DIR', os.path.join(task_dir, 'data', 'by_category_combined_test_run4'))
output_dir = os.environ.get('PEPSEF_MOTIF_FIG_DIR', os.path.join(task_dir, 'figures', 'motif_top5_gap_norm'))

# true labels requested for each category
categories = {
    'ABP': ['AGK', 'FLP', 'LKK'],
    'ACP': ['KKL', 'LAK', 'LKK'],
    'AFP': ['CNY', 'RRR', 'LKK'],
    'AVP': ['EES', 'LLE', 'QEK'],
}


def process_category(jsonl, out_png, out_json, true_labels):
    if not os.path.exists(jsonl):
        print('Missing:', jsonl)
        return

    records = []
    with open(jsonl, 'r', encoding='utf-8') as f:
        for line in f:
            rec = json.loads(line)
            records.append(rec)

    # build mapping from motif -> candidate records where motif truly appears in sequence
    label_map = {l: [] for l in true_labels}
    for rec in records:
        seq = (rec.get('sequence') or '').upper()
        if not seq:
            continue
        for l in true_labels:
            if l in seq:
                label_map[l].append(rec)

    # save full mapping (all records that contain each motif in extracted key_subsequences)
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, 'w', encoding='utf-8') as fo:
        json.dump(label_map, fo, indent=2)
    print('Wrote', out_json)

    # Collect all strict-matched candidates in this category, then select top-5 by gap=(top1-top2).
    candidates = []
    for rec in records:
        seq = (rec.get('sequence') or '').upper()
        att = rec.get('attention_importance') or []
        if not seq or not att:
            continue

        arr = np.array(att, dtype=float)
        if arr.size == 0:
            continue
        global_max_idx = int(np.argmax(arr))
        top1 = float(arr[global_max_idx])
        if arr.size >= 2:
            top2 = float(np.partition(arr, -2)[-2])
        else:
            top2 = 0.0
        gap = float(top1 - top2)

        matched_windows = []
        matched_motifs = []
        for l in true_labels:
            motif_len = len(l)
            for start in range(0, max(0, len(seq) - motif_len + 1)):
                if seq[start:start + motif_len] != l:
                    continue
                window = list(range(start, start + motif_len))
                if global_max_idx in window:
                    matched_windows.append(window)
                    matched_motifs.append(l)

        # strict condition: keep only sequences where at least one motif window contains global max
        if not matched_windows:
            continue

        candidates.append({
            'record': rec,
            'global_max_index': global_max_idx,
            'top1': top1,
            'top2': top2,
            'gap': gap,
            'matched_windows': matched_windows,
            'matched_motifs': matched_motifs,
        })

    candidates.sort(key=lambda x: x['gap'], reverse=True)
    selected = candidates[:5]

    if len(selected) == 0:
        print('No strict-matched records found (global max inside true motif window) in', jsonl)
        # still write an empty strict-matching file for traceability
        out_matched = out_json.replace('_label_sequences.json', '_strict_matched.json')
        with open(out_matched, 'w', encoding='utf-8') as fo:
            json.dump({}, fo, indent=2)
        print('Wrote', out_matched)
        return

    # Prepare figure
    n = len(selected)
    per_row = 0.9
    fig_h = 1 + n * per_row
    fig_w = 12
    fig, axes = plt.subplots(nrows=n, ncols=1, figsize=(fig_w, fig_h))
    if n == 1:
        axes = [axes]

    # collect global min/max for color normalization
    all_vals = []
    for item in selected:
        rec = item['record']
        vals = rec.get('attention_importance', []) or []
        all_vals.extend(vals)
    if len(all_vals) == 0:
        print('Selected records do not contain attention_importance values for', jsonl)
        return
    vmin = float(min(all_vals))
    vmax = float(max(all_vals))

    im = None
    # selected records are already strict-matched by construction
    labels_with_global_max = []
    for ax, item in zip(axes, selected):
        rec = item['record']
        motifs = item['matched_motifs']
        strict_windows = item['matched_windows']
        global_max_idx = item['global_max_index']
        gap = item['gap']
        seq = rec.get('sequence', '')
        imp = rec.get('attention_importance', []) or []
        L = len(imp)
        if L == 0:
            ax.set_yticks([])
            ax.set_title(f'category: {os.path.basename(jsonl).split(".")[0]}   seq: {seq}   (no attention values)', fontsize=10)
            continue
        # Normalize per-sequence scores for clearer visual contrast.
        arr_1d = np.array(imp, dtype=float)
        mn = float(np.min(arr_1d))
        mx = float(np.max(arr_1d))
        if mx > mn:
            arr_1d = (arr_1d - mn) / (mx - mn)
        else:
            arr_1d = np.zeros_like(arr_1d)
        arr = arr_1d[None, :]
        ax.set_yticks([])
        seq_labels = list(seq.upper()[:L])
        if len(seq_labels) < L:
            seq_labels = seq_labels + [''] * (L - len(seq_labels))
        # remove up to two trailing blank positions (padded labels) if present
        trim = 0
        while trim < 2 and len(seq_labels) > 0 and seq_labels[-1] == '':
            seq_labels.pop()
            if arr.shape[1] > 0:
                arr = arr[:, : -1]
                L -= 1
            trim += 1
        ax.set_xticks(np.arange(L))
        ax.set_xticklabels(seq_labels, rotation=90, fontsize=8)
        ax.set_xlim(-0.5, L - 0.5)
        im = ax.imshow(arr, aspect='auto', cmap='Blues', vmin=0.0, vmax=1.0)

        # Draw full motif windows (triplets) that contain the global max.
        used = set()
        for motif, w in zip(motifs, strict_windows):
            key = (motif, tuple(w))
            if key in used:
                continue
            used.add(key)
            start = int(min(w))
            end = int(max(w))
            width = end - start + 1
            rect = Rectangle((start - 0.5, -0.5), width, 1.0,
                             linewidth=1.8, edgecolor='red', facecolor='none')
            ax.add_patch(rect)
            labels_with_global_max.append(motif)

        motif_text = ','.join(sorted(set(motifs)))
        title = f"motif: {motif_text}   seq: {seq}   gap(top1-top2): {gap:.4f}"
        ax.set_title(title, fontsize=10)

    if im is not None:
        # Place a single colorbar to the right
        fig.subplots_adjust(right=0.88)
        cax = fig.add_axes([0.90, 0.15, 0.02, 0.7])
        cb = fig.colorbar(im, cax=cax, orientation='vertical')
        cb.set_label('attention importance (normalized)')

    plt.tight_layout(rect=[0, 0, 0.9, 1.0])
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png, dpi=300)
    plt.close(fig)
    print('Wrote', out_png)
    # save labels where global max was inside the true-label windows
    out_globals = out_json.replace('_label_sequences.json', '_label_globals.json')
    os.makedirs(os.path.dirname(out_globals), exist_ok=True)
    with open(out_globals, 'w', encoding='utf-8') as fo:
        json.dump({'labels_with_global_max': labels_with_global_max}, fo, indent=2)
    print('Wrote', out_globals)

    out_matched = out_json.replace('_label_sequences.json', '_strict_matched.json')
    with open(out_matched, 'w', encoding='utf-8') as fo:
        json.dump({
            'selected_top5_by_gap': selected,
            'all_candidates_count': len(candidates),
        }, fo, indent=2)
    print('Wrote', out_matched)


for cat, labels in categories.items():
    jsonl = os.path.join(base_dir, f"{cat}.fasta.key_subsequences.jsonl")
    os.makedirs(output_dir, exist_ok=True)
    out_png = os.path.join(output_dir, f"{cat}_attention_map_top5_gap_norm.png")
    out_json = os.path.join(output_dir, f"{cat}_label_sequences.json")
    process_category(jsonl, out_png, out_json, labels)
