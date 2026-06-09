#!/usr/bin/env python3
import os
import csv
import glob
import json
import numpy as np
import matplotlib.pyplot as plt

script_dir = os.path.dirname(os.path.abspath(__file__))
default_cmp_dir = os.path.normpath(os.path.join(script_dir, '..', 'results', 'cmp_output'))
cmp_dir = os.environ.get('PEPSEF_CMP_DIR', default_cmp_dir)
csv_path = os.path.join(cmp_dir, 'results.csv')
out_png = os.path.join(cmp_dir, 'results_comparison.png')

models = []
val_acc = []
test_acc = []
if not os.path.exists(csv_path):
    rows = []
    for json_path in sorted(glob.glob(os.path.join(cmp_dir, '*_result.json'))):
        with open(json_path, 'r', encoding='utf-8') as jf:
            item = json.load(jf)
        rows.append({
            'model': item.get('model', os.path.basename(json_path).replace('_result.json', '')),
            'val_acc': item.get('val_acc', item.get('val_metrics', {}).get('instance_acc')),
            'test_acc': item.get('test_acc', item.get('test_metrics', {}).get('instance_acc')),
        })
    if rows:
        with open(csv_path, 'w', encoding='utf-8', newline='') as cf:
            writer = csv.DictWriter(cf, fieldnames=['model', 'val_acc', 'test_acc'])
            writer.writeheader()
            writer.writerows(rows)

with open(csv_path, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        models.append(row['model'])
        val_acc.append(float(row['val_acc']))
        test_acc.append(float(row['test_acc']))

if len(models) == 0:
    raise SystemExit('No data found in ' + csv_path)

# Convert to numpy arrays
val = np.array(val_acc)
test = np.array(test_acc)

# Newer Matplotlib versions renamed bundled seaborn styles. Use the
# available name when present and fall back to the default style otherwise.
for style_name in ('seaborn-v0_8-muted', 'seaborn-muted', 'default'):
    if style_name == 'default' or style_name in plt.style.available:
        plt.style.use(style_name)
        break
N = len(models)
ind = np.arange(N)
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(10, 4), sharey=True)

# left: validation accuracies
bars1 = ax1.bar(ind, val, color='#4C78A8', width=0.7)
ax1.set_title('Validation Accuracy')
ax1.set_xticks(ind)
ax1.set_xticklabels(models, rotation=30, ha='right')
ax1.set_ylim(0, max(max(val), max(test)) * 1.12)
ax1.set_ylabel('Accuracy')

# right: test accuracies
bars2 = ax2.bar(ind, test, color='#F58518', width=0.7)
ax2.set_title('Test Accuracy')
ax2.set_xticks(ind)
ax2.set_xticklabels(models, rotation=30, ha='right')

# annotate bars
def annotate_ax(ax, bars):
    for b in bars:
        h = b.get_height()
        ax.annotate(f'{h:.3f}',
                    xy=(b.get_x() + b.get_width() / 2, h),
                    xytext=(0, 4),
                    textcoords='offset points',
                    ha='center', va='bottom', fontsize=9)

annotate_ax(ax1, bars1)
annotate_ax(ax2, bars2)

plt.suptitle('Model accuracy comparison')
plt.tight_layout(rect=[0, 0, 1, 0.95])
os.makedirs(os.path.dirname(out_png), exist_ok=True)
plt.savefig(out_png, dpi=300)
plt.close(fig)
print('Wrote', out_png)
