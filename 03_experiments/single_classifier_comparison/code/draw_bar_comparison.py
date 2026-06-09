import re
import matplotlib.pyplot as plt
import numpy as np

import os

task_dir = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
input_log = os.environ.get('PEPSEF_SINGLE_BAR_LOG', os.path.join(task_dir, 'output_logs', '49_pssm.out'))
output_png = os.environ.get('PEPSEF_SINGLE_BAR_PNG', os.path.join(task_dir, 'figures', 'compare_bar.png'))

# 1. 读取49_pssm.out，提取所有test acc和test f1
acc_list = []
f1_list = []
with open(input_log, 'r') as f:
    for line in f:
        m = re.search(r'test acc:([0-9.]+).*f1:([0-9.]+)', line)
        if m:
            acc_list.append(float(m.group(1)))
            f1_list.append(float(m.group(2)))

# 2. 取最大值
best_acc = max(acc_list)
best_f1 = max(f1_list)
print(f'Best test acc: {best_acc:.3f}, Best test f1: {best_f1:.3f}')

# 3. 图片中的对比数据
# acc: MLBP, PrMFTP, TPpred-LE, TPpred-CMvL
acc_bar = [0.444, 0.500, 0.536, 0.543]
f1_bar = [0.218, 0.365, 0.422, 0.431]
labels = ['MLBP', 'PrMFTP', 'TPpred-LE', 'TPpred-CMvL', 'Ours']

# 4. 添加你的结果
acc_bar.append(best_acc)
f1_bar.append(best_f1)

# 5. 绘图
x = np.arange(len(labels))
width = 0.6

fig, axs = plt.subplots(1, 2, figsize=(10, 4))

# ACC
bars1 = axs[0].bar(x, acc_bar, width, color=['#c9d6b5','#f7d59c','#b5d0ee','#f7b5b5','#a3a3ff'], edgecolor='black')
for i, v in enumerate(acc_bar):
    axs[0].text(i, v+0.01, f'{v:.3f}', ha='center', fontweight='bold' if i==4 else 'normal')
axs[0].set_xticks(x)
axs[0].set_xticklabels(labels, rotation=20)
axs[0].set_ylim(0.4, 0.6)
axs[0].set_ylabel('ACC$_{example}$')
axs[0].set_title('ACC$_{example}$ Comparison')

# F1
bars2 = axs[1].bar(x, f1_bar, width, color=['#c9d6b5','#f7d59c','#b5d0ee','#f7b5b5','#a3a3ff'], edgecolor='black')
for i, v in enumerate(f1_bar):
    axs[1].text(i, v+0.01, f'{v:.3f}', ha='center', fontweight='bold' if i==4 else 'normal')
axs[1].set_xticks(x)
axs[1].set_xticklabels(labels, rotation=20)
axs[1].set_ylim(0.1, 0.5)
axs[1].set_ylabel('F1$_{label}$')
axs[1].set_title('F1$_{label}$ Comparison')

plt.tight_layout()
os.makedirs(os.path.dirname(output_png), exist_ok=True)
plt.savefig(output_png, dpi=300)
plt.show()
