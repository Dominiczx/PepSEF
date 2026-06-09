import matplotlib.pyplot as plt
import numpy as np

# 数据（A对应2，R对应3，...，V对应20）
amino_acids = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']
numbers = list(range(2, 21))  # 2到20

# 颜色方案（字母圈和数字圈各自一组渐变色）
letter_colors = plt.cm.tab20(np.linspace(0, 1, len(amino_acids)))
number_colors = plt.cm.rainbow(np.linspace(0, 1, len(numbers)))

fig, ax = plt.subplots(figsize=(6,6))
ax.set_xlim(-1.2, 1.2)
ax.set_ylim(-1.2, 1.2)
ax.axis('off')

# 外圈（氨基酸）
radius_outer = 1.0
for i, aa in enumerate(amino_acids):
    angle = 2 * np.pi * i / len(amino_acids)
    x = radius_outer * np.cos(angle)
    y = radius_outer * np.sin(angle)
    ax.text(x, y, aa, fontsize=18, fontweight='bold', color=letter_colors[i], ha='center', va='center')

# 中圈（编号）
radius_inner = 0.8
for i, num in enumerate(numbers):
    angle = 2 * np.pi * i / len(numbers)
    x = radius_inner * np.cos(angle)
    y = radius_inner * np.sin(angle)
    ax.text(x, y, str(num), fontsize=16, fontweight='bold', color=number_colors[i], ha='center', va='center')

# 中心透明（不填充任何圆）
plt.tight_layout()
plt.savefig('amino_acid_token_circle.png', dpi=300, transparent=True)
plt.show()