import os
import re
import matplotlib.pyplot as plt

task_dir = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
input_dir = os.environ.get('PEPSEF_SINGLE_LOG_DIR', os.path.join(task_dir, 'output_logs'))
output_dir = os.environ.get('PEPSEF_SINGLE_FIG_DIR', os.path.join(task_dir, 'figures'))
pattern_exp = re.compile(r'^exp\d+.*\.out$')
pattern_num = re.compile(r'^\d+.*\.out$')

results = []

os.makedirs(output_dir, exist_ok=True)

for fname in os.listdir(input_dir):
    if pattern_exp.match(fname) or pattern_num.match(fname):
        exp_name = fname.replace('.out', '')
        val_accs = []
        test_accs = []
        with open(os.path.join(input_dir, fname), 'r') as f:
            for line in f:
                # val acc
                m_val = re.search(r'val acc:([0-9\.]+)', line)
                if m_val:
                    val_accs.append(float(m_val.group(1)))
                # test acc
                m_test = re.search(r'test acc:([0-9\.]+)', line)
                if m_test:
                    test_accs.append(float(m_test.group(1)))
        if val_accs and test_accs:
            results.append({
                'exp': exp_name,
                'val_acc': max(val_accs),
                'test_acc': max(test_accs)
            })

def extract_exp_num(exp_name):
    m = re.match(r'(exp)?(\d+)', exp_name)
    return int(m.group(2)) if m else -1  # 返回数字用于排序

# 按实验编号排序
results = sorted(results, key=lambda r: extract_exp_num(r['exp']))

labels = [str(extract_exp_num(r['exp'])) for r in results]
val_accs = [r['val_acc'] for r in results]
test_accs = [r['test_acc'] for r in results]

group_size = 10
num_groups = (len(labels) + group_size - 1) // group_size

for g in range(num_groups):
    start = g * group_size
    end = min((g + 1) * group_size, len(labels))
    x = range(end - start)
    plt.figure(figsize=(max(8, (end-start)*0.8), 5))
    bars1 = plt.bar(x, val_accs[start:end], width=0.4, label='Max Val Acc', color='#6baed6')
    bars2 = plt.bar([i+0.4 for i in x], test_accs[start:end], width=0.4, label='Max Test Acc', color='#fd8d3c')
    plt.xticks([i+0.2 for i in x], labels[start:end], rotation=45, ha='right')
    plt.ylabel('Accuracy')
    plt.title(f'Experiments {start+1}-{end} Max Val/Test Accuracy')
    plt.legend(loc='upper right')  # 图例放右上角
    # 标注数值
    for i in x:
        plt.text(i, val_accs[start+i]+0.005, f'{val_accs[start+i]:.3f}', ha='center', va='bottom', fontsize=9)
        plt.text(i+0.4, test_accs[start+i]+0.005, f'{test_accs[start+i]:.3f}', ha='center', va='bottom', fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'comparison_{start+1}_{end}.png'))
    plt.show()
