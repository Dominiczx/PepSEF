import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

def plot_abp_sequences(sequences, weights, seq_labels=None, aa_labels=None, 
                      cmap='viridis', figsize=(12, 8)):
    """
    绘制ABP序列权重分布图
    
    参数：
    sequences : numpy数组 (batch, seq_length)
        氨基酸序列的字符数组
    weights : numpy数组 (batch, seq_length)
        每个位置的权重值
    seq_labels : list (batch,)
        序列标签（如["Seq1", "Seq2"]）
    aa_labels : list (batch, seq_length)
        氨基酸字符数组（如果sequences不是字符数组时需要）
    cmap : str
        颜色映射名称
    figsize : tuple
        画布尺寸
    """
    batch_size, seq_length = weights.shape[:2]
    
    # 创建画布和网格布局
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(batch_size + 1, 1, height_ratios=[1]*batch_size + [0.1])  # 最后一行给colorbar
    
    # 设置全局颜色规范
    vmin, vmax = np.min(weights), np.max(weights)
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    
    # 绘制每个序列的权重图
    axes = []
    for i in range(batch_size):
        ax = fig.add_subplot(gs[i, 0])
        axes.append(ax)
        
        # 显示权重热力图
        im = ax.imshow(weights[i][np.newaxis, :], cmap=cmap, aspect='auto', norm=norm)
        
        # 添加氨基酸标签
        if aa_labels is not None:
            seq_str = aa_labels[i]
        else:
            seq_str = sequences[i]  # 假设sequences是字符数组
            
        for j, aa in enumerate(seq_str):
            ax.text(j, 0, aa, ha='center', va='center', 
                   color='white' if weights[i,j] > (vmax+vmin)/2 else 'black')
        
        # 设置坐标轴
        ax.set_yticks([])
        if seq_labels is not None:
            ax.set_ylabel(seq_labels[i], rotation=0, ha='right', va='center')
        
        # 添加右侧数值标注（示例图中的-0.15等）
        ax.text(seq_length+0.5, 0, f"{np.mean(weights[i]):.2f}", 
               ha='left', va='center')
        
        ax.set_xlim(-0.5, seq_length-0.5)
    
    # 添加共用colorbar
    cbar_ax = fig.add_subplot(gs[batch_size, :])
    plt.colorbar(im, cax=cbar_ax, orientation='horizontal')
    cbar_ax.set_xlabel('Weights Scale', labelpad=10)
    
    plt.tight_layout()
    return fig, axes

# 示例使用
# if __name__ == "__main__":
#     # 生成模拟数据
#     batch_size = 4
#     seq_length = 30
#     sequences = np.array([
#         'GIFSVKGAAPLAGKGLAEKGKGLELIACKIAKQC',
#         'GLMSLFRGVLKTAGKHIFKNVGGSLLDQAKCKITGEC',
#         'FTLKKSQLLLLFFLGTINFSLOEERNDYPEERDYPEERDSYEKDVEK',
#         'GIFUKDKLIGKALLEGAVQRRTDATIQTAVAQAAANVAATAKQ'
#     ])
#     weights = np.random.randn(batch_size, seq_length) * 0.1 + 0.1
    
#     # 调用绘图函数
#     fig, axes = plot_abp_sequences(
#         sequences=sequences,
#         weights=weights,
#         seq_labels=[f"Seq{i+1}" for i in range(batch_size)],
#         cmap='YlGnBu',
#         figsize=(12, 8)
#     )
    
#     # 保存图像
#     plt.savefig('abp_weights_plot.png', dpi=300, bbox_inches='tight')
#     plt.show()

def plot_key_scores(peptides, key_scores, output_path=None):
    """
    Plot a line chart for normalized key scores of each amino acid position in peptides.

    Args:
        peptides (list): List of peptide sequences.
        key_scores (numpy.ndarray): Array of key scores for each position (shape: [num_peptides, max_length]).
        output_path (str): Path to save the plot. If None, the plot will be displayed.
    """
    for i, (peptide, scores) in enumerate(zip(peptides, key_scores)):
        # Trim scores to match peptide length
        scores = scores[0][:len(peptide)]
        
        # Check if scores are valid
        if np.min(scores) == np.max(scores):
            print(f"Warning: All scores are the same for peptide {i + 1}. Skipping normalization.")
            normalized_scores = scores  # No normalization needed
        else:
            # Normalize the scores
            normalized_scores = (scores - np.min(scores)) / (np.max(scores) - np.min(scores))

        # Plot the line chart
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(peptide) + 1), normalized_scores, marker='o', label='Normalized Key Score', color='red')
        plt.xticks(range(1, len(peptide) + 1), list(peptide), fontsize=10)
        plt.xlabel('Amino Acid Position', fontsize=12)
        plt.ylabel('Normalized Key Score', fontsize=12)
        plt.title(f'Key Scores for Peptide {i + 1}', fontsize=14)
        plt.grid(True)
        plt.legend()

        # Save or show the plot
        if output_path:
            os.makedirs(output_path, exist_ok=True)  # Ensure the directory exists
            plt.savefig(f'{output_path}/peptide_{i + 1}.png', dpi=300, bbox_inches='tight')
        else:
            plt.show()
        plt.close()

def plot_key_scores_and_delta_G(peptides, key_scores, delta_G, output_path=None):
    """
    Plot a line chart for normalized key scores and delta_G values for each amino acid position in peptides.

    Args:
        peptides (list): List of peptide sequences.
        key_scores (list): List of key scores for each peptide (shape: [num_peptides, max_length]).
        delta_G (list): List of delta_G values for each peptide.
        output_path (str): Path to save the plot. If None, the plot will be displayed.
    """
    for i, (peptide, scores, delta_g) in enumerate(zip(peptides, key_scores, delta_G)):
        # Trim scores to match peptide length
        scores = scores[0][:len(peptide)]
        
        # Normalize key_scores
        if np.min(scores) != np.max(scores):
            normalized_scores = (scores - np.min(scores)) / (np.max(scores) - np.min(scores))
        else:
            normalized_scores = scores  # No normalization needed

        # Normalize delta_G
        if np.min(delta_g) != np.max(delta_g):
            normalized_delta_g = (delta_g - np.min(delta_g)) / (np.max(delta_g) - np.min(delta_g))
        else:
            normalized_delta_g = delta_g  # No normalization needed

        # Create the plot
        fig, ax1 = plt.subplots(figsize=(10, 6))

        # print(len(peptide), len(normalized_scores), len(normalized_delta_g))
        # Plot key_scores on the left y-axis
        ax1.plot(range(1, len(peptide) + 1), normalized_scores, marker='o', label='Normalized Key Scores', color='red')
        ax1.set_xlabel('Amino Acid Position', fontsize=12)
        ax1.set_ylabel('Normalized Key Scores', fontsize=12, color='red')
        ax1.tick_params(axis='y', labelcolor='red')
        ax1.set_xticks(range(1, len(peptide) + 1))
        ax1.set_xticklabels(list(peptide), fontsize=10)

        # Plot delta_G on the right y-axis
        ax2 = ax1.twinx()
        ax2.plot(range(1, len(peptide) + 1), normalized_delta_g, marker='o', label='Normalized Delta G', color='blue')
        ax2.set_ylabel('Normalized Delta G', fontsize=12, color='blue')
        ax2.tick_params(axis='y', labelcolor='blue')

        # Add title and legend
        plt.title(f'Key Scores and Delta G for Peptide {i + 1}', fontsize=14)
        fig.tight_layout()

        # Save or show the plot
        if output_path:
            os.makedirs(output_path, exist_ok=True)  # Ensure the directory exists
            plt.savefig(f'{output_path}/peptide_{i + 1}_comparison.png', dpi=300, bbox_inches='tight')
        else:
            plt.show()
        plt.close()