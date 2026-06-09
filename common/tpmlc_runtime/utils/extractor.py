import torch
import torch.nn as nn
import torch.nn.functional as F

class KeyPositionExtractor(nn.Module):
    def __init__(self, d_model):
        """
        Extracts key positions based on attention scores.
        Args:
            d_model: Dimensionality of the model (e.g., 256).
        """
        super(KeyPositionExtractor, self).__init__()
        self.attention_score_proj = nn.Linear(d_model, 1)  # Project features to a single attention score per position

    def forward(self, features):
        """
        Args:
            features: Tensor of shape [batch, seq_len, d_model].
        Returns:
            key_scores: Tensor of shape [batch, seq_len] containing attention scores for each position.
            enhanced_features: Tensor of shape [batch, seq_len, d_model] with weighted features.
        """
        # Compute attention scores
        key_scores = self.attention_score_proj(features).squeeze(-1)  # [batch, seq_len]
        key_scores = F.softmax(key_scores, dim=-1)  # Normalize scores across the sequence

        # Compute enhanced features using attention scores
        enhanced_features = features * key_scores.unsqueeze(-1)  # [batch, seq_len, d_model]

        return key_scores, enhanced_features
    
class KeyPositionExtractor2(nn.Module):
    def __init__(self, d_model, n_heads=8, conv_kernel=5):
        super().__init__()
        # 多头注意力层（捕捉长程依赖）
        self.multihead_attn = nn.MultiheadAttention(d_model, n_heads)
        
        # 门控卷积层（捕捉局部模式）
        self.conv_gate = nn.Sequential(
            nn.Conv1d(d_model, d_model*2, conv_kernel, padding=conv_kernel//2),
            nn.GLU(dim=1)  # 门控线性单元
        )
        
        # 权重融合层
        self.fusion = nn.Linear(d_model*2, 1)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        """
        x: [batch_size, seq_len, d_model]
        return: [batch_size, seq_len] (关键分数)
        """
        # 残差连接保留原始信息
        residual = x
        
        # 多头注意力计算
        attn_output, _ = self.multihead_attn(x, x, x)  # [batch, seq, d_model]
        
        # 门控卷积计算
        conv_output = self.conv_gate(x.transpose(1,2)).transpose(1,2)  # [batch, seq, d_model]
        
        # 特征拼接与融合
        combined = torch.cat([attn_output, conv_output], dim=-1)  # [batch, seq, 2*d_model]
        weights = torch.sigmoid(self.fusion(combined)).squeeze(-1)  # [batch, seq]
        
        # 层归一化输出
        output = self.layer_norm(residual + attn_output)
        return weights, output