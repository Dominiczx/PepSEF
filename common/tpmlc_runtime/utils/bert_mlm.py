from torch import nn
import torch
from d2l import torch as d2l
from torch import Tensor
from typing import Optional
from utils.aff import AFF

class BERTEncoder(nn.Module):
    """BERT编码器"""
    def __init__(self, vocab_size, num_hiddens, norm_shape, ffn_num_input,
                 ffn_num_hiddens, num_heads, num_layers, dropout,
                 max_len=256, key_size=1024, query_size=1024, value_size=1024,
                 **kwargs):
        super(BERTEncoder, self).__init__(**kwargs)
        self.token_embedding = nn.Embedding(vocab_size, num_hiddens)
        self.blks = nn.Sequential()
        for i in range(num_layers): # 加入多少个EncoderBlock
            self.blks.add_module(f"{i}", d2l.EncoderBlock(
                key_size, query_size, value_size, num_hiddens, norm_shape,
                ffn_num_input, ffn_num_hiddens, num_heads, dropout, True))
        # 在BERT中，位置嵌入是可学习的，因此我们创建一个足够长的位置嵌入参数
        # batch_size=1，随机初始化pos_embedding
        self.pos_embedding = nn.Parameter(torch.randn(1, max_len,
                                                      num_hiddens))

    def forward(self, tokens, valid_lens=None):
        # 在以下代码段中，X的形状保持不变：（批量大小，最大序列长度，num_hiddens）
        X = self.token_embedding(tokens)
        X = X + self.pos_embedding.data[:, :X.shape[1], :]
        # 以上两行代码是对输入X进行处理
        for blk in self.blks: # 接着让X输入到之前设置好的blk中，自行计算
            X = blk(X, valid_lens)
        return X

class MaskLM(nn.Module):
    """BERT的掩蔽语言模型任务"""
    def __init__(self, vocab_size, num_hiddens, num_inputs=1024, **kwargs):
        super(MaskLM, self).__init__(**kwargs)
        self.mlp = nn.Sequential(nn.Linear(num_inputs, num_hiddens),
                                 nn.ReLU(),
                                 nn.LayerNorm(num_hiddens),
                                 nn.Linear(num_hiddens, vocab_size))

    def forward(self, X, pred_positions):
        num_pred_positions = pred_positions.shape[1]
        pred_positions = pred_positions.reshape(-1)
        batch_size = X.shape[0]
        batch_idx = torch.arange(0, batch_size)
        # 假设batch_size=2，num_pred_positions=3
        # 那么batch_idx是np.array（[0,0,0,1,1]）
        batch_idx = torch.repeat_interleave(batch_idx, num_pred_positions)
        masked_X = X[batch_idx, pred_positions]
        masked_X = masked_X.reshape((batch_size, num_pred_positions, -1))
        mlm_Y_hat = self.mlp(masked_X)
        return mlm_Y_hat
    
class BERTModel(nn.Module):
    """BERT模型"""
    def __init__(self, vocab_size, num_hiddens, norm_shape, ffn_num_input,
                 ffn_num_hiddens, num_heads, num_layers, dropout,
                 max_len=256, key_size=256, query_size=256, value_size=256,
                 hid_in_features=256, mlm_in_features=256):
        super(BERTModel, self).__init__()
        self.encoder = BERTEncoder(vocab_size, num_hiddens, norm_shape,
                    ffn_num_input, ffn_num_hiddens, num_heads, num_layers,
                    dropout, max_len=max_len, key_size=key_size,
                    query_size=query_size, value_size=value_size)
        self.hidden = nn.Sequential(nn.Linear(hid_in_features, num_hiddens),
                                    nn.Tanh())
        self.mlm = MaskLM(vocab_size, num_hiddens, mlm_in_features)

    def forward(self, tokens, valid_lens=None,
                pred_positions=None):
        encoded_X = self.encoder(tokens, valid_lens)
        if pred_positions is not None:
            mlm_Y_hat = self.mlm(encoded_X, pred_positions)
        else:
            mlm_Y_hat = None
        return encoded_X, mlm_Y_hat
    
# mlm = MaskLM(vocab_size, num_hiddens)
# mlm_positions = torch.tensor([[1, 5, 2], [6, 1, 5]])
# mlm_Y_hat = mlm(encoded_X, mlm_positions)
# print(mlm_Y_hat.shape) #torch.Size([2, 3, 10000])

# mlm_Y = torch.tensor([[7, 8, 9], [10, 20, 30]])
# loss = nn.CrossEntropyLoss(reduction='none')
# mlm_l = loss(mlm_Y_hat.reshape((-1, vocab_size)), mlm_Y.reshape(-1))
# print(mlm_l.shape) #torch.Size([6])

class TransformerDecoderLayer(nn.Module):
    """
    Transformer decoder
    """
    __constants__ = ['batch_first']

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, layer_norm_eps=1e-5, batch_first=False, device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(TransformerDecoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first,
                                            **factory_kwargs)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first,
                                                 **factory_kwargs)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward, **factory_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, **factory_kwargs)

        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm3 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = nn.ReLU()

    def forward(self, tgt: Tensor, memory: Tensor, tgt_mask: Optional[Tensor] = None, memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None, memory_key_padding_mask: Optional[Tensor] = None):
        """
        Args:
            tgt: the sequence to the decoder layer (required).
            memory: the sequence from the last layer of the encoder (required).
            tgt_mask: the mask for the tgt sequence (optional).
            memory_mask: the mask for the memory sequence (optional).
            tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
            memory_key_padding_mask: the mask for the memory keys per batch (optional).

        """
        tgt2, att_tgt = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2, att_cross = self.multihead_attn(tgt, memory, memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt, att_tgt, att_cross
    
class Bert_Decoder(nn.Module):
    def __init__(self, encoder, vocab_size, num_hiddens, norm_shape, ffn_num_input, 
                 ffn_num_hiddens, num_heads, num_layers, dropout, max_len=128, key_size=256, query_size=256, value_size=256,d_model=256, nhead=4, n_dec_layers=12):
        super(Bert_Decoder, self).__init__()
        self.label_embedding = nn.Embedding(15, d_model)
        self.encoder = encoder
        # self.encoder = BERTEncoder(vocab_size, num_hiddens, norm_shape,
        #             ffn_num_input, ffn_num_hiddens, num_heads, num_layers,
        #             dropout, max_len=max_len, key_size=key_size,
        #             query_size=query_size, value_size=value_size)
        self.decoder_layers = nn.ModuleList(
            [TransformerDecoderLayer(d_model=d_model, nhead=nhead, batch_first=True, dim_feedforward=1024, dropout=dropout) for _ in range(n_dec_layers)]
        )
        self.fcs = nn.ModuleList([
            nn.Sequential(

                nn.Linear(d_model, 1),
                nn.Sigmoid()
            )
            for _ in range(15)
        ])

        self._reset_parameters()
        
    def _reset_parameters(self):
        r"""Initiate parameters in the transformer model."""

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        
    def forward(self, tokens, labels, valid_lens=None, att_mask=None):
        encoded_X = self.encoder(tokens, valid_lens)
        
        atts_tgt = []
        atts_cross = []
        y = self.label_embedding(labels)
        for i, decoder in enumerate(self.decoder_layers):
            y, att_tgt, att_cross = decoder(y, encoded_X, tgt_mask=att_mask)
            atts_tgt.append(att_tgt)
            atts_cross.append(att_cross)
        outputs = []
        for i, fc in enumerate(self.fcs):
            output = fc(y[:, i, :])    # (batch_size, d_model) * (d_mode, 1)
            outputs.append(output)
        outputs = torch.cat(outputs, dim=-1)
        return outputs
    
