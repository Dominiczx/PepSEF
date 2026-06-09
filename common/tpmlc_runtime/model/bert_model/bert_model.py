import torch
from torch import nn

class BERTEncoder(nn.Module):
    def __init__(self, vocab_size, num_hiddens, num_heads, num_layers, ffn_num_hiddens, max_length, dropout):
        super(BERTEncoder, self).__init__()
        self.token_embedding = nn.Embedding(vocab_size, num_hiddens)
        self.position_embedding = nn.Embedding(max_length, num_hiddens)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=num_hiddens,
                nhead=num_heads,
                dim_feedforward=ffn_num_hiddens,
                dropout=dropout,
                batch_first=True
            ) for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids, attention_mask):
        seq_len = input_ids.size(1)
        position_ids = torch.arange(seq_len, device=input_ids.device, dtype=torch.long).unsqueeze(0)  # Ensure LongTensor
        embeddings = self.token_embedding(input_ids) + self.position_embedding(position_ids)
        embeddings = self.dropout(embeddings)

        for layer in self.layers:
            embeddings = layer(embeddings, src_key_padding_mask=~attention_mask.bool())
        return embeddings

class MaskLM(nn.Module):
    def __init__(self, num_hiddens, vocab_size):
        super(MaskLM, self).__init__()
        self.mlm_head = nn.Sequential(
            nn.Linear(num_hiddens, num_hiddens),
            nn.ReLU(),
            nn.LayerNorm(num_hiddens),
            nn.Linear(num_hiddens, vocab_size)
        )

    def forward(self, hidden_states, mlm_positions):
        # Ensure mlm_positions is of type LongTensor
        mlm_positions = mlm_positions.long()
        mlm_hidden_states = hidden_states[torch.arange(hidden_states.size(0)).unsqueeze(1), mlm_positions]
        return self.mlm_head(mlm_hidden_states)

class BERTModel(nn.Module):
    def __init__(self, vocab_size, num_hiddens, num_heads, num_layers, ffn_num_hiddens, max_length, dropout):
        super(BERTModel, self).__init__()
        self.encoder = BERTEncoder(vocab_size, num_hiddens, num_heads, num_layers, ffn_num_hiddens, max_length, dropout)
        self.mlm = MaskLM(num_hiddens, vocab_size)

    def forward(self, input_ids, attention_mask, mlm_positions=None):
        hidden_states = self.encoder(input_ids, attention_mask)
        if mlm_positions is not None:
            mlm_logits = self.mlm(hidden_states, mlm_positions)
            return hidden_states, mlm_logits
        return hidden_states, None
    
def initialize_weights(module):
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, mean=0, std=0.01)

