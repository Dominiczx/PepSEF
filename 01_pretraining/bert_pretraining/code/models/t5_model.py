import torch
from torch import nn

class T5Encoder(nn.Module):
    def __init__(self, vocab_size, hidden_dim, num_heads, num_layers, max_len, dropout):
        super(T5Encoder, self).__init__()
        self.token_embedding = nn.Embedding(vocab_size, hidden_dim)
        self.position_embedding = nn.Embedding(max_len, hidden_dim)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * 4,
                dropout=dropout,
                batch_first=True
            ) for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids, attention_mask):
        seq_len = input_ids.size(1)
        position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        embeddings = self.token_embedding(input_ids) + self.position_embedding(position_ids)
        embeddings = self.dropout(embeddings)

        for layer in self.layers:
            embeddings = layer(embeddings, src_key_padding_mask=~attention_mask.bool())
        return embeddings

class T5Decoder(nn.Module):
    def __init__(self, vocab_size, hidden_dim, num_heads, num_layers, max_len, dropout):
        super(T5Decoder, self).__init__()
        self.token_embedding = nn.Embedding(vocab_size, hidden_dim)
        self.position_embedding = nn.Embedding(max_len, hidden_dim)
        self.layers = nn.ModuleList([
            nn.TransformerDecoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * 4,
                dropout=dropout,
                batch_first=True
            ) for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)
        self.output_layer = nn.Linear(hidden_dim, vocab_size)

    def forward(self, input_ids, encoder_output, attention_mask, memory_mask):
        seq_len = input_ids.size(1)
        position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        embeddings = self.token_embedding(input_ids) + self.position_embedding(position_ids)
        embeddings = self.dropout(embeddings)

        for layer in self.layers:
            embeddings = layer(
                embeddings,
                encoder_output,
                tgt_key_padding_mask=~attention_mask.bool(),
                memory_key_padding_mask=~memory_mask.bool()
            )
        return self.output_layer(embeddings)

class T5Model(nn.Module):
    def __init__(self, vocab_size, hidden_dim, num_heads, num_layers, max_len, dropout):
        super(T5Model, self).__init__()
        self.encoder = T5Encoder(vocab_size, hidden_dim, num_heads, num_layers, max_len, dropout)
        self.decoder = T5Decoder(vocab_size, hidden_dim, num_heads, num_layers, max_len, dropout)
        self.lm_head = nn.Linear(hidden_dim, vocab_size)

    def forward(self, encoder_input_ids, decoder_input_ids, encoder_attention_mask, decoder_attention_mask):
        encoder_output = self.encoder(encoder_input_ids, encoder_attention_mask)
        decoder_output = self.decoder(decoder_input_ids, encoder_output, decoder_attention_mask, encoder_attention_mask)
        logits = self.lm_head(decoder_output)
        return decoder_output