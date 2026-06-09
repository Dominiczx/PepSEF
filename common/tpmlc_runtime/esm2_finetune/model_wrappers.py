import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import re
from utils.aff import CrossAttentionFusion
from utils.extractor import KeyPositionExtractor


class AttentionFeatureFusion(nn.Module):
    def __init__(self, feature_dim: int, d_model: int = 256, r: int = 4):
        super().__init__()
        inter_channels = max(1, d_model // max(1, int(r)))
        self.feature_proj = nn.Sequential(
            nn.Linear(feature_dim, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(inplace=True),
        )
        self.local_att = nn.Sequential(
            nn.Conv1d(d_model, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(inter_channels, d_model, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(d_model),
        )
        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(d_model, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(inter_channels, d_model, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(d_model),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor, feature: torch.Tensor):
        feature = self.feature_proj(feature.float())
        xa = x + feature
        xa_t = xa.transpose(1, 2)  # [B, D, L]
        wei = self.sigmoid(self.local_att(xa_t) + self.global_att(xa_t)).transpose(1, 2)
        return x * wei + feature * (1.0 - wei)


class ConcatFusion(nn.Module):
    def __init__(self, feature_dim: int, d_model: int = 256):
        super().__init__()
        self.feature_proj = nn.Sequential(
            nn.Linear(feature_dim, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(inplace=True),
        )
        self.out_proj = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor, feature: torch.Tensor):
        feat = self.feature_proj(feature.float())
        return self.out_proj(torch.cat([x, feat], dim=-1))


class PepTrainableESM2_AFF_LSTM(nn.Module):
    """
    Trainable wrapper for finetuned ESM2 that:
    - Accepts a loaded `esm_model` and `tokenizer`.
    - Freezes the ESM encoder parameters (optionally keep them frozen).
    - Uses `CrossAttentionFusion` to fuse ESM embeddings (original hidden size) with PSSM features.
    - Exposes the same forward signature used by fusion2: (seq_or_ids, pssm, attention_mask) -> (logits, key_scores)
    """

    def __init__(self, args, esm_config: dict, aff_config: dict, lstm_config: dict,
                 esm_model, tokenizer, multi_lstm: bool = False, freeze_esm: bool = True):
        super().__init__()
        self.args = args
        self.multi_lstm = multi_lstm
        self.esm_model = esm_model
        self.tokenizer = tokenizer
        # determine hidden size
        hidden_size = None
        try:
            cfg = getattr(self.esm_model, 'config', None)
            hidden_size = getattr(cfg, 'hidden_size', None) or getattr(cfg, 'd_model', None)
        except Exception:
            hidden_size = esm_config.get('hidden_size', 256) if isinstance(esm_config, dict) else 256
        if hidden_size is None:
            hidden_size = 256

        # freeze ESM encoder parameters by default
        if freeze_esm:
            for p in self.esm_model.parameters():
                p.requires_grad = False
            self.esm_model.eval()

        # use original ESM hidden size directly for fusion
        self.d_model = hidden_size

        # CrossAttentionFusion expects feature_dim from pssm/hmm
        if getattr(self.args, 'pssm_hmm', 'hmm') == 'both':
            feature_dim = 50
        elif getattr(self.args, 'pssm_hmm', 'hmm') == 'pssm':
            feature_dim = 20
        elif getattr(self.args, 'pssm_hmm', 'hmm') == 'hmm':
            feature_dim = 30
        else:
            feature_dim = 0

        self.feature_dim = feature_dim
        self.fusion_method = str(getattr(self.args, 'fusion_method', 'cross_attention')).strip().lower()
        if self.fusion_method in {'cross', 'crossattn', 'cross_attention_fusion'}:
            self.fusion_method = 'cross_attention'
        if self.fusion_method not in {'cross_attention', 'aff', 'concat'}:
            self.fusion_method = 'cross_attention'

        self.caf = None
        self.aff_fuser = None
        self.concat_fuser = None
        if feature_dim > 0:
            if self.fusion_method == 'cross_attention':
                self.caf = CrossAttentionFusion(args, feature_dim, d_model=self.d_model)
            elif self.fusion_method == 'aff':
                self.aff_fuser = AttentionFeatureFusion(feature_dim, d_model=self.d_model)
            elif self.fusion_method == 'concat':
                self.concat_fuser = ConcatFusion(feature_dim, d_model=self.d_model)
        self.key_extractor = KeyPositionExtractor(d_model=self.d_model)
        self.proj_norm = nn.LayerNorm(self.d_model)
        self.pssm_dropout = nn.Dropout(float(getattr(self.args, 'pssm_dropout', 0.1)))
        self.fusion_gate_logit = nn.Parameter(torch.tensor(float(getattr(self.args, 'fusion_alpha_init', -2.0))))

        # small mlp classifier to match downstream heads used in fusion
        self.mlp = nn.Sequential(
            nn.Linear(self.d_model, 1024),
            nn.LayerNorm(1024),
            nn.LeakyReLU(),
            nn.Linear(1024, 512),
            nn.LayerNorm(512),
            nn.LeakyReLU(),
            nn.Linear(512, 15)
        )

    def forward(self, seq_or_ids, pssm, attention_mask: Optional[torch.Tensor] = None):
        """
        seq_or_ids: either list[str] (raw sequences) or torch.Tensor of token ids
        pssm: [B, L, feat_dim]
        attention_mask: optional attention mask matching ids
        """
        # Allow caller to pass a single tuple (ids, pssm, attention_mask) to work with DataParallel
        if isinstance(seq_or_ids, (tuple, list)) and len(seq_or_ids) == 3 and isinstance(seq_or_ids[0], torch.Tensor):
            ids = seq_or_ids[0]
            pssm = seq_or_ids[1]
            attention_mask = seq_or_ids[2]
            try:
                param_dev = next(self.parameters()).device
            except StopIteration:
                param_dev = torch.device('cpu')
            ids = ids.to(param_dev)
            if attention_mask is not None:
                attention_mask = attention_mask.to(ids.device)
        elif isinstance(seq_or_ids, torch.Tensor):
            # seq_or_ids is already token ids tensor
            try:
                param_dev = next(self.parameters()).device
            except StopIteration:
                param_dev = torch.device('cpu')
            ids = seq_or_ids.to(param_dev)
            if attention_mask is not None:
                attention_mask = attention_mask.to(ids.device)
        else:
            # seq_or_ids is list[str]
            seq_or_ids = [re.sub(r"[UZOB]", "X", s) for s in seq_or_ids]
            max_len = None
            try:
                max_len = int(getattr(self.args, 'max_length', 0))
            except Exception:
                max_len = None
            if max_len is not None and max_len > 0:
                inputs = self.tokenizer(seq_or_ids, return_tensors='pt', padding=True, truncation=True, max_length=max_len + 2)
            else:
                inputs = self.tokenizer(seq_or_ids, return_tensors='pt', padding=True, truncation=True)
            try:
                param_dev = next(self.parameters()).device
            except StopIteration:
                param_dev = torch.device('cpu')
            ids = inputs.input_ids.to(param_dev)
            attention_mask = inputs.attention_mask.to(ids.device) if 'attention_mask' in inputs else None

        # get ESM hidden states (no grad by default when frozen)
        with torch.set_grad_enabled(any(p.requires_grad for p in self.esm_model.parameters())):
            out = self.esm_model(ids, attention_mask=attention_mask, output_hidden_states=True, return_dict=True)
            if hasattr(out, 'hidden_states') and out.hidden_states is not None:
                esm_hidden = out.hidden_states[-1]
            elif isinstance(out, tuple) and len(out) > 0:
                esm_hidden = out[0]
            else:
                esm_hidden = getattr(out, 'logits', out)

        # drop first/last embeddings (special tokens) before fusion
        if esm_hidden is not None and esm_hidden.shape[1] >= 2:
            esm_hidden = esm_hidden[:, 1:-1, :]
            if attention_mask is not None and attention_mask.shape[1] >= 2:
                attention_mask = attention_mask[:, 1:-1]

        # pad/truncate to model expected length if needed
        proj = self.proj_norm(esm_hidden)
        # ensure seq length matches pssm length by padding/truncating
        target_L = pssm.shape[1]
        if proj.shape[1] < target_L:
            pad_len = target_L - proj.shape[1]
            proj = F.pad(proj, (0, 0, 0, pad_len), mode='constant', value=0)
        elif proj.shape[1] > target_L:
            proj = proj[:, :target_L, :]

        # ensure attention mask aligns with projected key length (pad/truncate as needed)
        key_attn = None
        if attention_mask is not None:
            # attention_mask shape: [B, L_ids]; proj/key length is target_L
            key_attn = attention_mask
            if key_attn.shape[1] < target_L:
                pad_len = target_L - key_attn.shape[1]
                key_attn = F.pad(key_attn, (0, pad_len), value=0)
            elif key_attn.shape[1] > target_L:
                key_attn = key_attn[:, :target_L]
            key_attn = key_attn.to(proj.device)
        else:
            key_attn = torch.ones((proj.shape[0], proj.shape[1]), device=proj.device, dtype=torch.long)

        # fuse using CrossAttentionFusion when auxiliary features are available
        if self.feature_dim <= 0 or pssm is None or (hasattr(pssm, 'shape') and pssm.shape[-1] == 0):
            aff_result = proj
        else:
            pssm_in = pssm.float()
            if self.training:
                pssm_in = self.pssm_dropout(pssm_in)
            if self.fusion_method == 'cross_attention' and self.caf is not None:
                caf_out, attn_weights = self.caf(proj, pssm_in, attention_mask=key_attn)
                gate = torch.sigmoid(self.fusion_gate_logit)
                aff_result = proj + gate * (caf_out - proj)
            elif self.fusion_method == 'aff' and self.aff_fuser is not None:
                aff_result = self.aff_fuser(proj, pssm_in)
            elif self.fusion_method == 'concat' and self.concat_fuser is not None:
                aff_result = self.concat_fuser(proj, pssm_in)
            else:
                aff_result = proj
        key_scores = torch.sum(aff_result, 2) / (aff_result.shape[2] + 1e-8)
        x = F.leaky_relu(aff_result)
        # masked mean pooling to avoid averaging over padded positions
        if key_attn is not None:
            mask = key_attn.float().unsqueeze(-1)  # [B, L, 1]
            denom = torch.clamp(mask.sum(dim=1), min=1.0)
            x = (x * mask).sum(dim=1) / denom
        else:
            x = torch.sum(x, 1) / x.shape[1]
        logits = self.mlp(x)
        return logits, key_scores
