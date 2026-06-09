import torch
import torch.nn as nn
import re
import os

# from utils.bert_mlm import BERTModel, BERTEncoder
from utils.bert_model import BERTModel
from utils.aff import AFF, new_AFF, iAFF, CrossAttentionFusion
from utils.lstm import LSTM_ML
from utils.extractor import KeyPositionExtractor
try:
    from motif_plot import plot_abp_sequences
except Exception:
    plot_abp_sequences = None

import torch.nn.functional as F
import numpy as np
from transformers import BertForMaskedLM, BertModel, EsmForMaskedLM, EsmTokenizer, T5EncoderModel, T5Tokenizer
try:
    from esm import pretrained
except Exception:
    pretrained = None


class BERT_AFF(nn.Module):
    def __init__(self, args, channels, r, vocab_size, num_hiddens, norm_shape, ffn_num_input,
                 ffn_num_hiddens, num_heads, num_layers, dropout, 
                 max_len, key_size, query_size, value_size, **kwargs):
        super(BERT_AFF, self).__init__()
        self.aff = new_AFF(args, channels=channels, r=r).to(args.device)
        self.encoder = BERTModel(vocab_size, num_hiddens, norm_shape,
                    ffn_num_input, ffn_num_hiddens, num_heads, num_layers,
                    dropout, max_len=max_len, key_size=key_size,
                    query_size=query_size, value_size=value_size, **kwargs)
        # self.checkpoint = torch.load('./model/bert_lstm/pytorch_model_fcs8.bin')
        # self.encoder.load_state_dict(self.checkpoint['encoder'])
        self.checkpoint = torch.load('model/pretrain_bert/pytorch_model.bin')
        self.encoder.load_state_dict(self.checkpoint)
        self.encoder = self.encoder.encoder
        
    def forward(self, data, attention_mask=None):
        encoded_X = self.encoder(seq)
        fusion = self.aff(encoded_X, pssm)
        return fusion
    
class BERT_AFF_Test(nn.Module):
    def __init__(self, args, channels, r, vocab_size, num_hiddens, norm_shape, ffn_num_input,
                 ffn_num_hiddens, num_heads, num_layers, dropout, 
                 max_len, key_size, query_size, value_size, **kwargs):
        super(BERT_AFF_Test, self).__init__()
        self.aff = new_AFF(args, channels=channels, r=r).to(args.device)
        self.encoder = BERTModel(vocab_size, num_hiddens, norm_shape,
                    ffn_num_input, ffn_num_hiddens, num_heads, num_layers,
                    dropout, max_len=max_len, key_size=key_size,
                    query_size=query_size, value_size=value_size, **kwargs)
        self.encoder = self.encoder.encoder
        self.checkpoint = torch.load('model/bert_pssm/aff_lstm16.bin')
        self.encoder.load_state_dict(self.checkpoint['encoder'])
        
    def forward(self, seq, pssm):
        encoded_X = self.encoder(seq)
        fusion = self.aff(encoded_X, pssm)         
        return fusion
    
class ProtBERT_AFF(nn.Module):
    def __init__(self, args, channels, r, **kwargs):
        """
        ProtBERT adapter that uses CrossAttentionFusion to fuse ProtBERT embeddings with PSSM.
        """
        super(ProtBERT_AFF, self).__init__()
        self.args = args
        # feature dim depends on pssm/hmm choice like in Bert_AFF
        if args.pssm_hmm == 'both':
            feature_dim = 50
        elif args.pssm_hmm == 'pssm':
            feature_dim = 20
        elif args.pssm_hmm == 'hmm':
            feature_dim = 30
        elif args.pssm_hmm == 'none':
            feature_dim = 0
        # Load HF ProtBERT encoder (local folder name used elsewhere is 'model/ProtBert')
        self.encoder = BertModel.from_pretrained('model/ProtBert').to(args.device)
        # project encoder hidden dim to CAF expected d_model (256)
        try:
            enc_h = self.encoder.config.hidden_size
        except Exception:
            enc_h = None
        if enc_h is not None and enc_h != 256:
            self.proj = nn.Linear(enc_h, 256).to(args.device)
        else:
            self.proj = None
        # Cross-attention fusion module
        self.caf = CrossAttentionFusion(args, feature_dim)

    def forward(self, data, attention_mask=None):
        # data is (seq, pssm)
        seq, pssm = data[0], data[1]
        # encoded_X: [B, L, H]
        encoded_X = self.encoder(seq, attention_mask=attention_mask)[0]
        # If encoder hidden dim != 256, project to 256 for CAF
        if hasattr(self, 'proj') and self.proj is not None:
            encoded_X = self.proj(encoded_X)
        aff_result, attn_weights = self.caf(encoded_X, pssm, attention_mask=attention_mask)
        return aff_result, attn_weights

class ESM2_AFF(nn.Module):
    def __init__(self, args, channels, r, esm_model_path="model/esm2"):
        """
        ESM2-based AFF model.
        Args:
            args: Arguments for the model.
            channels: Number of channels for AFF.
            r: Reduction ratio for AFF.
            esm_model_path: Path to the pre-downloaded ESM2 model directory.
        """
        super(ESM2_AFF, self).__init__()
        self.args = args
        # We'll use CrossAttentionFusion (same pattern as Bert_AFF) instead of original AFF
        self.args = args
        if args.pssm_hmm == 'both':
            feature_dim = 50
        elif args.pssm_hmm == 'pssm':
            feature_dim = 20
        elif args.pssm_hmm == 'hmm':
            feature_dim = 30
        elif args.pssm_hmm == 'none':
            feature_dim = 0
        self.feature_dim = feature_dim
        self.caf = CrossAttentionFusion(args, feature_dim) if feature_dim > 0 else None

        # Load the pretrained ESM2 model and tokenizer from the local directory
        self.esm_model = EsmForMaskedLM.from_pretrained(esm_model_path).to(args.device)
        self.tokenizer = EsmTokenizer.from_pretrained(esm_model_path)
        self.esm_model.eval()  # Set ESM2 to evaluation mode
        # projection to 256 if needed (keep original projection for compatibility)
        self.esm_256 = nn.Linear(self.esm_model.config.hidden_size if hasattr(self.esm_model, 'config') else 33, 256)

    def forward(self, data, attention_mask=None):
        """
        Forward pass for ESM2_AFF.
        Args:
            seq: Input sequence tensor [batch, seq_len].
            pssm: PSSM matrix [batch, seq_len, 20].
        Returns:
            Fusion of ESM2 embeddings and PSSM features.
        """
        seq, pssm = data[0], data[1]
        # If caller provided token ids tensor, use it directly; otherwise tokenize list[str]
        if isinstance(seq, torch.Tensor):
            ids = seq.to(self.args.device)
            if attention_mask is not None and isinstance(attention_mask, torch.Tensor):
                attention_mask = attention_mask.to(self.args.device)
        else:
            inputs = self.tokenizer(seq, return_tensors="pt", padding=True, truncation=True)
            ids = inputs.input_ids.to(self.args.device)
            attention_mask = inputs.attention_mask.to(self.args.device) if 'attention_mask' in inputs else None
        # Generate embeddings using ESM2 — request hidden states (last layer) rather than logits
        with torch.no_grad():  # Disable gradient computation for ESM2
            model_out = self.esm_model(ids, attention_mask=attention_mask, output_hidden_states=True, return_dict=True)
            # prefer last hidden state from hidden_states if available
            if hasattr(model_out, 'hidden_states') and model_out.hidden_states is not None:
                esm_hidden = model_out.hidden_states[-1]
            elif isinstance(model_out, tuple) and len(model_out) > 0:
                esm_hidden = model_out[0]
            else:
                # fallback to logits as last resort (may be vocab logits)
                esm_hidden = getattr(model_out, 'logits', model_out)
            esm_embeddings = self.esm_256(esm_hidden)
            # pad/truncate to length 128 if needed
            if esm_embeddings.shape[1] < 128:
                pad_length = 128 - esm_embeddings.shape[1]
                esm_embeddings = F.pad(esm_embeddings, (0, 0, 0, pad_length, 0, 0), mode='constant', value=0)
            elif esm_embeddings.shape[1] > 128:
                esm_embeddings = esm_embeddings[:, :128, :]

        # pad/truncate attention_mask to match sequence length (128)
        if attention_mask is not None:
            am = attention_mask
            if am.shape[1] < 128:
                pad_len = 128 - am.shape[1]
                pad = torch.zeros((am.shape[0], pad_len), dtype=am.dtype, device=am.device)
                attention_mask = torch.cat([am, pad], dim=1)
            elif am.shape[1] > 128:
                attention_mask = am[:, :128]

        # Fuse using CrossAttentionFusion when PSSM/HHM features are available
        if self.caf is None or pssm is None or (hasattr(pssm, 'shape') and pssm.shape[-1] == 0):
            # no auxiliary features: return esm embeddings directly
            attn_weights = None
            return esm_embeddings, attn_weights
        aff_result, attn_weights = self.caf(esm_embeddings, pssm, attention_mask=attention_mask)
        return aff_result, attn_weights
    
class ProtT5_AFF(nn.Module):
    def __init__(self, args, channels, r, prot_t5_model_path="model/Prott5"):
        """
        ProtT5-based AFF model.
        Args:
            args: Arguments for the model.
            channels: Number of channels for AFF.
            r: Reduction ratio for AFF.
            prot_t5_model_path: Path to the pre-downloaded ProtT5 model directory.
        """
        super(ProtT5_AFF, self).__init__()
        self.args = args
        self.args = args
        if args.pssm_hmm == 'both':
            feature_dim = 50
        elif args.pssm_hmm == 'pssm':
            feature_dim = 20
        elif args.pssm_hmm == 'hmm':
            feature_dim = 30
        elif args.pssm_hmm == 'none':
            feature_dim = 0
        self.caf = CrossAttentionFusion(args, feature_dim)

        # Load the pretrained ProtT5 model and tokenizer from the local directory
        self.prot_t5_model = T5EncoderModel.from_pretrained(prot_t5_model_path).to(args.device)
        self.tokenizer = T5Tokenizer.from_pretrained(prot_t5_model_path)
        self.prot_t5_model.eval()  # Set ProtT5 to evaluation mode
        self.prot_t5_256 = nn.Linear(self.prot_t5_model.config.d_model if hasattr(self.prot_t5_model, 'config') else 768, 256)

    def forward(self, data, attention_mask=None):
        """
        Forward pass for ProtT5_AFF.
        Args:
            seq: Input sequence tensor [batch, seq_len].
            pssm: PSSM matrix [batch, seq_len, 20].
        Returns:
            Fusion of ProtT5 embeddings and PSSM features.
        """
        seq, pssm = data[0], data[1]
        # Tokenize the input sequences
        seqs = [" ".join(list(re.sub(r"[UZOB]", "X", sequence))) for sequence in seq]
        seqs = ["<AA2fold>" + " " + s if s.isupper() else "<fold2AA>" + " " + s for s in seqs]
        inputs = self.tokenizer.batch_encode_plus(seqs, add_special_tokens=True, padding="longest", return_tensors='pt')
        # Generate embeddings using ProtT5
        with torch.no_grad():  # Disable gradient computation for ProtT5
            model_out = self.prot_t5_model(**{k: v.to(self.args.device) for k, v in inputs.items()})
            # obtain last_hidden_state
            if hasattr(model_out, 'last_hidden_state'):
                last = model_out.last_hidden_state
            elif isinstance(model_out, tuple) and len(model_out) > 0:
                last = model_out[0]
            else:
                last = model_out
            prot_t5_embeddings = self.prot_t5_256(last)
            if prot_t5_embeddings.shape[1] < 128:
                pad_length = 128 - prot_t5_embeddings.shape[1]
                prot_t5_embeddings = F.pad(prot_t5_embeddings, (0, 0, 0, pad_length, 0, 0), mode='constant', value=0)
            elif prot_t5_embeddings.shape[1] > 128:
                prot_t5_embeddings = prot_t5_embeddings[:, :128, :]
        # Fuse ProtT5 embeddings with PSSM features using CrossAttentionFusion
        aff_result, attn_weights = self.caf(prot_t5_embeddings, pssm, attention_mask=attention_mask)
        return aff_result, attn_weights

class Bert_AFF(nn.Module):
    def __init__(self, args, bert_config, aff_config) -> None:
        super(Bert_AFF, self).__init__()
        self.args = args
        self.bert = BERTModel(vocab_size=args.vocab_size, **bert_config)
        # self.bert.load_state_dict(torch.load('model/pretrain_bert/pytorch_model.bin'))
        # self.encoder = self.bert.encoder
        self.sigmoid = nn.Sigmoid()
        # self.checkpoint = torch.load('model/bert_lstm/bert_lstm_model2.bin')
        # self.encoder = self.bert.encoder
        # self.encoder.load_state_dict(self.checkpoint['encoder'])
        ckpt_candidates = [
            '/home/dataset-local/chenzixu/PepSEF/01_pretraining/bert_pretraining/code/models/best_model2.pth',
            '/home/dataset-local/chenzixu/PepSEF/01_pretraining/bert_pretraining/code/models/best_model2.pth',
            'model/bert_model/best_model.pth',
        ]
        ckpt_path = None
        for p in ckpt_candidates:
            if os.path.exists(p):
                ckpt_path = p
                break
        if ckpt_path is None:
            raise FileNotFoundError(f'Cannot find BERT checkpoint, tried: {ckpt_candidates}')
        self.checkpoint = torch.load(ckpt_path, map_location=getattr(args, 'device', 'cpu'))
        self.bert.load_state_dict(self.checkpoint)
        self.encoder = self.bert.encoder   
        # self.encoder.load_state_dict(self.checkpoint)
        # self.aff = new_AFF(args, **aff_config)
        if args.pssm_hmm == 'both':
            feature_dim = 50
        elif args.pssm_hmm == 'pssm':
            feature_dim = 20
        elif args.pssm_hmm == 'hmm':
            feature_dim = 30
        elif args.pssm_hmm == 'none':
            return None
        self.caf = CrossAttentionFusion(args, feature_dim)


    def forward(self, data, attention_mask=None):
        seq, pssm = data[0], data[1]
        # attention_mask = [[1 if token != self.args.tokenizer.special_token['pad_token'] else 0 for token in s] for s in seq ]
        encoded_X = self.encoder(seq, attention_mask)
        # encoded_X = self.sigmoid(encoded_X)
        # encoded_X = F.normalize(encoded_X, dim=2)
        # print(encoded_X, pssm, sep='\n')
        # print(encoded_X.shape, pssm.shape, sep='\n')
        # exit(0)
        # aff_result = self.aff(encoded_X, pssm)
        aff_result, attn_weights = self.caf(encoded_X, pssm, attention_mask=attention_mask)
        return aff_result, attn_weights
    
class Bert_iAFF(nn.Module):
    def __init__(self, args, bert_config, aff_config) -> None:
        super(Bert_iAFF, self).__init__()
        self.bert = BERTModel(vocab_size=args.vocab_size, **bert_config)
        # self.bert.load_state_dict(torch.load('model/pretrain_bert/pytorch_model.bin'))
        # self.encoder = self.bert.encoder
        self.sigmoid = nn.Sigmoid()
        self.checkpoint = torch.load('model/bert_lstm/bert_lstm_model2.bin')
        self.encoder = self.bert.encoder
        self.encoder.load_state_dict(self.checkpoint['encoder'])
        self.aff = iAFF(args, **aff_config)

    def forward(self, data):
        seq, pssm = data[0], data[1]
        encoded_X = self.encoder(seq)
        # encoded_X = self.sigmoid(encoded_X)
        # encoded_X = F.normalize(encoded_X, dim=2)
        # print(encoded_X, pssm, sep='\n')
        # print(encoded_X.shape, pssm.shape, sep='\n')
        # exit(0)
        aff_result = self.aff(encoded_X, pssm)
        return aff_result
        
class Bert_AFF_LSTM(nn.Module):
    def __init__(self, args, bert_config, aff_config, lstm_config, multi_lstm) -> None:
        super(Bert_AFF_LSTM, self).__init__()
        self.multi_lstm = multi_lstm
        self.args = args
        self.sigmoid = nn.Sigmoid()
        self.leakyReLU = nn.LeakyReLU()
        self.bert_aff = Bert_AFF(args, bert_config, aff_config)
        self.key_extractor = KeyPositionExtractor(d_model=bert_config['num_hiddens'])
        self.LayerNorm = nn.LayerNorm(bert_config['num_hiddens'])
        # self.LSTM_ML = LSTM_ML(multi_lstm=self.multi_lstm, **lstm_config)
        # self.LSTM_ML = self.LSTM_ML.load_state_dict(torch.load('model/bert_lstm/bert_lstm_model.bin')['lstm'])
        if multi_lstm:
            self.LSTM = nn.ModuleList([nn.Sequential(Bert_AFF(args, bert_config, aff_config),
                                                    #  nn.ReLU(),
                                                    LSTM_ML(multi_lstm=self.multi_lstm, **lstm_config))for _ in range(15)])
        elif not multi_lstm:
            # self.LSTM = LSTM_ML(multi_lstm=self.multi_lstm, **lstm_config)
            # self.LSTM.load_state_dict(torch.load('model/bert_lstm/bert_lstm_model.bin')['lstm'])
            self.mlp = nn.Sequential(
                nn.Linear(256, 1024),
                nn.LayerNorm(1024), 
                nn.LeakyReLU(),
                nn.Linear(1024, 512), 
                nn.LayerNorm(512), 
                nn.LeakyReLU(),
                nn.Linear(512, 15))
            input_dim = 256
            dropout_rate = 0.3
            self.sequence_pool = nn.Sequential(
            nn.LayerNorm(128),
            nn.AdaptiveAvgPool1d(1)  # 对序列维度进行平均池化
            )
        #     self.mlp = nn.Sequential(
        #     nn.Linear(input_dim, 1024),
        #     nn.BatchNorm1d(1024),
        #     nn.GELU(),
        #     nn.Dropout(0.5),
            
        #     nn.Linear(1024, 512),
        #     nn.LayerNorm(512),
        #     nn.SiLU(),
        #     nn.Dropout(0.3),
            
        #     nn.Linear(512, 256),
        #     nn.BatchNorm1d(256),
        #     nn.GELU(),
            
        #     nn.Linear(256, 15)
        # )
            
            # print(self.LSTM)
            
        # self.checkpoint = torch.load('./model/bert_lstm/fcs_RandomOverSampler_0.8.bin')
        # for i, clf in enumerate(self.LSTM):
        #     clf.load_state_dict(self.checkpoint[f'clf{i}'])
        # self.mlp = nn.Sequential(
        #     nn.Linear(256, 15), 
        #     nn.ReLU())
            
        # for p in self.parameters():
        #     if p.dim() > 1:
        #         nn.init.xavier_uniform_(p)

    def forward(self, seq, pssm, attention_mask=None):
        data = (seq, pssm)
        if self.multi_lstm:
            # data = self.sigmoid(self.bert_aff(data))
            for lstm in self.LSTM:
                single_peptide_clf = lstm(data)  # [batch * 1]
                # single_peptide_clf = F.normalize(single_peptide_clf)
                all_peptide_clf.append(single_peptide_clf)
            all_peptide_clf = torch.cat(all_peptide_clf, dim=-1) # [batch, 15]
        elif not self.multi_lstm:
            # pass attention mask through to Bert_AFF so encoder and cross-attention can use it
            data, attn_weights = self.bert_aff(data, attention_mask)
            # attn_weights: [batch, num_heads, seq_len, seq_len]
            # key_scores, enhanced_feature = self.skey_extractor(data)
            key_scores = torch.sum(data, 2) / data.shape[2]
            data = self.leakyReLU(data)
            # data = self.LayerNorm(data)
            
            # print(data.shape)
            # LSTM分类器
            data = torch.sum(data, 1) / data.shape[1]
            # all_peptide_clf = self.LSTM(data)
            
            # 特征聚合
            # data = data.permute(0, 2, 1)  # [B, 256, L]
            # data = self.sequence_pool(data).squeeze(-1)  # [B, 256]
            all_peptide_clf = self.mlp(data)
        # all_peptide_clf = F.normalize(all_peptide_clf, dim=-1)
        # print(aff_result.shape)
        # exit(0)
        # all_peptide_clf = self.mlp(aff_result)
        return all_peptide_clf, key_scores
    
    def extract_key_subsequences(self, seq, pssm, valid_lens, top_k, window=3):
        """
        Extract key sub-sequences based on cross-attention weights while ignoring padding.
        Args:
            seq: [batch, seq_len] token ids (torch.Tensor)
            pssm: pssm features (passed to bert_aff)
            valid_lens: iterable or tensor of true lengths (no padding) for each sample
            top_k: number of key positions to return per sample
            window: radius around each chosen index to return as subsequence (0 -> single token)
        Returns:
            selected_indices: list[list[int]] per-sample selected token positions (relative indices)
            key_ids: list[list[int]] per-sample token ids at those positions (or windows flattened)
            key_subsequences: list[list[str]] decoded token strings (if tokenizer available)
        """
        with torch.no_grad():
            # forward to get attention weights from CrossAttentionFusion
            _fusion, attn_weights = self.bert_aff((seq, pssm))
            # attn_weights expected shape: [batch, num_heads, query_len, key_len]
            device = attn_weights.device

            # compute per-key importance: average over heads then take max over queries
            # attn_weights: [B, H, Lq, Lk] -> importance: [B, Lk]
            importance = attn_weights.mean(dim=1).max(dim=1).values

            # ensure valid_lens is a tensor on same device
            if not isinstance(valid_lens, torch.Tensor):
                valid_lens_t = torch.tensor(list(valid_lens), dtype=torch.long, device=device)
            else:
                valid_lens_t = valid_lens.to(device).long()

            B, L = importance.shape
            # build padding mask: True for positions that are padding (pos >= valid_len)
            arange = torch.arange(L, device=device).unsqueeze(0).expand(B, L)  # [B, L]
            pad_mask = arange >= valid_lens_t.unsqueeze(1)  # [B, L] True where padding

            # mask padded positions by setting importance to -inf for selection
            importance_masked = importance.masked_fill(pad_mask, float('-inf'))

            # For fixed-length non-overlapping windows we interpret `window` as window_size
            window_size = max(1, int(window))

            # For convolution we need finite numbers: replace -inf with 0 for aggregation
            imp_for_conv = importance_masked.clone()
            imp_for_conv[imp_for_conv == float('-inf')] = 0.0

            # compute mass for possible window start positions: conv without padding
            # imp_for_conv: [B, L] -> reshape [B, 1, L]
            if window_size > L:
                # degenerate: window larger than sequence, treat whole valid region as one window
                kernel = torch.ones(1, 1, L, device=device)
                mass = torch.nn.functional.conv1d(imp_for_conv.unsqueeze(1), kernel).squeeze(1)  # [B, 1]
                max_starts = 1
            else:
                kernel = torch.ones(1, 1, window_size, device=device)
                mass = torch.nn.functional.conv1d(imp_for_conv.unsqueeze(1), kernel, padding=0).squeeze(1)  # [B, L - window_size + 1]
                max_starts = mass.size(1)

            seq_cpu = seq.cpu() if seq.device != torch.device('cpu') else seq

            selected_indices = []
            key_ids = []
            key_subsequences = []

            for i in range(B):
                valid_len_i = int(valid_lens_t[i].item()) if i < valid_lens_t.size(0) else L

                # create mass_i over start positions
                mass_i = mass[i].cpu().numpy()

                # create boolean mask of valid start indices: start must satisfy start+window_size <= valid_len_i
                if window_size > L:
                    valid_starts = np.array([0])
                else:
                    valid_starts = np.array([s for s in range(max_starts) if (s + window_size) <= valid_len_i])

                # if no valid starts (e.g., valid_len_i==0), fallback to position 0
                if len(valid_starts) == 0:
                    selected_indices.append([0])
                    key_ids.append([int(seq_cpu[i, 0].item()) if seq_cpu.ndim == 2 else int(seq_cpu[i][0])])
                    key_subsequences.append([str(key_ids[-1][0])])
                    continue

                # set mass entries for invalid starts to -inf
                full_mass = np.full(max_starts, -np.inf, dtype=float)
                full_mass[:mass_i.shape[0]] = mass_i
                invalid_mask = np.ones_like(full_mass, dtype=bool)
                invalid_mask[valid_starts] = False
                full_mass[invalid_mask] = -np.inf

                # Build sets of special token ids and token strings if available
                spec_ids = set()
                spec_toks = set()
                if hasattr(self.args, 'tokenizer') and hasattr(self.args, 'special_token'):
                    spec = self.args.tokenizer.special_token
                    for v in spec.values():
                        if isinstance(v, int):
                            spec_ids.add(v)
                        elif isinstance(v, str):
                            spec_toks.add(v)

                # Greedily pick up to top_k non-overlapping windows, skipping any window that contains special tokens.
                picks = []
                covered = np.zeros(L, dtype=bool)

                # Continue scanning best candidate starts until we have top_k picks or exhaust starts
                while len(picks) < top_k and np.any(np.isfinite(full_mass)):
                    s = int(np.nanargmax(full_mass))
                    if not np.isfinite(full_mass[s]) or full_mass[s] == -np.inf:
                        break
                    # overlapping suppression
                    if covered[s:s + window_size].any():
                        full_mass[s] = -np.inf
                        continue

                    # evaluate candidate window [s, s+window_size)
                    start = s
                    end = min(s + window_size, valid_len_i)

                    # gather ids for this window
                    ids_window = []
                    for p in range(start, end):
                        try:
                            ids_window.append(int(seq_cpu[i, p].item()))
                        except Exception:
                            ids_window.append(int(seq_cpu[i][p]))

                    # check for special tokens by id
                    contains_special = any((_id in spec_ids) for _id in ids_window)

                    # if tokenizer provides string tokens and we haven't flagged special by id, check strings
                    if (not contains_special) and hasattr(self.args, 'tokenizer') and hasattr(self.args.tokenizer, 'convert_ids_to_tokens'):
                        try:
                            toks_window = self.args.tokenizer.convert_ids_to_tokens(ids_window)
                        except Exception:
                            toks_window = [str(x) for x in ids_window]
                        for t in toks_window:
                            # treat any special-token names or tokens containing CLS/SEP as special
                            if t in spec_toks or (isinstance(t, str) and (('CLS' in t.upper()) or ('SEP' in t.upper()))):
                                contains_special = True
                                break

                    if contains_special:
                        # skip this start and continue scanning further starts
                        full_mass[s] = -np.inf
                        continue

                    # accept this window
                    picks.append(list(range(start, end)))
                    covered[start:end] = True
                    # suppress starts that would overlap with this window
                    start_sup = max(0, start - window_size + 1)
                    end_sup = min(max_starts, end)
                    full_mass[start_sup:end_sup] = -np.inf

                # if no picks found (unlikely), fallback to first valid window
                if len(picks) == 0:
                    picks = [list(range(0, min(window_size, valid_len_i)))]

                # convert picks to ids and token strings and group per-sample
                picks_ids = []
                picks_toks = []
                for expanded in picks:
                    ids = []
                    for p in expanded:
                        try:
                            ids.append(int(seq_cpu[i, p].item()))
                        except Exception:
                            ids.append(int(seq_cpu[i][p]))
                    picks_ids.append(ids)
                    # decode tokens if tokenizer available and filter out special tokens like CLS/SEP
                    toks = []
                    if hasattr(self.args, 'tokenizer') and hasattr(self.args.tokenizer, 'convert_ids_to_tokens'):
                        try:
                            toks = self.args.tokenizer.convert_ids_to_tokens(ids)
                        except Exception:
                            toks = [str(x) for x in ids]
                    else:
                        toks = [str(x) for x in ids]
                    # filter tokens that are special (explicit names) or contain CLS/SEP markers
                    try:
                        filtered = [t for t in toks if not (t in spec_toks or (isinstance(t, str) and (('CLS' in t.upper()) or ('SEP' in t.upper()))))]
                    except Exception:
                        filtered = toks
                    # if filtering removed everything, fall back to original toks
                    if len(filtered) == 0:
                        filtered = toks
                    picks_toks.append(''.join(filtered) if isinstance(filtered, list) else filtered)

                selected_indices.append(picks)
                key_ids.append(picks_ids)
                key_subsequences.append(picks_toks)

        return selected_indices, key_ids, key_subsequences
    
class ESM2_AFF_LSTM(nn.Module):
    def __init__(self, args, esm2_config, aff_config, lstm_config, multi_lstm=False):
        super(ESM2_AFF_LSTM, self).__init__()
        self.multi_lstm = multi_lstm
        self.args = args
        self.sigmoid = nn.Sigmoid()
        self.leakyReLU = nn.LeakyReLU()
        self.esm2_aff = ESM2_AFF(args, **aff_config, esm_model_path="model/esm2")  # Use local esm2 model
        self.key_extractor = KeyPositionExtractor(d_model=esm2_config['hidden_size'])
        self.LayerNorm = nn.LayerNorm(esm2_config['hidden_size'])

        if not multi_lstm:
            # use same MLP as Bert_AFF_LSTM to keep classifier identical across models
            self.mlp = nn.Sequential(
                nn.Linear(256, 1024),
                nn.LayerNorm(1024),
                nn.LeakyReLU(),
                nn.Linear(1024, 512),
                nn.LayerNorm(512),
                nn.LeakyReLU(),
                nn.Linear(512, 15)
            )

    def forward(self, seq, pssm, attention_mask=None):
        data = (seq, pssm)
        # esm2_aff now returns (aff_result, attn_weights)
        aff_result, attn_weights = self.esm2_aff(data, attention_mask=attention_mask)
        key_scores = torch.sum(aff_result, 2) / aff_result.shape[2]
        data = self.leakyReLU(aff_result)
        data = torch.sum(data, 1) / data.shape[1]
        all_peptide_clf = self.mlp(data)
        all_peptide_clf = F.normalize(all_peptide_clf, dim=-1)
        return all_peptide_clf, key_scores

class PepESM2_AFF_LSTM(nn.Module):
    """
    Peptide-specific ESM2 wrapper that freezes the pretrained ESM2 encoder
    and exposes the same AFF+LSTM interface used by the rest of the code.

    This class accepts an already-loaded ESM model/tokenizer (optional).
    If provided, the ESM model's parameters will be frozen so only the
    downstream fusion/classifier layers are trainable.
    """
    def __init__(self, args, esm2_config, aff_config, lstm_config, multi_lstm=False, esm_model=None, tokenizer=None):
        super(PepESM2_AFF_LSTM, self).__init__()
        self.multi_lstm = multi_lstm
        self.args = args
        self.sigmoid = nn.Sigmoid()
        self.leakyReLU = nn.LeakyReLU()

        # Use the existing ESM2_AFF to build the fusion backbone; we will replace
        # its internal esm_model/tokenizer if the caller provided them.
        self.pep_esm_aff = ESM2_AFF(args, **aff_config, esm_model_path="model/esm2")

        if esm_model is not None:
            # replace the internal model and tokenizer
            self.pep_esm_aff.esm_model = esm_model
        if tokenizer is not None:
            self.pep_esm_aff.tokenizer = tokenizer

        # Freeze pretrained ESM parameters to avoid updating them during training
        try:
            for p in self.pep_esm_aff.esm_model.parameters():
                p.requires_grad = False
            self.pep_esm_aff.esm_model.eval()
        except Exception:
            pass

        # key extractor and layernorm sized from the actual loaded esm model when possible
        hidden_size = None
        try:
            cfg = getattr(self.pep_esm_aff.esm_model, 'config', None)
            hidden_size = getattr(cfg, 'hidden_size', None) or getattr(cfg, 'd_model', None)
        except Exception:
            hidden_size = None
        # fallback to provided esm2_config if model config not available
        if hidden_size is None and isinstance(esm2_config, dict):
            try:
                hidden_size = esm2_config.get('hidden_size', None)
            except Exception:
                hidden_size = None
        if hidden_size is None:
            hidden_size = 256
        # Ensure the internal projection matches the loaded ESM hidden size
        try:
            self.pep_esm_aff.esm_256 = nn.Linear(hidden_size, 256).to(getattr(args, 'device', torch.device('cpu')))
        except Exception:
            try:
                self.pep_esm_aff.esm_256 = nn.Linear(hidden_size, 256)
            except Exception:
                pass
        self.key_extractor = KeyPositionExtractor(d_model=hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size)

        # same classifier/MLP as other AFF_LSTM classes
        if not multi_lstm:
            self.mlp = nn.Sequential(
                nn.Linear(256, 1024),
                nn.LayerNorm(1024),
                nn.LeakyReLU(),
                nn.Linear(1024, 512),
                nn.LayerNorm(512),
                nn.LeakyReLU(),
                nn.Linear(512, 15)
            )

    def forward(self, seq, pssm, attention_mask=None):
        data = (seq, pssm)
        # pep_esm_aff returns (aff_result, attn_weights) like ESM2_AFF
        aff_result, attn_weights = self.pep_esm_aff(data, attention_mask=attention_mask)
        key_scores = torch.sum(aff_result, 2) / aff_result.shape[2]
        data = self.leakyReLU(aff_result)
        data = torch.sum(data, 1) / data.shape[1]
        all_peptide_clf = self.mlp(data)
        all_peptide_clf = F.normalize(all_peptide_clf, dim=-1)
        return all_peptide_clf, key_scores
class ProtBERT_AFF_LSTM(nn.Module):
    def __init__(self, args, bert_config, aff_config, lstm_config, multi_lstm=False):
        super(ProtBERT_AFF_LSTM, self).__init__()
        self.multi_lstm = multi_lstm
        self.args = args
        self.sigmoid = nn.Sigmoid()
        self.leakyReLU = nn.LeakyReLU()
        self.protbert_aff = ProtBERT_AFF(args, **aff_config)
        self.key_extractor = KeyPositionExtractor(d_model=bert_config['num_hiddens'])
        self.LayerNorm = nn.LayerNorm(bert_config['num_hiddens'])

        if not multi_lstm:
            self.mlp = nn.Sequential(
                nn.Linear(256, 1024),
                nn.LayerNorm(1024),
                nn.LeakyReLU(),
                nn.Linear(1024, 512),
                nn.LayerNorm(512),
                nn.LeakyReLU(),
                nn.Linear(512, 15)
            )

    def forward(self, seq, pssm, attention_mask=None):
        data = (seq, pssm)
        aff_result, attn_weights = self.protbert_aff(data, attention_mask=attention_mask)
        key_scores = torch.sum(aff_result, 2) / aff_result.shape[2]
        data = self.leakyReLU(aff_result)
        data = torch.sum(data, 1) / data.shape[1]
        all_peptide_clf = self.mlp(data)
        all_peptide_clf = F.normalize(all_peptide_clf, dim=-1)
        return all_peptide_clf, key_scores

class ProtT5_AFF_LSTM(nn.Module):
    def __init__(self, args, prot_t5_config, aff_config, lstm_config, multi_lstm=False):
        super(ProtT5_AFF_LSTM, self).__init__()
        self.multi_lstm = multi_lstm
        self.args = args
        self.sigmoid = nn.Sigmoid()
        self.leakyReLU = nn.LeakyReLU()
        self.prot_t5_aff = ProtT5_AFF(args, **aff_config, prot_t5_model_path="model/Prott5")  # Use ProtT5 model
        self.key_extractor = KeyPositionExtractor(d_model=prot_t5_config['d_model'])
        self.LayerNorm = nn.LayerNorm(prot_t5_config['d_model'])

        if not multi_lstm:
            self.mlp = nn.Sequential(
                nn.Linear(256, 1024),
                nn.LayerNorm(1024),
                nn.LeakyReLU(),
                nn.Linear(1024, 512),
                nn.LayerNorm(512),
                nn.LeakyReLU(),
                nn.Linear(512, 15)
            )

    def forward(self, seq, pssm, attention_mask=None):
        data = (seq, pssm)
        aff_result, attn_weights = self.prot_t5_aff(data, attention_mask=attention_mask)
        key_scores = torch.sum(aff_result, 2) / aff_result.shape[2]
        data = self.leakyReLU(aff_result)
        data = torch.sum(data, 1) / data.shape[1]
        all_peptide_clf = self.mlp(data)
        all_peptide_clf = F.normalize(all_peptide_clf, dim=-1)
        return all_peptide_clf, key_scores
    
class Bert_iAFF_LSTM(nn.Module):
    def __init__(self, args, bert_config, aff_config, lstm_config, multi_lstm) -> None:
        super(Bert_iAFF_LSTM, self).__init__()
        self.multi_lstm = multi_lstm
        self.sigmoid = nn.Sigmoid()
        self.leakyReLU = nn.LeakyReLU()
        self.bert_iaff = Bert_iAFF(args, bert_config, aff_config)
        # self.LSTM_ML = LSTM_ML(multi_lstm=self.multi_lstm, **lstm_config)
        # self.LSTM_ML = self.LSTM_ML.load_state_dict(torch.load('model/bert_lstm/bert_lstm_model.bin')['lstm'])
        if multi_lstm:
            self.LSTM = nn.ModuleList([nn.Sequential(Bert_AFF(args, bert_config, aff_config),
                                                    #  nn.ReLU(),
                                                    LSTM_ML(multi_lstm=self.multi_lstm, **lstm_config))for _ in range(15)])
        elif not multi_lstm:
            self.LSTM = LSTM_ML(multi_lstm=self.multi_lstm, **lstm_config)
            self.LSTM.load_state_dict(torch.load('model/bert_lstm/bert_lstm_model.bin')['lstm'])

    def forward(self, seq, pssm):
        data = (seq, pssm)
        if self.multi_lstm:
            # data = self.sigmoid(self.bert_aff(data))
            for lstm in self.LSTM:
                single_peptide_clf = lstm(data)  # [batch * 1]
                # single_peptide_clf = F.normalize(single_peptide_clf)
                all_peptide_clf.append(single_peptide_clf)
            all_peptide_clf = torch.cat(all_peptide_clf, dim=-1) # [batch, 15]
        elif not self.multi_lstm:
            data = self.bert_iaff(data)
            data = self.leakyReLU(data)
            all_peptide_clf = self.LSTM(data)
        all_peptide_clf = F.normalize(all_peptide_clf, dim=-1)
        # print(aff_result.shape)
        # exit(0)
        # all_peptide_clf = self.mlp(aff_result)
        return all_peptide_clf
    
class Bert_Concat_PSSM(nn.Module):
    def __init__(self, args, bert_config, lstm_config, multi_lstm) -> None:
        super(Bert_Concat_PSSM, self).__init__()
        self.bert = BERTModel(vocab_size=args.vocab_size, **bert_config)
        self.checkpoint = torch.load('model/bert_lstm/bert_lstm_model2.bin')
        self.encoder = self.bert.encoder
        self.encoder.load_state_dict(self.checkpoint['encoder'])
        self.multi_lstm = multi_lstm
        self.activation = nn.LeakyReLU()
        if multi_lstm:
            self.LSTM = nn.ModuleList([nn.Sequential(LSTM_ML(multi_lstm=self.multi_lstm, **lstm_config))for _ in range(15)])
        elif not multi_lstm:
            self.LSTM = nn.Sequential(LSTM_ML(multi_lstm=self.multi_lstm, **lstm_config)) # .load_state_dict(torch.load('model/bert_lstm/bert_lstm_model.bin')['lstm'])

    def forward(self, seq, pssm):
        encoded_X = self.activation(self.encoder(seq))
        seq_pssm = torch.cat([encoded_X, pssm], dim=2)
        seq_pssm = torch.tensor(seq_pssm, dtype=torch.float32)
        if self.multi_lstm:
            all_peptide_clf = []
            for lstm in self.LSTM:
                single_peptide_clf = lstm(seq_pssm)  # [batch * 1]
                # single_peptide_clf = F.normalize(single_peptide_clf)
                all_peptide_clf.append(single_peptide_clf)
            all_peptide_clf = torch.cat(all_peptide_clf, dim=-1) # [batch, 15]
        else:
            all_peptide_clf = self.LSTM(seq_pssm)
        all_peptide_clf = F.normalize(all_peptide_clf, dim=-1)
        return all_peptide_clf


    
class Bert_LSTM(nn.Module):
    def __init__(self,args, bert_config, lstm_config, multi_lstm) -> None:
        super(Bert_LSTM, self).__init__()
        self.multi_lstm = multi_lstm
        self.bert = BERTModel(vocab_size=args.vocab_size, **bert_config)
        self.bert.load_state_dict(torch.load('model/pretrain_bert/pytorch_model.bin'))
        self.encoder = self.bert.encoder
        self.ReLU = nn.ReLU()
        if multi_lstm:
            self.LSTM = nn.ModuleList([LSTM_ML(multi_lstm=self.multi_lstm, **lstm_config) for _ in range(15)])
        elif not multi_lstm:
            self.LSTM = LSTM_ML(multi_lstm=self.multi_lstm, **lstm_config)
        # self._reset_parameters()
        for p in self.LSTM.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, seq):
        encoded_X = self.encoder(seq)
        encoded_X = self.ReLU(encoded_X)
        if self.multi_lstm:
            all_peptide_clf = []
            for lstm in self.LSTM:
                single_peptide_clf = lstm(encoded_X)  # [batch * 1]
                all_peptide_clf.append(torch.tensor(single_peptide_clf))
            all_peptide_clf = torch.cat(all_peptide_clf, dim=1) # [batch, 15]
        elif not self.multi_lstm:
            all_peptide_clf = self.LSTM(encoded_X)
        # print(aff_result.shape)
        # exit(0)
        all_peptide_clf = F.normalize(all_peptide_clf, dim=1)
        # all_peptide_clf = self.mlp(aff_result)
        return all_peptide_clf  
    
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

class PSSM_LSTM(nn.Module):
    def __init__(self, args, lstm_config, multi_lstm) -> None:
        super(PSSM_LSTM, self).__init__()
        self.multi_lstm = multi_lstm
        if multi_lstm:
            self.LSTM = nn.ModuleList([LSTM_ML(multi_lstm=self.multi_lstm, **lstm_config) for _ in range(15)])
        elif not multi_lstm:
            self.LSTM = nn.Sequential(
                nn.Linear(20, 256),
                LSTM_ML(multi_lstm=self.multi_lstm, **lstm_config)
            )
        # self.checkpoint = torch.load('./model/bert_lstm/fcs_RandomOverSampler_0.8.bin')
        # for i, clf in enumerate(self.LSTM):
        #     clf.load_state_dict(self.checkpoint[f'clf{i}'])
        # self.mlp = nn.Sequential(
        #     nn.Linear(256, 15), 
        #     nn.ReLU())
            
        # for p in self.parameters():
        #     if p.dim() > 1:
        #         nn.init.xavier_uniform_(p)

    def forward(self, pssm):
        pssm = torch.tensor(pssm, dtype=torch.float)
        if self.multi_lstm:
            all_peptide_clf = []
            for lstm in self.LSTM:
                single_peptide_clf = lstm(pssm)  # [batch * 1]
                all_peptide_clf.append(torch.tensor(single_peptide_clf))
            all_peptide_clf = torch.cat(all_peptide_clf, dim=1) # [batch, 15]
        elif not self.multi_lstm:
            all_peptide_clf = self.LSTM(pssm)
        # print(aff_result.shape)
        # exit(0)
        all_peptide_clf = F.normalize(all_peptide_clf, dim=1)
        # all_peptide_clf = self.mlp(aff_result)
        return all_peptide_clf  
    
class Structure_LSTM(nn.Module):
    def __init__(self, args, lstm_config, multi_lstm) -> None:
        super(Structure_LSTM, self).__init__()
        self.multi_lstm = multi_lstm
        if multi_lstm:
            self.LSTM = nn.ModuleList([LSTM_ML(multi_lstm=self.multi_lstm, **lstm_config) for _ in range(15)])
        elif not multi_lstm:
            self.LSTM = nn.Sequential(
                nn.Linear(20, 256),
                LSTM_ML(multi_lstm=self.multi_lstm, **lstm_config)
            )

    def forward(self, pssm):
        pssm = torch.tensor(pssm, dtype=torch.float)
        if self.multi_lstm:
            all_peptide_clf = []
            for lstm in self.LSTM:
                single_peptide_clf = lstm(pssm)  # [batch * 1]
                all_peptide_clf.append(torch.tensor(single_peptide_clf))
            all_peptide_clf = torch.cat(all_peptide_clf, dim=1) # [batch, 15]
        elif not self.multi_lstm:
            all_peptide_clf = self.LSTM(pssm)
        # print(aff_result.shape)
        # exit(0)
        all_peptide_clf = F.normalize(all_peptide_clf, dim=1)
        # all_peptide_clf = self.mlp(aff_result)
        return all_peptide_clf  
    

class IU_LSTM(nn.Module):
    def __init__(self, args, lstm_config, multi_lstm) -> None:
        super(IU_LSTM, self).__init__()
        self.multi_lstm = multi_lstm
        if multi_lstm:
            self.LSTM = nn.ModuleList([LSTM_ML(multi_lstm=self.multi_lstm, **lstm_config) for _ in range(15)])
        elif not multi_lstm:
            self.LSTM = nn.Sequential(
                nn.Linear(3, 256),
                LSTM_ML(multi_lstm=self.multi_lstm, **lstm_config)
            )

    def forward(self, iu):
        iu = torch.tensor(iu, dtype=torch.float)
        # iu = torch.unsqueeze(iu, dim=2)
        # print(iu.shape)
        if self.multi_lstm:
            all_peptide_clf = []
            for lstm in self.LSTM:
                single_peptide_clf = lstm(iu)  # [batch * 1]
                all_peptide_clf.append(torch.tensor(single_peptide_clf))
            all_peptide_clf = torch.cat(all_peptide_clf, dim=1) # [batch, 15]
        elif not self.multi_lstm:
            all_peptide_clf = self.LSTM(iu)
        # print(aff_result.shape)
        # exit(0)
        all_peptide_clf = F.normalize(all_peptide_clf, dim=1)
        # all_peptide_clf = self.mlp(aff_result)
        return all_peptide_clf  
    
class SCR_LSTM(nn.Module):
    def __init__(self, args, lstm_config, multi_lstm) -> None:
        super(SCR_LSTM, self).__init__()
        self.multi_lstm = multi_lstm
        if multi_lstm:
            self.LSTM = nn.ModuleList([LSTM_ML(multi_lstm=self.multi_lstm, **lstm_config) for _ in range(15)])
        elif not multi_lstm:
            self.LSTM = nn.Sequential(
                nn.Linear(1, 256),
                LSTM_ML(multi_lstm=self.multi_lstm, **lstm_config)
            )

    def forward(self, scr):
        scr = torch.tensor(scr, dtype=torch.float)
        scr = torch.unsqueeze(scr, dim=2)
        if self.multi_lstm:
            all_peptide_clf = []
            for lstm in self.LSTM:
                single_peptide_clf = lstm(scr)  # [batch * 1]
                all_peptide_clf.append(torch.tensor(single_peptide_clf))
            all_peptide_clf = torch.cat(all_peptide_clf, dim=1) # [batch, 15]
        elif not self.multi_lstm:
            all_peptide_clf = self.LSTM(scr)
        all_peptide_clf = F.normalize(all_peptide_clf, dim=1)
        return all_peptide_clf  
    
class SCR_AFF(nn.Module):
    def __init__(self, args, channels=128, r=16):
        super(SCR_AFF, self).__init__()
        inter_channels = 16
        self.args = args
        self.linear = nn.Linear(1, 256).to(args.device)
        self.linear3 = nn.Linear(512, 256)
        self.tanh1 = nn.Tanh()
        self.tanh2 = nn.Tanh()

        # 局部注意力
        self.local_att = nn.Sequential(
            nn.Conv1d(in_channels=channels, out_channels=inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(channels),
        )

        # 全局注意力
        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(channels),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x, residual):
        # print(f"x shape {x.shape} pssm shape {residual.shape}")  # x shape torch.Size([16, 128, 256]) pssm shape torch.Size([16, 128, 20])
        residual = residual.float().to(self.args.device)
        residual = self.tanh1(self.linear(residual))
        x = self.tanh2(x)
        # x = self.linear2(x)
        xa = x + residual # [batch, len, d_model]
        # print(f'xa: {xa.shape}') # xa: torch.Size([16, 128, 256])
        # xa = torch.unsqueeze(xa, dim=1) # [batch, 1, len, d_model]
        # print(f"xa.shape {xa.shape}") # xa.shape torch.Size([16, 1, 128, 256])
        xl = self.local_att(xa)
        # exit(0)
        xg = self.global_att(xa)
        xlg = xl + xg
        # print(xl.shape, xg.shape, xlg.shape)
        wei = torch.squeeze(self.sigmoid(xlg))

        # wei = torch.full_like(wei, 0.9)
        # xo = x * wei + residual * (1 - wei)

        xo = torch.concat([x * wei, residual * (1 - wei)], dim=-1)
        xo = self.linear3(xo)

        return xo


class Bert_SCR_AFF(nn.Module):
    def __init__(self, args, bert_config, aff_config):
        super(Bert_SCR_AFF, self).__init__()
        self.bert = BERTModel(vocab_size=args.vocab_size, **bert_config)
        self.bert.load_state_dict(torch.load('model/pretrain_bert/pytorch_model.bin'))
        self.encoder = self.bert.encoder
        self.sigmoid = nn.Sigmoid()
        self.checkpoint = torch.load('model/bert_lstm/bert_lstm_model2.bin')
        self.encoder = self.bert.encoder
        self.encoder.load_state_dict(self.checkpoint['encoder'])
        self.aff = SCR_AFF(args, **aff_config)

    def forward(self, data):
        seq, scr = data[0], data[1]
        scr = torch.tensor(scr, dtype=torch.float)
        scr = torch.unsqueeze(scr, dim=2)
        encoded_X = self.encoder(seq)
        aff_result = self.aff(encoded_X, scr)
        return aff_result

class SCR_AFF_LSTM(nn.Module):
    def __init__(self, args, bert_config, aff_config, lstm_config, multi_lstm) -> None:
        super(SCR_AFF_LSTM, self).__init__()
        self.multi_lstm = multi_lstm
        self.sigmoid = nn.Sigmoid()
        self.leakyReLU = nn.LeakyReLU()
        self.bert_aff = Bert_SCR_AFF(args, bert_config, aff_config).to(args.device)
        # self.LSTM_ML = LSTM_ML(multi_lstm=self.multi_lstm, **lstm_config)
        # self.LSTM_ML = self.LSTM_ML.load_state_dict(torch.load('model/bert_lstm/bert_lstm_model.bin')['lstm'])
        if multi_lstm:
            self.LSTM = nn.ModuleList([nn.Sequential(SCR_AFF(args, bert_config, aff_config),
                                                    #  nn.ReLU(),
                                                    LSTM_ML(multi_lstm=self.multi_lstm, **lstm_config))for _ in range(15)])
        elif not multi_lstm:
            self.LSTM = LSTM_ML(multi_lstm=self.multi_lstm, **lstm_config)
            self.LSTM.load_state_dict(torch.load('model/bert_lstm/bert_lstm_model.bin')['lstm'])
            self.LSTM.to(args.device)
            # print(self.LSTM)

    def forward(self, seq, scr):
        data = (seq, scr)
        if self.multi_lstm:
            # data = self.sigmoid(self.bert_aff(data))
            for lstm in self.LSTM:
                single_peptide_clf = lstm(data)  # [batch * 1]
                # single_peptide_clf = F.normalize(single_peptide_clf)
                all_peptide_clf.append(single_peptide_clf)
            all_peptide_clf = torch.cat(all_peptide_clf, dim=-1) # [batch, 15]
        elif not self.multi_lstm:
            data = self.bert_aff(data)
            data = self.leakyReLU(data)
            all_peptide_clf = self.LSTM(data)
        all_peptide_clf = F.normalize(all_peptide_clf, dim=-1)
        # print(aff_result.shape)
        # exit(0)
        # all_peptide_clf = self.mlp(aff_result)
        return all_peptide_clf
