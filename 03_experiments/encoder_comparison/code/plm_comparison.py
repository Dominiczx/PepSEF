import argparse
import json
import yaml
import torch
import random
import numpy as np
import os
import warnings
import subprocess
import sys

from transformers import BertModel, AutoTokenizer, T5Tokenizer, BertTokenizer, T5EncoderModel
from transformers import EsmForMaskedLM, EsmTokenizer
from model.bert_model.bert_model import BERTModel as LocalBERTModel

from utils.bert_aff import Bert_AFF_LSTM, ESM2_AFF_LSTM, ProtT5_AFF_LSTM, ProtBERT_AFF_LSTM
from esm2_finetune.model_wrappers import PepTrainableESM2_AFF_LSTM
from utils.data_processer import PeptideDataProcessor
from utils.tokenizer import PeptideTokenizer
from utils.losses import FocalDiceLoss
from utils.validation import validate
from utils.metrics import instances_overall_metrics, label_overall_metrics

try:
    from peft import PeftModel, PeftConfig
except Exception:
    PeftModel = None
    PeftConfig = None


FINETUNED_ESM2_MODEL_DIR = '/home/dataset-local/chenzixu/PepSEF/01_pretraining/esm2_650m_pretraining/outputs/finetuned_stage2_best'
ESM2_BASE_MODEL_DIR = '/home/dataset-local/chenzixu/PepSEF/01_pretraining/esm2_650m_pretraining/models/esm2-650M'


# Helper wrapper to make HF tokenizers compatible with PeptideDataProcessor.encode_plus API
class HFTokenizerWrapper:
    def __init__(self, hf_tokenizer, max_length):
        self.t = hf_tokenizer
        self.max_length = max_length

    def encode_plus(self, seq, padding=True):
        # return (input_ids, attention_mask) similar to PeptideTokenizer
        # hf tokenizer returns dict of lists
        out = self.t(seq, add_special_tokens=True, padding='max_length' if padding else False,
                     truncation=True, max_length=self.max_length)
        input_ids = out['input_ids'] if isinstance(out['input_ids'], list) else out['input_ids'][0]
        attention_mask = out['attention_mask'] if isinstance(out['attention_mask'], list) else out['attention_mask'][0]
        return input_ids, attention_mask

    def encode(self, seq, padding=False):
        return self.encode_plus(seq, padding)[0]

    def convert_ids_to_tokens(self, ids):
        return self.t.convert_ids_to_tokens(ids)



def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_configs():
    # load necessary config files used in fusion2.py
    with open('model/bert_model/config.yaml', 'r', encoding='utf-8') as fh:
        bert_config = yaml.load(fh.read(), Loader=yaml.CLoader)
    with open('model/bert_pssm/lstm_config.yaml', 'r', encoding='utf-8') as fh:
        lstm_config = yaml.load(fh.read(), Loader=yaml.CLoader)
    with open('model/bert_pssm/aff_config.yaml', 'r', encoding='utf-8') as fh:
        aff_config = yaml.load(fh.read(), Loader=yaml.CLoader)
    with open('model/esm2/config.json', 'r', encoding='utf-8') as fh:
        esm2_config = json.load(fh)
    with open('model/Prott5/config.json', 'r', encoding='utf-8') as fh:
        prot_t5_config = json.load(fh)
    return bert_config, esm2_config, prot_t5_config, aff_config, lstm_config


def load_finetuned_esm2(model_dir, device):
    esm_model = None
    esm_tokenizer = None

    loaded = False
    if PeftModel is not None:
        try:
            base = EsmForMaskedLM.from_pretrained(ESM2_BASE_MODEL_DIR, trust_remote_code=True)
            if PeftConfig is not None:
                try:
                    peft_cfg = PeftConfig.from_pretrained(model_dir)
                    if getattr(peft_cfg, 'task_type', None) == 'CAUSAL_LM':
                        peft_cfg.task_type = 'FEATURE_EXTRACTION'
                    peft_wrapped = PeftModel.from_pretrained(base, model_dir, config=peft_cfg)
                except Exception:
                    peft_wrapped = PeftModel.from_pretrained(base, model_dir)
            else:
                peft_wrapped = PeftModel.from_pretrained(base, model_dir)
            esm_model = peft_wrapped.to(device)
            loaded = True
        except Exception:
            loaded = False

    if not loaded:
        esm_model = EsmForMaskedLM.from_pretrained(model_dir, trust_remote_code=True).to(device)

    try:
        esm_tokenizer = EsmTokenizer.from_pretrained(model_dir)
    except Exception:
        esm_tokenizer = EsmTokenizer.from_pretrained(ESM2_BASE_MODEL_DIR)

    return esm_model, esm_tokenizer


def build_model(name, args, bert_config, esm2_config, prot_t5_config, aff_config, lstm_config, device):
    name = name.lower()
    if name == 'bert':
        model = Bert_AFF_LSTM(args, bert_config, aff_config, lstm_config, multi_lstm=False)
    elif name == 'esm2':
        # ESM2_AFF expects only channels and r from aff_config; filter out unrelated keys
        aff_cfg = {k: aff_config[k] for k in ('channels', 'r') if k in aff_config}
        model = ESM2_AFF_LSTM(args, esm2_config, aff_cfg, lstm_config, multi_lstm=False)
    elif name == 'esm2_ft':
        aff_cfg = {k: aff_config[k] for k in ('channels', 'r') if k in aff_config}
        esm_model, esm_tokenizer = load_finetuned_esm2(FINETUNED_ESM2_MODEL_DIR, device)
        model = PepTrainableESM2_AFF_LSTM(
            args, esm2_config, aff_cfg, lstm_config,
            esm_model=esm_model, tokenizer=esm_tokenizer,
            multi_lstm=False, freeze_esm=False
        )
    elif name == 'prott5' or name == 'prot_t5':
        # ProtT5_AFF expects channels and r only; filter aff_config to avoid unexpected kwargs
        aff_cfg = {k: aff_config[k] for k in ('channels', 'r') if k in aff_config}
        model = ProtT5_AFF_LSTM(args, prot_t5_config, aff_cfg, lstm_config, multi_lstm=False)
    elif name == 'protbert':
        aff_cfg = {k: aff_config[k] for k in ('channels', 'r') if k in aff_config}
        model = ProtBERT_AFF_LSTM(args, bert_config, aff_cfg, lstm_config, multi_lstm=False)
    else:
        raise ValueError(f'Unsupported model name: {name}')
    return model.to(device)


class HFEncoderWithMLP(torch.nn.Module):
    """Simple adapter: HF encoder -> projection -> mlp classifier similar to Bert_AFF_LSTM"""
    def __init__(self, hf_model, hidden_size, device):
        super().__init__()
        self.hf = hf_model.to(device)
        self.proj = torch.nn.Linear(hidden_size, 256).to(device)
        self.leakyReLU = torch.nn.LeakyReLU()
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(256, 1024),
            torch.nn.LayerNorm(1024),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(1024, 512),
            torch.nn.LayerNorm(512),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(512, 15)
        ).to(device)

    def forward(self, tokens, pssm=None, attention_mask=None):
        # tokens: tensor of input ids
        if attention_mask is None:
            out = self.hf(tokens)[0]
        else:
            out = self.hf(input_ids=tokens, attention_mask=attention_mask)[0]
        # out: [B, L, H]
        x = self.proj(out)  # [B, L, 256]
        x = self.leakyReLU(x)
        x = torch.sum(x, 1) / x.shape[1]
        logits = self.mlp(x)
        # create a dummy key_scores similar to other wrappers
        key_scores = torch.sum(out, 2) / out.shape[2]
        return logits, key_scores


class BertFTNoFusion(torch.nn.Module):
    """No-fusion classifier head for bert_ft when pssm_hmm=none."""
    def __init__(self, bert_encoder, hidden_size, device):
        super().__init__()
        self.bert = bert_encoder.to(device)
        self.proj = torch.nn.Linear(hidden_size, 256).to(device)
        self.act = torch.nn.LeakyReLU()
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(256, 1024),
            torch.nn.LayerNorm(1024),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(1024, 512),
            torch.nn.LayerNorm(512),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(512, 15)
        ).to(device)

    def forward(self, tokens, pssm=None, attention_mask=None):
        if attention_mask is None:
            attention_mask = (tokens != 0).long()
        out, _ = self.bert(tokens, attention_mask)
        x = self.act(self.proj(out))
        mask = attention_mask.float().unsqueeze(-1)
        denom = torch.clamp(mask.sum(dim=1), min=1.0)
        x = (x * mask).sum(dim=1) / denom
        logits = self.mlp(x)
        key_scores = torch.sum(out, 2) / out.shape[2]
        return logits, key_scores


class EncoderNoFusion(torch.nn.Module):
    """Generic no-fusion classifier for HF-style encoders."""
    def __init__(self, encoder, hidden_size, device):
        super().__init__()
        self.encoder = encoder.to(device)
        self.proj = torch.nn.Linear(hidden_size, 256).to(device)
        self.act = torch.nn.LeakyReLU()
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(256, 1024),
            torch.nn.LayerNorm(1024),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(1024, 512),
            torch.nn.LayerNorm(512),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(512, 15)
        ).to(device)

    def _extract_hidden(self, out):
        if hasattr(out, 'last_hidden_state') and out.last_hidden_state is not None:
            return out.last_hidden_state
        if hasattr(out, 'hidden_states') and out.hidden_states is not None and len(out.hidden_states) > 0:
            return out.hidden_states[-1]
        if isinstance(out, tuple) and len(out) > 0:
            return out[0]
        return out

    def forward(self, tokens, pssm=None, attention_mask=None):
        if attention_mask is None:
            attention_mask = (tokens != 0).long()
        out = self.encoder(input_ids=tokens, attention_mask=attention_mask, output_hidden_states=True, return_dict=True)
        hidden = self._extract_hidden(out)
        x = self.act(self.proj(hidden))
        mask = attention_mask.float().unsqueeze(-1)
        denom = torch.clamp(mask.sum(dim=1), min=1.0)
        x = (x * mask).sum(dim=1) / denom
        logits = self.mlp(x)
        key_scores = torch.sum(hidden, 2) / hidden.shape[2]
        return logits, key_scores


class ProtT5NoFusion(torch.nn.Module):
    """No-fusion classifier for ProtT5 that tokenizes raw sequences internally."""
    def __init__(self, model_path, device):
        super().__init__()
        self.encoder = T5EncoderModel.from_pretrained(model_path).to(device)
        self.tokenizer = T5Tokenizer.from_pretrained(model_path)
        hidden_size = int(getattr(self.encoder.config, 'd_model', 1024))
        self.proj = torch.nn.Linear(hidden_size, 256).to(device)
        self.act = torch.nn.LeakyReLU()
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(256, 1024),
            torch.nn.LayerNorm(1024),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(1024, 512),
            torch.nn.LayerNorm(512),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(512, 15)
        ).to(device)

    def forward(self, seqs, pssm=None, attention_mask=None):
        seqs = [" ".join(list(s)) for s in seqs]
        seqs = ["<AA2fold> " + s if s.isupper() else "<fold2AA> " + s for s in seqs]
        tokenized = self.tokenizer.batch_encode_plus(seqs, add_special_tokens=True, padding=True, return_tensors='pt')
        tokenized = {k: v.to(next(self.parameters()).device) for k, v in tokenized.items()}
        out = self.encoder(**tokenized)
        hidden = out.last_hidden_state if hasattr(out, 'last_hidden_state') else out[0]
        x = self.act(self.proj(hidden))
        mask = tokenized['attention_mask'].float().unsqueeze(-1)
        denom = torch.clamp(mask.sum(dim=1), min=1.0)
        x = (x * mask).sum(dim=1) / denom
        logits = self.mlp(x)
        key_scores = torch.sum(hidden, 2) / hidden.shape[2]
        return logits, key_scores


class ProtBERTNoFusion(torch.nn.Module):
    """No-fusion classifier for ProtBERT with proper AA tokenization."""
    def __init__(self, model_path, device):
        super().__init__()
        self.encoder = BertModel.from_pretrained(model_path).to(device)
        self.tokenizer = BertTokenizer.from_pretrained(model_path, do_lower_case=False)
        hidden_size = int(getattr(self.encoder.config, 'hidden_size', 1024))
        self.proj = torch.nn.Linear(hidden_size, 256).to(device)
        self.drop = torch.nn.Dropout(0.2)
        self.act = torch.nn.LeakyReLU()
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(256, 1024),
            torch.nn.LayerNorm(1024),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(1024, 512),
            torch.nn.LayerNorm(512),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(512, 15)
        ).to(device)

    def forward(self, seqs, pssm=None, attention_mask=None):
        dev = next(self.parameters()).device
        if isinstance(seqs, torch.Tensor):
            tokenized = {
                'input_ids': seqs.to(dev),
                'attention_mask': attention_mask.to(dev) if isinstance(attention_mask, torch.Tensor) else (seqs != 0).long().to(dev)
            }
        else:
            seqs = [" ".join(list(s)) for s in seqs]
            tokenized = self.tokenizer.batch_encode_plus(seqs, add_special_tokens=True, padding=True, truncation=True, return_tensors='pt')
            tokenized = {k: v.to(dev) for k, v in tokenized.items()}
        out = self.encoder(**tokenized)
        hidden = out.last_hidden_state if hasattr(out, 'last_hidden_state') else out[0]
        x = self.act(self.proj(hidden))
        x = self.drop(x)
        mask = tokenized['attention_mask'].float().unsqueeze(-1)
        denom = torch.clamp(mask.sum(dim=1), min=1.0)
        x = (x * mask).sum(dim=1) / denom
        logits = self.mlp(x)
        key_scores = torch.sum(hidden, 2) / hidden.shape[2]
        return logits, key_scores


class ESMNoFusion(torch.nn.Module):
    """No-fusion classifier for ESM/ESM-ft from raw sequences."""
    def __init__(self, encoder, tokenizer, hidden_size, device):
        super().__init__()
        self.encoder = encoder.to(device)
        self.tokenizer = tokenizer
        self.proj = torch.nn.Linear(hidden_size, 256).to(device)
        self.drop = torch.nn.Dropout(0.2)
        self.act = torch.nn.LeakyReLU()
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(256, 1024),
            torch.nn.LayerNorm(1024),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(1024, 512),
            torch.nn.LayerNorm(512),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(512, 15)
        ).to(device)

    def _extract_hidden(self, out):
        if hasattr(out, 'hidden_states') and out.hidden_states is not None and len(out.hidden_states) > 0:
            return out.hidden_states[-1]
        if hasattr(out, 'last_hidden_state') and out.last_hidden_state is not None:
            return out.last_hidden_state
        if isinstance(out, tuple) and len(out) > 0:
            return out[0]
        return out

    def forward(self, seqs, pssm=None, attention_mask=None):
        dev = next(self.parameters()).device
        if isinstance(seqs, torch.Tensor):
            tokenized = {
                'input_ids': seqs.to(dev),
                'attention_mask': attention_mask.to(dev) if isinstance(attention_mask, torch.Tensor) else (seqs != 0).long().to(dev)
            }
        else:
            tokenized = self.tokenizer(seqs, return_tensors='pt', padding=True, truncation=True)
            tokenized = {k: v.to(dev) for k, v in tokenized.items()}
        out = self.encoder(**tokenized, output_hidden_states=True, return_dict=True)
        hidden = self._extract_hidden(out)
        x = self.act(self.proj(hidden))
        x = self.drop(x)
        mask = tokenized['attention_mask'].float().unsqueeze(-1)
        denom = torch.clamp(mask.sum(dim=1), min=1.0)
        x = (x * mask).sum(dim=1) / denom
        logits = self.mlp(x)
        key_scores = torch.sum(hidden, 2) / hidden.shape[2]
        return logits, key_scores


def _forward_by_model_name(model_name, model, seq, token, feature, attention_mask, device, pssm_hmm='none'):
    if model_name in ('my', 'bert', 'bert_ft') and pssm_hmm == 'none':
        tokens = token.to(device)
        attention_masks = attention_mask.to(device)
        outputs, key_scores = model(tokens, None, attention_masks)
    elif model_name == 'protbert' and pssm_hmm == 'none':
        seqs = list(seq)
        outputs, key_scores = model(seqs, None, None)
    elif model_name == 'esm2' and pssm_hmm == 'none':
        seqs = list(seq)
        outputs, key_scores = model(seqs, None, None)
    elif model_name == 'esm2_ft' and pssm_hmm == 'none':
        seqs = list(seq)
        outputs, key_scores = model(seqs, None, None)
    elif model_name in ('my', 'bert', 'bert_ft'):
        tokens = token.to(device)
        features = feature.to(device)
        attention_masks = attention_mask.to(device)
        outputs, key_scores = model(tokens, features, attention_masks)
    elif model_name == 'esm2':
        seqs = list(seq)
        features = feature.to(device)
        outputs, key_scores = model(seqs, features)
    elif model_name == 'esm2_ft':
        seqs = list(seq)
        features = feature.to(device)
        outputs, key_scores = model(seqs, features)
    elif model_name in ('prott5', 'prot_t5'):
        seqs = list(seq)
        features = feature.to(device)
        attention_masks = attention_mask.to(device)
        outputs, key_scores = model(seqs, features, attention_masks)
    elif model_name == 'protbert':
        tokens = token.to(device)
        attention_masks = attention_mask.to(device)
        features = feature.to(device)
        outputs, key_scores = model(tokens, features, attention_masks)
    else:
        raise ValueError('Unknown model type')
    return outputs, key_scores


def _evaluate_full_metrics(model_name, model, dataloader, device, pssm_hmm='none', threshold=0.5):
    model.eval()
    all_probs, all_true = [], []
    with torch.no_grad():
        for seq, token, feature, label, valid_len, attention_mask in dataloader:
            outputs, _ = _forward_by_model_name(model_name, model, seq, token, feature, attention_mask, device, pssm_hmm=pssm_hmm)
            probs = torch.sigmoid(outputs).detach().cpu().numpy()
            y_true = label.detach().cpu().numpy()
            all_probs.append(probs)
            all_true.append(y_true)

    if len(all_probs) == 0:
        return {
            'instance_acc': 0.0,
            'instance_precision': 0.0,
            'instance_recall': 0.0,
            'instance_f1': 0.0,
            'hamming_loss': 0.0,
            'exact_match': 0.0,
            'macro_f1': 0.0,
            'micro_f1': 0.0,
            'macro_auc': np.nan,
            'micro_auc': np.nan,
            'macro_aupr': np.nan,
            'micro_aupr': np.nan,
            'macro_aupoc': np.nan,
            'micro_aupoc': np.nan,
            'macro_mcc': np.nan,
            'macro_bacc': np.nan,
            'macro_precision': 0.0,
            'micro_precision': 0.0,
            'macro_recall': 0.0,
            'micro_recall': 0.0,
            'macro_acc': 0.0,
        }

    probs = np.concatenate(all_probs, axis=0)
    y_true = np.concatenate(all_true, axis=0)
    pred = (probs >= float(threshold)).astype(np.float32)

    ins = instances_overall_metrics(pred, y_true)
    lbl = label_overall_metrics(probs, y_true)
    p = float(ins.get('Precision', 0.0))
    r = float(ins.get('Recall', 0.0))
    f1_ins = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0

    return {
        'instance_acc': float(ins.get('Accuracy', 0.0)),
        'instance_precision': p,
        'instance_recall': r,
        'instance_f1': float(f1_ins),
        'hamming_loss': float(ins.get('HLoss', 0.0)),
        'exact_match': float(ins.get('Absolute true', 0.0)),
        'macro_f1': float(lbl['F1'][0]),
        'micro_f1': float(lbl['F1'][1]),
        'macro_auc': float(lbl['AUC'][0]),
        'micro_auc': float(lbl['AUC'][1]),
        'macro_aupr': float(lbl['AUPR'][0]),
        'micro_aupr': float(lbl['AUPR'][1]),
        # AUPOC is reported as the same curve-area family as AUPR for compatibility.
        'macro_aupoc': float(lbl['AUPR'][0]),
        'micro_aupoc': float(lbl['AUPR'][1]),
        'macro_mcc': float(lbl['MCC'][0]),
        'macro_bacc': float(lbl['BACC'][0]),
        'macro_precision': float(lbl['Precision'][0]),
        'micro_precision': float(lbl['Precision'][1]),
        'macro_recall': float(lbl['Recall'][0]),
        'micro_recall': float(lbl['Recall'][1]),
        'macro_acc': float(lbl['Accuracy'][0]),
    }


def _fmt_metric(v):
    if isinstance(v, (float, np.floating)):
        if np.isnan(v):
            return 'nan'
        return f'{float(v):.4f}'
    return str(v)


def train_one(model_name, model, args, train_dataloader, val_dataloader, test_dataloader):
    device = args.device
    criterion = FocalDiceLoss()
    bce_aux = torch.nn.BCEWithLogitsLoss()
    if model_name in ('esm2_ft', 'esm2', 'protbert', 'prott5', 'prot_t5'):
        backbone_params = []
        head_params = []
        for n, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if any(k in n.lower() for k in ('encoder', 'esm_model', 'backbone', 'hf', 'bert', 't5')):
                backbone_params.append(p)
            else:
                head_params.append(p)
        if not head_params:
            head_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW([
            {'params': backbone_params, 'lr': min(args.learning_rate, 5e-6), 'weight_decay': 0.01},
            {'params': head_params, 'lr': max(args.learning_rate * 6.0, 1.5e-5), 'weight_decay': 0.01},
        ])
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    # Scheduler: reduce LR on plateau (monitor val accuracy)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)

    best = {
        'epoch': -1,
        'score': -1e9,
        'val_acc': -1.0,
        'test_acc': -1.0,
        'val_metrics': {},
        'test_metrics': {}
    }
    for e in range(args.epochs):
        model.train()
        total_acc = 0
        for seq, token, feature, label, valid_len, attention_mask in train_dataloader:
            outputs, key_scores = _forward_by_model_name(
                model_name, model, seq, token, feature, attention_mask, device, pssm_hmm=args.pssm_hmm
            )

            if model_name in ('esm2_ft', 'esm2', 'protbert', 'prott5', 'prot_t5'):
                loss = criterion(outputs.float(), label.to(device).float()) + 0.3 * bce_aux(outputs.float(), label.to(device).float())
            else:
                loss = criterion(outputs.float(), label.to(device).float())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            probs = torch.sigmoid(outputs)
            pred = (probs > 0.5).float().cpu().detach().numpy()
            y_true = label.cpu().detach().numpy()
            df = instances_overall_metrics(pred, y_true)
            total_acc += df.get('Accuracy', 0)

        mean_acc = total_acc / (len(train_dataloader))
        print(f'[{model_name}] epoch {e+1}/{args.epochs} train instance-acc: {mean_acc:.4f}')

        model.eval()
        with torch.no_grad():
            val_acc, val_pre, val_rec, val_f1 = validate(args, val_dataloader, model, phase='val', save_csv=False)
            test_acc, test_pre, test_rec, test_f1 = validate(args, test_dataloader, model, phase='test', save_csv=False)
            val_metrics = _evaluate_full_metrics(model_name, model, val_dataloader, device, pssm_hmm=args.pssm_hmm)
            test_metrics = _evaluate_full_metrics(model_name, model, test_dataloader, device, pssm_hmm=args.pssm_hmm)
            print(
                f"[{model_name}] val: inst_acc={_fmt_metric(val_metrics['instance_acc'])}, "
                f"macro_f1={_fmt_metric(val_metrics['macro_f1'])}, micro_f1={_fmt_metric(val_metrics['micro_f1'])}, "
                f"macro_auc={_fmt_metric(val_metrics['macro_auc'])}, macro_aupr={_fmt_metric(val_metrics['macro_aupr'])}, "
                f"macro_aupoc={_fmt_metric(val_metrics['macro_aupoc'])}, hloss={_fmt_metric(val_metrics['hamming_loss'])} | "
                f"test: inst_acc={_fmt_metric(test_metrics['instance_acc'])}, macro_f1={_fmt_metric(test_metrics['macro_f1'])}, "
                f"micro_f1={_fmt_metric(test_metrics['micro_f1'])}, macro_auc={_fmt_metric(test_metrics['macro_auc'])}, "
                f"macro_aupoc={_fmt_metric(test_metrics['macro_aupoc'])}, hloss={_fmt_metric(test_metrics['hamming_loss'])}"
            )
            # Step scheduler using validation accuracy
            try:
                scheduler.step(val_acc)
            except Exception:
                pass

            # select best checkpoint by validation macro F1, then validation instance_acc
            cur_score = float(val_metrics.get('macro_f1', 0.0))
            tie_break = float(val_metrics.get('instance_acc', 0.0))
            best_tie_break = float(best.get('val_metrics', {}).get('instance_acc', -1.0)) if best.get('val_metrics') else -1.0
            improved = (cur_score > best['score'] + 1e-8) or (abs(cur_score - best['score']) <= 1e-8 and tie_break >= best_tie_break)
            if improved:
                best['epoch'] = e + 1
                best['score'] = cur_score
                best['val_acc'] = float(val_acc)
                best['test_acc'] = float(test_acc)
                best['val_metrics'] = val_metrics
                best['test_metrics'] = test_metrics
                # Keep rerun artifacts in the experiment result directory instead
                # of writing beside the source file.
                out_dir = getattr(args, 'out_dir', None)
                if not out_dir:
                    base_dir = os.path.dirname(os.path.abspath(__file__))
                    out_dir = os.path.join(base_dir, '..', 'results', 'cmp_output')
                os.makedirs(out_dir, exist_ok=True)
                save_path = os.path.join(out_dir, f'{model_name}_best.bin')
                try:
                    torch.save(model.state_dict(), save_path)
                    print(f'saved best {model_name} to {save_path}')
                except Exception as ex:
                    print('failed to save model:', ex)

    print(
        f"[{model_name}] best@epoch {best['epoch']}: "
        f"val(inst_acc={_fmt_metric(best.get('val_metrics', {}).get('instance_acc', np.nan))}, "
        f"macro_f1={_fmt_metric(best.get('val_metrics', {}).get('macro_f1', np.nan))}, "
        f"micro_f1={_fmt_metric(best.get('val_metrics', {}).get('micro_f1', np.nan))}, "
        f"macro_auc={_fmt_metric(best.get('val_metrics', {}).get('macro_auc', np.nan))}, "
        f"macro_aupoc={_fmt_metric(best.get('val_metrics', {}).get('macro_aupoc', np.nan))}, "
        f"hloss={_fmt_metric(best.get('val_metrics', {}).get('hamming_loss', np.nan))}) | "
        f"test(inst_acc={_fmt_metric(best.get('test_metrics', {}).get('instance_acc', np.nan))}, "
        f"macro_f1={_fmt_metric(best.get('test_metrics', {}).get('macro_f1', np.nan))}, "
        f"micro_f1={_fmt_metric(best.get('test_metrics', {}).get('micro_f1', np.nan))}, "
        f"macro_auc={_fmt_metric(best.get('test_metrics', {}).get('macro_auc', np.nan))}, "
        f"macro_aupoc={_fmt_metric(best.get('test_metrics', {}).get('macro_aupoc', np.nan))}, "
        f"hloss={_fmt_metric(best.get('test_metrics', {}).get('hamming_loss', np.nan))})"
    )

    # after training finishes, write a per-model result JSON file so the
    # parallel launcher (parent) can aggregate results without races/duplicates
    out_dir = getattr(args, 'out_dir', None)
    if not out_dir:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        out_dir = os.path.join(base_dir, '..', 'results', 'cmp_output')
    os.makedirs(out_dir, exist_ok=True)
    result_path = os.path.join(out_dir, f'{model_name}_result.json')
    try:
        with open(result_path, 'w') as fh:
            json.dump({
                'model': model_name,
                'best_epoch': best['epoch'],
                'selection_metric': 'val_macro_f1',
                'val_acc': best['val_acc'],
                'test_acc': best['test_acc'],
                'val_metrics': best.get('val_metrics', {}),
                'test_metrics': best.get('test_metrics', {}),
            }, fh)
    except Exception as ex:
        print('failed to write per-model result json:', ex)
    return best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=5e-5)
    parser.add_argument('--max_length', type=int, default=128)
    parser.add_argument('--pssm_hmm', type=str, default='None')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--mode', type=str, default='all', choices=['all', 'parallel', 'single'],
                        help='Run mode: all (sequential), parallel (spawn per-model processes), single (run one model)')
    parser.add_argument('--model_name', type=str, default=None, help='When --mode single, the model to run')
    parser.add_argument('--gpu_id', type=int, default=None, help='GPU id to use when running single model')
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--out_dir', type=str, default=os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results', 'cmp_output'),
                        help='Directory for encoder-comparison models, json metrics, and aggregate CSV')
    args = parser.parse_args()
    # reduce noisy warnings (sklearn ranking warnings etc.)
    warnings.filterwarnings('ignore')

    set_seed(args.seed)
    # Normalize pssm_hmm so values like None/none/null are handled consistently.
    raw_mode = str(args.pssm_hmm).strip().lower() if args.pssm_hmm is not None else 'none'
    if raw_mode in ('none', 'null', 'false', 'no', ''):
        args.pssm_hmm = 'none'
    elif raw_mode in ('pssm', 'hmm', 'both'):
        args.pssm_hmm = raw_mode
    else:
        print(f"[WARN] Unknown pssm_hmm={args.pssm_hmm}, fallback to none")
        args.pssm_hmm = 'none'

    # If single mode with explicit gpu_id, bind to that device.
    # Note: when children are launched from the parallel launcher we set
    # CUDA_VISIBLE_DEVICES in the parent's environment so the child sees
    # only a single visible device (which will be index 0 inside the child).
    # Therefore here we set device to 'cuda:0' so the moved model lands on the
    # visible GPU. If the user runs single mode directly without parent,
    # set CUDA_VISIBLE_DEVICES accordingly.
    if args.mode == 'single' and args.gpu_id is not None:
        if 'CUDA_VISIBLE_DEVICES' not in os.environ:
            # standalone single mode: restrict visible devices to the requested id
            os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
        # inside the process the assigned GPU is visible as cuda:0
        args.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    else:
        args.device = torch.device(args.device)
    args.batch_size = args.batch_size
    args.max_length = args.max_length
    args.pssm_hmm = args.pssm_hmm
    # tokenizer files used by PeptideTokenizer
    args.vocab_path = 'utils/uniprot_1kmer_vocab.txt'
    args.special_token_path = 'utils/special_tokens.json'
    args.save = True

    bert_config, esm2_config, prot_t5_config, aff_config, lstm_config = load_configs()

    model_names = ['bert_ft', 'protbert', 'prott5', 'esm2', 'esm2_ft']
    results = {}

    # Parallel mode: spawn per-model subprocesses and wait for them to finish
    if args.mode == 'parallel':
        print('Launching parallel jobs for models')
        procs = []
        gpu_map = { 'bert_ft': 0, 'protbert': 1, 'prott5': 2, 'esm2': 3, 'esm2_ft': 4 }
        num_gpu = max(torch.cuda.device_count(), 1)
        for name in model_names:
            gpu = gpu_map.get(name, 0) % num_gpu
            cmd = [sys.executable, os.path.abspath(__file__),
                   '--mode', 'single', '--model_name', name, '--gpu_id', str(gpu),
                     '--epochs', str(args.epochs), '--batch_size', str(args.batch_size), '--learning_rate', str(args.learning_rate), '--max_length', str(args.max_length), '--pssm_hmm', str(args.pssm_hmm), '--out_dir', str(args.out_dir)]
            print('Starting:', ' '.join(cmd))
            # ensure each child process sees only its assigned GPU at process start
            child_env = os.environ.copy()
            child_env['CUDA_VISIBLE_DEVICES'] = str(gpu)
            # ensure Python stdout/stderr are unbuffered for real-time logs
            child_env['PYTHONUNBUFFERED'] = '1'
            p = subprocess.Popen(cmd, env=child_env)
            procs.append((name, p))
        # wait and collect exit codes
        for name, p in procs:
            ret = p.wait()
            print(f'Process for {name} exited with code {ret}')

        # aggregate per-model result json files into a single CSV (overwrite)
        base_dir = os.path.dirname(os.path.abspath(__file__))
        out_dir = args.out_dir
        os.makedirs(out_dir, exist_ok=True)
        csv_path = os.path.join(out_dir, 'results.csv')
        rows_map = {}
        for name in model_names:
            rpath = os.path.join(out_dir, f'{name}_result.json')
            if os.path.exists(rpath):
                try:
                    with open(rpath, 'r') as fh:
                        data = json.load(fh)
                        rows_map[data.get('model', name)] = data
                except Exception as ex:
                    print('failed to read', rpath, ex)
            else:
                print('missing result file for', name, 'expected at', rpath)

        # write aggregate CSV atomically and deduplicate by model (keep one row per model)
        tmp_path = csv_path + '.tmp'
        try:
            with open(tmp_path, 'w') as fh:
                fh.write(
                    'model,best_epoch,val_inst_acc,val_macro_f1,val_micro_f1,val_macro_auc,val_macro_aupr,val_macro_aupoc,val_hamming_loss,val_exact_match,'
                    'test_inst_acc,test_macro_f1,test_micro_f1,test_macro_auc,test_macro_aupr,test_macro_aupoc,test_hamming_loss,test_exact_match\n'
                )
                for name in model_names:
                    if name in rows_map:
                        d = rows_map[name]
                        vm = d.get('val_metrics', {}) if isinstance(d, dict) else {}
                        tm = d.get('test_metrics', {}) if isinstance(d, dict) else {}
                        row = [
                            name,
                            d.get('best_epoch', -1),
                            vm.get('instance_acc', np.nan),
                            vm.get('macro_f1', np.nan),
                            vm.get('micro_f1', np.nan),
                            vm.get('macro_auc', np.nan),
                            vm.get('macro_aupr', np.nan),
                            vm.get('macro_aupoc', np.nan),
                            vm.get('hamming_loss', np.nan),
                            vm.get('exact_match', np.nan),
                            tm.get('instance_acc', np.nan),
                            tm.get('macro_f1', np.nan),
                            tm.get('micro_f1', np.nan),
                            tm.get('macro_auc', np.nan),
                            tm.get('macro_aupr', np.nan),
                            tm.get('macro_aupoc', np.nan),
                            tm.get('hamming_loss', np.nan),
                            tm.get('exact_match', np.nan),
                        ]
                        fh.write(','.join([str(x) for x in row]) + '\n')
            os.replace(tmp_path, csv_path)
        except Exception as ex:
            print('failed to write aggregated results csv:', ex)
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass

        # print best-summary across models for quick comparison in output logs
        valid = [d for d in rows_map.values() if isinstance(d, dict)]
        if len(valid) > 0:
            best_val_macro_f1 = max(valid, key=lambda x: float(x.get('val_metrics', {}).get('macro_f1', -1e9)))
            best_test_macro_f1 = max(valid, key=lambda x: float(x.get('test_metrics', {}).get('macro_f1', -1e9)))
            best_test_inst_acc = max(valid, key=lambda x: float(x.get('test_metrics', {}).get('instance_acc', -1e9)))
            print('\nParallel summary:')
            print(f"best val macro_f1: {best_val_macro_f1.get('model')} @ {_fmt_metric(best_val_macro_f1.get('val_metrics', {}).get('macro_f1', np.nan))}")
            print(f"best test macro_f1: {best_test_macro_f1.get('model')} @ {_fmt_metric(best_test_macro_f1.get('test_metrics', {}).get('macro_f1', np.nan))}")
            print(f"best test inst_acc: {best_test_inst_acc.get('model')} @ {_fmt_metric(best_test_inst_acc.get('test_metrics', {}).get('instance_acc', np.nan))}")

        # after parallel runs and aggregation, exit
        return

    # Single-model mode: only run one model specified by --model_name on --gpu_id
    if args.mode == 'single':
        if not args.model_name:
            raise ValueError('When --mode single you must provide --model_name')
        model_names = [args.model_name]

    for name in model_names:
        print('\n' + '='*60)
        print('Preparing tokenizer and dataloader for:', name)
        # set tokenizer per-model
        if name in ('my', 'my_model', 'bert', 'bert_ft'):
            args.tokenizer = PeptideTokenizer(args)
        elif name == 'protbert':
            hf_tok = BertTokenizer.from_pretrained('model/ProtBert', do_lower_case=False)
            args.tokenizer = HFTokenizerWrapper(hf_tok, args.max_length)
        elif name in ('prott5', 'prot_t5'):
            hf_tok = T5Tokenizer.from_pretrained('model/Prott5')
            args.tokenizer = HFTokenizerWrapper(hf_tok, args.max_length)
        elif name == 'esm2':
            hf_tok = AutoTokenizer.from_pretrained('model/esm2')
            args.tokenizer = HFTokenizerWrapper(hf_tok, args.max_length)
        elif name == 'esm2_ft':
            try:
                hf_tok = EsmTokenizer.from_pretrained(FINETUNED_ESM2_MODEL_DIR)
            except Exception:
                hf_tok = EsmTokenizer.from_pretrained(ESM2_BASE_MODEL_DIR)
            args.tokenizer = HFTokenizerWrapper(hf_tok, args.max_length)
        else:
            raise ValueError('Unknown model name for tokenizer setup')

        # build dataloaders with this tokenizer
        pdp = PeptideDataProcessor('data/ft_data/', args, aug=False)
        train_dataset, train_dataloader, val_dataset, val_dataloader, test_dataset, test_dataloader = pdp.form_ml_dataloader()
        print('datasets sizes', len(train_dataset), len(val_dataset), len(test_dataset))

        # ensure vocab_size is set when using PeptideTokenizer
        if hasattr(args, 'tokenizer') and hasattr(args.tokenizer, 'vocab_size'):
            args.vocab_size = args.tokenizer.vocab_size

        print('Building model:', name)
        try:
            if name in ('my', 'my_model', 'bert', 'bert_ft'):
                if name == 'bert_ft' and args.pssm_hmm == 'none':
                    local_bert = LocalBERTModel(
                        vocab_size=int(getattr(args, 'vocab_size', 30)),
                        num_hiddens=int(bert_config.get('num_hiddens', 256)),
                        num_heads=int(bert_config.get('num_heads', 8)),
                        num_layers=int(bert_config.get('num_layers', 12)),
                        ffn_num_hiddens=int(bert_config.get('ffn_num_hiddens', 1024)),
                        max_length=int(args.max_length),
                        dropout=float(bert_config.get('dropout', 0.2)),
                    )
                    ckpt_path = 'model/bert_model/best_model.pth'
                    if os.path.exists(ckpt_path):
                        state = torch.load(ckpt_path, map_location='cpu')
                        if isinstance(state, dict) and 'model_state_dict' in state:
                            state = state['model_state_dict']
                        local_bert.load_state_dict(state, strict=False)
                    model = BertFTNoFusion(local_bert, int(bert_config.get('num_hiddens', 256)), args.device)
                else:
                    model = build_model('bert', args, bert_config, esm2_config, prot_t5_config, aff_config, lstm_config, args.device)
            elif name == 'protbert':
                if args.pssm_hmm == 'none':
                    model = ProtBERTNoFusion('model/ProtBert', args.device)
                else:
                    model = build_model('protbert', args, bert_config, esm2_config, prot_t5_config, aff_config, lstm_config, args.device)
            elif name in ('prott5', 'prot_t5'):
                if args.pssm_hmm == 'none':
                    model = ProtT5NoFusion('model/Prott5', args.device)
                else:
                    model = build_model('prott5', args, bert_config, esm2_config, prot_t5_config, aff_config, lstm_config, args.device)
            elif name == 'esm2':
                if args.pssm_hmm == 'none':
                    backbone = EsmForMaskedLM.from_pretrained('model/esm2', trust_remote_code=True)
                    tok = AutoTokenizer.from_pretrained('model/esm2')
                    hidden_size = int(getattr(backbone.config, 'hidden_size', 1280))
                    model = ESMNoFusion(backbone, tok, hidden_size, args.device)
                else:
                    model = build_model('esm2', args, bert_config, esm2_config, prot_t5_config, aff_config, lstm_config, args.device)
            elif name == 'esm2_ft':
                if args.pssm_hmm == 'none':
                    esm_model, esm_tok = load_finetuned_esm2(FINETUNED_ESM2_MODEL_DIR, args.device)
                    hidden_size = None
                    try:
                        cfg = getattr(esm_model, 'config', None)
                        hidden_size = int(getattr(cfg, 'hidden_size', None) or getattr(cfg, 'd_model', None))
                    except Exception:
                        hidden_size = None
                    if hidden_size is None:
                        hidden_size = 1280
                    model = ESMNoFusion(esm_model, esm_tok, hidden_size, args.device)
                else:
                    model = build_model('esm2_ft', args, bert_config, esm2_config, prot_t5_config, aff_config, lstm_config, args.device)
            else:
                raise ValueError('Unknown model name for building')

            print('Training model:', name)
            best = train_one(name, model, args, train_dataloader, val_dataloader, test_dataloader)
            results[name] = best
        except Exception as ex:
            print(f'[WARN] Skip model {name} due to runtime error under pssm_hmm={args.pssm_hmm}: {ex}')
            continue

    print('\nComparison results (best on val macro_f1 -> test):')
    for k, v in results.items():
        vm = v.get('val_metrics', {}) if isinstance(v, dict) else {}
        tm = v.get('test_metrics', {}) if isinstance(v, dict) else {}
        print(
            f"{k}: epoch={v.get('epoch', -1)}, "
            f"val(inst_acc={_fmt_metric(vm.get('instance_acc', np.nan))}, macro_f1={_fmt_metric(vm.get('macro_f1', np.nan))}, micro_f1={_fmt_metric(vm.get('micro_f1', np.nan))}, macro_auc={_fmt_metric(vm.get('macro_auc', np.nan))}, macro_aupoc={_fmt_metric(vm.get('macro_aupoc', np.nan))}, hloss={_fmt_metric(vm.get('hamming_loss', np.nan))}), "
            f"test(inst_acc={_fmt_metric(tm.get('instance_acc', np.nan))}, macro_f1={_fmt_metric(tm.get('macro_f1', np.nan))}, micro_f1={_fmt_metric(tm.get('micro_f1', np.nan))}, macro_auc={_fmt_metric(tm.get('macro_auc', np.nan))}, macro_aupoc={_fmt_metric(tm.get('macro_aupoc', np.nan))}, hloss={_fmt_metric(tm.get('hamming_loss', np.nan))})"
        )

    if len(results) > 0:
        best_val_macro_f1_name, best_val_macro_f1 = max(
            results.items(),
            key=lambda kv: float(kv[1].get('val_metrics', {}).get('macro_f1', -1e9))
        )
        best_test_macro_f1_name, best_test_macro_f1 = max(
            results.items(),
            key=lambda kv: float(kv[1].get('test_metrics', {}).get('macro_f1', -1e9))
        )
        best_test_inst_acc_name, best_test_inst_acc = max(
            results.items(),
            key=lambda kv: float(kv[1].get('test_metrics', {}).get('instance_acc', -1e9))
        )
        print('\nBest summary:')
        print(
            f"best val macro_f1 => {best_val_macro_f1_name}, "
            f"macro_f1={_fmt_metric(best_val_macro_f1.get('val_metrics', {}).get('macro_f1', np.nan))}, "
            f"epoch={best_val_macro_f1.get('epoch', -1)}"
        )
        print(
            f"best test macro_f1 => {best_test_macro_f1_name}, "
            f"macro_f1={_fmt_metric(best_test_macro_f1.get('test_metrics', {}).get('macro_f1', np.nan))}, "
            f"epoch={best_test_macro_f1.get('epoch', -1)}"
        )
        print(
            f"best test instance_acc => {best_test_inst_acc_name}, "
            f"instance_acc={_fmt_metric(best_test_inst_acc.get('test_metrics', {}).get('instance_acc', np.nan))}, "
            f"epoch={best_test_inst_acc.get('epoch', -1)}"
        )


if __name__ == '__main__':
    main()
