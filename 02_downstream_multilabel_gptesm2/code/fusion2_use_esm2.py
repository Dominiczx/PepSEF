#!/usr/bin/env python3
"""
fusion2_use_esm2.py

A variant of fusion2.py that substitutes the BERT-based model with the finetuned ESM-2 model.
It mirrors the training/validation loop from fusion2.py but constructs an `ESM2_AFF_LSTM` wrapper
and injects the finetuned ESM-2 weights/tokenizer located at the user-provided path.

Usage: run similarly to fusion2.py. This script will attempt to load LoRA/PEFT adapters
from the finetuned directory using `peft.PeftModel.from_pretrained` when available, and
falls back to loading the model directory directly with `EsmForMaskedLM.from_pretrained`.
"""

import argparse
import torch
import yaml
import os
import json
import random
import numpy as np
import warnings
from datetime import datetime
import socket
warnings.filterwarnings("ignore")
# os.environ["CUDA_VISIBLE_DEVICES"] = "4, 5, 6, 7"

from tqdm import *
from torch.utils.data import DataLoader
from torch import nn
from utils.tokenizer import PeptideTokenizer
from utils.data_processer import PeptideDataProcessor
from utils.metrics import instances_overall_metrics, label_overall_metrics
from utils.validation import validate
from utils.losses import FocalDiceLoss, AsymmetricLoss

# import ESM wrapper classes
from utils.bert_aff import ESM2_AFF_LSTM, ESM2_AFF, PepESM2_AFF_LSTM
from esm2_finetune.model_wrappers import PepTrainableESM2_AFF_LSTM
from transformers import EsmForMaskedLM, EsmTokenizer

# try PEFT for loading adapters
try:
    from peft import PeftModel, PeftConfig
except Exception:
    PeftModel = None
    PeftConfig = None


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_pos_weight(dataset, cap=None):
    labels = np.asarray(dataset.labels, dtype=np.float32)
    if labels.ndim != 2:
        return torch.ones(15, dtype=torch.float32)
    pos = labels.sum(axis=0)
    neg = labels.shape[0] - pos
    pos_weight = neg / (pos + 1e-6)
    if cap is not None and cap > 0:
        pos_weight = np.clip(pos_weight, 1.0, float(cap))
    return torch.tensor(pos_weight, dtype=torch.float32)


def _binary_score(y_true_c, y_pred_c, mode='balanced'):
    y_true_c = y_true_c.astype(np.int32)
    y_pred_c = y_pred_c.astype(np.int32)
    tp = np.sum((y_true_c == 1) & (y_pred_c == 1))
    tn = np.sum((y_true_c == 0) & (y_pred_c == 0))
    fp = np.sum((y_true_c == 0) & (y_pred_c == 1))
    fn = np.sum((y_true_c == 1) & (y_pred_c == 0))
    acc = (tp + tn) / max(tp + tn + fp + fn, 1)
    pre = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = (2 * pre * rec / (pre + rec)) if (pre + rec) > 0 else 0.0
    if mode == 'acc':
        return acc
    if mode == 'f1':
        return f1
    return 0.5 * acc + 0.5 * f1


def evaluate_with_threshold(model, dataloader, esm_tokenizer, device, max_length, threshold=0.5, search_threshold=False, opt_metric='balanced', f1_mode='macro', threshold_mode='per_class', max_batches=None):
    all_probs, all_true = [], []
    model.eval()
    with torch.no_grad():
        for bidx, (seq, token, feature, label, valid_len, attention_mask) in enumerate(dataloader):
            seqs = list(seq)
            tokenized = esm_tokenizer(
                seqs,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=max_length + 2
            )
            ids = tokenized.input_ids.to(device)
            token_attn = tokenized.attention_mask.to(device)
            features = feature.to(device)
            labels = label.to(device)

            outputs, _ = model(ids, features, token_attn)
            probs = torch.sigmoid(outputs).detach().cpu().numpy()
            y_true = labels.detach().cpu().numpy()
            all_probs.append(probs)
            all_true.append(y_true)
            if max_batches is not None and (bidx + 1) >= int(max_batches):
                break

    if len(all_probs) == 0:
        empty_extra = {
            'hloss': 0.0,
            'aupoc_macro': np.nan,
            'auc_macro': np.nan,
            'mcc_macro': np.nan,
        }
        return 0.0, 0.0, 0.0, 0.0, threshold, empty_extra

    probs = np.concatenate(all_probs, axis=0)
    y_true = np.concatenate(all_true, axis=0)

    if isinstance(threshold, (list, tuple, np.ndarray)):
        best_threshold = np.array(threshold, dtype=np.float32)
    else:
        best_threshold = float(threshold)
    if search_threshold:
        if threshold_mode == 'per_class':
            n_class = y_true.shape[1]
            cand = np.linspace(0.05, 0.90, 86)
            thrs = np.full((n_class,), 0.5, dtype=np.float32)
            for c in range(n_class):
                yc = y_true[:, c]
                best_s = -1.0
                best_t = 0.5
                for t in cand:
                    pc = (probs[:, c] >= t).astype(np.int32)
                    s = _binary_score(yc, pc, mode=opt_metric)
                    if s > best_s:
                        best_s = s
                        best_t = float(t)
                thrs[c] = best_t
            best_threshold = thrs
        else:
            best_score = -1.0
            for thr in np.linspace(0.05, 0.90, 86):
                pred = (probs >= thr).astype(np.float32)
                met = instances_overall_metrics(pred, y_true)
                label_met = label_overall_metrics(pred, y_true)
                f1 = float(label_met['F1'][0] if f1_mode == 'macro' else label_met['F1'][1])
                if opt_metric == 'acc':
                    score = met['Accuracy']
                elif opt_metric == 'f1':
                    score = f1
                else:
                    score = 0.5 * f1 + 0.5 * met['Accuracy']
                if score > best_score:
                    best_score = score
                    best_threshold = float(thr)

    pred = (probs >= best_threshold).astype(np.float32)
    met = instances_overall_metrics(pred, y_true)
    p, r = met['Precision'], met['Recall']
    # Use probability scores for score-based metrics (AUC/AUPR).
    label_met = label_overall_metrics(probs, y_true)
    f1 = float(label_met['F1'][0] if f1_mode == 'macro' else label_met['F1'][1])
    extra = {
        'hloss': float(met.get('HLoss', 0.0)),
        # AUPOC is reported with AUPR-compatible computation for consistency.
        'aupoc_macro': float(label_met['AUPR'][0]),
        'auc_macro': float(label_met['AUC'][0]),
        'mcc_macro': float(label_met['MCC'][0]),
    }
    return float(met['Accuracy']), float(p), float(r), float(f1), best_threshold, extra


parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', type=str, required=False, default="/home/dataset-local/chenzixu/PepSEF/01_pretraining/esm2_650m_pretraining/outputs/finetuned_stage2_best")
parser.add_argument('--save_path', type=str, default="/home/dataset-local/chenzixu/PepSEF/02_downstream_multilabel_gptesm2/outputs/model/model_esm2.bin")
parser.add_argument('--csv_save_path', type=str, default="/home/dataset-local/chenzixu/PepSEF/02_downstream_multilabel_gptesm2/results/training_result_esm2.csv")
parser.add_argument('-e', '--epochs', type=int, default=100)
parser.add_argument('-bs', '--batch_size', type=int, default=32)
parser.add_argument('-lr', '--learning_rate', type=float, default=5e-6)
parser.add_argument('-l', '--max_length', type=int, default=128)
parser.add_argument('-p_h', '--pssm_hmm', type=str, default='both')
parser.add_argument('--seed', type=int, default=123)
parser.add_argument('--unfreeze_esm_layers', type=int, default=32)
parser.add_argument('--clip_grad_norm', type=float, default=1.0)
parser.add_argument('--use_scheduler', action='store_true', default=True)
parser.add_argument('--gpu_ids', type=str, default="0,1")
parser.add_argument('--master_port', type=int, default=12355)
parser.add_argument('--loss_type', type=str, default='focal_dice', choices=['bce', 'focal_dice', 'asym'])
parser.add_argument('--use_aug', action='store_true', default=False)
parser.add_argument('--opt_metric', type=str, default='balanced', choices=['acc', 'f1', 'balanced'])
parser.add_argument('--early_stop_patience', type=int, default=10)
parser.add_argument('--threshold_search_start', type=int, default=1)
parser.add_argument('--f1_mode', type=str, default='macro', choices=['macro', 'micro'])
parser.add_argument('--pos_weight_cap', type=float, default=20.0)
parser.add_argument('--threshold_mode', type=str, default='per_class', choices=['global', 'per_class'])
parser.add_argument('--smoke_test', action='store_true', default=False)
parser.add_argument('--max_train_batches', type=int, default=0)
parser.add_argument('--max_eval_batches', type=int, default=0)
parser.add_argument('--head_lr_mult', type=float, default=8.0)
parser.add_argument('--fusion_lr_mult', type=float, default=24.0)
parser.add_argument('--min_improve_delta', type=float, default=1e-4)
parser.add_argument('--pssm_dropout', type=float, default=0.1)
parser.add_argument('--fusion_alpha_init', type=float, default=-1.2)
parser.add_argument('--hhm_root', type=str, default='data/hhm')
parser.add_argument('--fusion_method', type=str, default='cross_attention', choices=['cross_attention', 'aff', 'concat'])

args = parser.parse_args()
set_seed(args.seed)

# normalize pssm/hmm option for downstream components
if args.pssm_hmm is None:
    args.pssm_hmm = 'none'
else:
    args.pssm_hmm = str(args.pssm_hmm).strip().lower()
    if args.pssm_hmm in {'null', 'none', 'no', 'false'}:
        args.pssm_hmm = 'none'
if args.pssm_hmm not in {'pssm', 'hmm', 'both', 'none'}:
    print(f"[WARN] Unknown pssm_hmm='{args.pssm_hmm}', fallback to 'both'.")
    args.pssm_hmm = 'both'

if args.smoke_test:
    # fast sanity-check configuration
    args.epochs = 1
    if args.max_train_batches <= 0:
        args.max_train_batches = 1
    if args.max_eval_batches <= 0:
        args.max_eval_batches = 1
    args.threshold_search_start = 999999

# ensure save directories exist
try:
    if args.save_path:
        os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    if args.csv_save_path:
        os.makedirs(os.path.dirname(args.csv_save_path), exist_ok=True)
except Exception:
    pass

# log start time (only in main process)

# ensure tokenizer args exist (defaults used in original script)
if not hasattr(args, 'vocab_path'):
    args.vocab_path = 'utils/uniprot_1kmer_vocab.txt'
if not hasattr(args, 'special_token_path'):
    args.special_token_path = 'utils/special_tokens.json'
if not hasattr(args, 'max_length'):
    args.max_length = 128
if not hasattr(args, 'save'):
    args.save = True

# load configs
with open('model/bert_model/config.yaml', 'r', encoding='utf-8') as f:
    bert_config = yaml.load(f, Loader=yaml.CLoader)
with open('model/esm2/config.json', 'r', encoding='utf-8') as f:
    esm2_config = json.load(f)
with open('model/Prott5/config.json', 'r', encoding='utf-8') as f:
    prot_t5_config = json.load(f)
with open('model/bert_pssm/lstm_config.yaml', 'r', encoding='utf-8') as f:
    lstm_config = yaml.load(f, Loader=yaml.CLoader)
with open('model/bert_pssm/aff_config.yaml', 'r', encoding='utf-8') as f:
    aff_config = yaml.load(f, Loader=yaml.CLoader)

# Filter aff_config
if isinstance(aff_config, dict):
    aff_config = {k: aff_config[k] for k in ('channels', 'r') if k in aff_config}

# tokenizer and data
tokenizer = PeptideTokenizer(args)
args.tokenizer = tokenizer
args.vocab_size = tokenizer.vocab_size
path = 'data/ft_data/'
aug_flag = bool(args.use_aug)
pdp = PeptideDataProcessor(path, args, aug=aug_flag)
train_dataset, train_dataloader, val_dataset, val_dataloader, test_dataset, test_dataloader = pdp.form_ml_dataloader()
pass

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from esm2_finetune.ddp_helpers import set_args_device, prepare_dataloaders_for_ddp, run_validate_on_device


def main_worker(rank, world_size, requested_gpu_ids):
    # map rank to absolute GPU id and set device
    if requested_gpu_ids is not None and len(requested_gpu_ids) > 0:
        abs_gpu = requested_gpu_ids[rank]
    else:
        abs_gpu = rank
    device = torch.device(f'cuda:{abs_gpu}') if torch.cuda.is_available() else torch.device('cpu')
    if torch.cuda.is_available():
        torch.cuda.set_device(device)

    # initialize process group only when using multi-GPU DDP
    if world_size > 1 and torch.cuda.is_available():
        dist.init_process_group(backend='nccl', init_method=f'tcp://127.0.0.1:{args.master_port}', world_size=world_size, rank=rank)
        # DDP for train only. Evaluate on full val/test set at rank 0 to avoid split-metric bias.
        train_loader, _, _, train_sampler, _, _ = prepare_dataloaders_for_ddp(
            train_dataset, val_dataset, test_dataset, args.batch_size, world_size, rank, num_workers_train=4, num_workers_val=2
        )
        if rank == 0:
            val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
            test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
        else:
            val_loader, test_loader = None, None
    else:
        train_loader, val_loader, test_loader = train_dataloader, val_dataloader, test_dataloader
        train_sampler = None

    if rank == 0:
        try:
            sample_feature = train_dataset[0][2]
            print(f"[rank {rank}] pssm_hmm={args.pssm_hmm}, feature_dim={sample_feature.shape[-1]}, feature_shape={tuple(sample_feature.shape)}")
            print(f"[rank {rank}] fusion_method={args.fusion_method}")
            if args.pssm_hmm in ('hmm', 'both'):
                found_hhm = int(getattr(pdp, 'train_hhm_found_count', -1))
                print(f"[rank {rank}] HHM root={args.hhm_root}, train_hhm_found={found_hhm}/{len(train_dataset)}")
        except Exception as _e:
            print(f"[rank {rank}] pssm_hmm={args.pssm_hmm}, feature shape unavailable: {_e}")

    # load model and tokenizer inside each process
    esm_model_dir = args.model_dir
    loaded = False
    try:
        if PeftModel is not None:
            base_path = '/home/dataset-local/chenzixu/PepSEF/01_pretraining/esm2_650m_pretraining/models/esm2-650M'
            base = EsmForMaskedLM.from_pretrained(base_path, trust_remote_code=True)
            if PeftConfig is not None:
                try:
                    peft_cfg = PeftConfig.from_pretrained(esm_model_dir)
                    if getattr(peft_cfg, 'task_type', None) == 'CAUSAL_LM':
                        peft_cfg.task_type = 'FEATURE_EXTRACTION'
                    peft_wrapped = PeftModel.from_pretrained(base, esm_model_dir, config=peft_cfg)
                except Exception:
                    peft_wrapped = PeftModel.from_pretrained(base, esm_model_dir)
            else:
                peft_wrapped = PeftModel.from_pretrained(base, esm_model_dir)
            esm_model = peft_wrapped.to(device)
            esm_tokenizer = EsmTokenizer.from_pretrained(esm_model_dir)
            loaded = True
        if not loaded:
            direct = EsmForMaskedLM.from_pretrained(esm_model_dir, trust_remote_code=True).to(device)
            esm_model = direct
            try:
                esm_tokenizer = EsmTokenizer.from_pretrained(esm_model_dir)
            except Exception:
                esm_tokenizer = EsmTokenizer.from_pretrained('/home/dataset-local/chenzixu/PepSEF/01_pretraining/esm2_650m_pretraining/models/esm2-650M')
            loaded = True
    except Exception:
        return

    # set worker-local device into args so downstream utilities (validate, etc.) can access it
    set_args_device(args, device)

    # build wrapper and optionally unfreeze
    model_wrapper = PepTrainableESM2_AFF_LSTM(args, esm2_config, aff_config, lstm_config, esm_model=esm_model, tokenizer=esm_tokenizer, multi_lstm=False, freeze_esm=True)
    if args.unfreeze_esm_layers and args.unfreeze_esm_layers > 0:
        esm_mod = getattr(model_wrapper, 'esm_model', None) or getattr(getattr(model_wrapper, 'pep_esm_aff', None), 'esm_model', None)
        if esm_mod is not None:
            import re
            # Prefer explicit layer access when available (ESM2 HF naming)
            base = getattr(esm_mod, 'esm', None)
            encoder = getattr(base, 'encoder', None) if base is not None else getattr(esm_mod, 'encoder', None)
            layer_list = getattr(encoder, 'layer', None)
            if layer_list is not None and len(layer_list) > 0:
                # unfreeze last N layers by module index
                start = max(0, len(layer_list) - args.unfreeze_esm_layers)
                for layer in layer_list[start:]:
                    for p in layer.parameters():
                        p.requires_grad = True
            else:
                # fallback to name regex for last N layers
                names = [n for n, _ in esm_mod.named_parameters()]
                layer_idxs = []
                for n in names:
                    m = re.search(r"layer[._](\d+)", n)
                    if m:
                        layer_idxs.append(int(m.group(1)))
                if len(layer_idxs) > 0:
                    max_idx = max(layer_idxs)
                    cutoff = max_idx - args.unfreeze_esm_layers + 1
                    for n, p in esm_mod.named_parameters():
                        m = re.search(r"layer[._](\d+)", n)
                        if m and int(m.group(1)) >= cutoff:
                            p.requires_grad = True

    model_wrapper.to(device)
    if world_size > 1 and torch.cuda.is_available():
        ddp_model = torch.nn.parallel.DistributedDataParallel(model_wrapper, device_ids=[abs_gpu], find_unused_parameters=True)
    else:
        ddp_model = model_wrapper

    # optimizer and scheduler
    if args.loss_type == 'bce':
        pos_weight = compute_pos_weight(train_dataset, cap=args.pos_weight_cap).to(device)
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        if rank == 0:
            print(f'[rank {rank}] Using BCEWithLogitsLoss with pos_weight (min={pos_weight.min().item():.3f}, max={pos_weight.max().item():.3f})')
    elif args.loss_type == 'asym':
        criterion = AsymmetricLoss(gamma_neg=4, gamma_pos=1, clip=0.05)
        if rank == 0:
            print(f'[rank {rank}] Using AsymmetricLoss')
    else:
        criterion = FocalDiceLoss()
        bce_aux = torch.nn.BCEWithLogitsLoss(pos_weight=compute_pos_weight(train_dataset, cap=args.pos_weight_cap).to(device))
        if rank == 0:
            print(f'[rank {rank}] Using FocalDiceLoss + 0.3*BCEWithLogitsLoss')
    # differential LR: ESM (low), fusion blocks (high), classifier head (mid)
    named_params = [(n, p) for n, p in ddp_model.named_parameters() if p.requires_grad]
    esm_params = [p for n, p in named_params if 'esm_model' in n]
    fusion_name_keys = ('caf', 'fusion_gate_logit', 'proj_norm')
    fusion_params = [p for n, p in named_params if ('esm_model' not in n) and any(k in n for k in fusion_name_keys)]
    head_params = [p for n, p in named_params if ('esm_model' not in n) and (not any(k in n for k in fusion_name_keys))]
    head_lr = min(max(args.learning_rate * args.head_lr_mult, args.learning_rate), 1e-3)
    fusion_lr = min(max(args.learning_rate * args.fusion_lr_mult, args.learning_rate), 2e-3)
    param_groups = []
    if len(esm_params) > 0:
        param_groups.append({'params': esm_params, 'lr': args.learning_rate})
    if len(fusion_params) > 0:
        param_groups.append({'params': fusion_params, 'lr': fusion_lr})
    if len(head_params) > 0:
        param_groups.append({'params': head_params, 'lr': head_lr})
    if len(param_groups) == 0:
        raise RuntimeError('No trainable parameters found for optimizer.')
    optimizer = torch.optim.AdamW(param_groups, lr=args.learning_rate, weight_decay=1e-2)
    if rank == 0:
        print(f'[rank {rank}] Optimizer groups: esm={len(esm_params)} params @ {args.learning_rate:g}, fusion={len(fusion_params)} params @ {fusion_lr:g}, head={len(head_params)} params @ {head_lr:g}')
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3) if args.use_scheduler else None

    # training loop
    max_acc = 0.0
    best_score = -1e9
    best_thr = 0.5
    no_improve = 0
    for e in range(args.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(e)
        ddp_model.train()
        total_acc = 0.0
        total_pre = 0.0
        total_rec = 0.0
        total_f1 = 0.0
        for bidx, (seq, token, feature, label, valid_len, attention_mask) in enumerate(train_loader):
            seqs = list(seq)
            tokenized = esm_tokenizer(
                seqs,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=args.max_length + 2
            )
            ids = tokenized.input_ids.to(device)
            token_attn = tokenized.attention_mask.to(device)
            features = feature.to(device)
            labels = label.to(device)

            # pass tokenizer attention mask to ESM; wrapper will align to pssm length
            outputs, key_scores = ddp_model(ids, features, token_attn)
            if args.loss_type == 'focal_dice':
                loss = criterion(outputs.float(), labels.float()) + 0.3 * bce_aux(outputs.float(), labels.float())
            else:
                loss = criterion(outputs.float(), labels.float())
            optimizer.zero_grad()
            loss.backward()
            if args.clip_grad_norm and args.clip_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, ddp_model.parameters()), args.clip_grad_norm)
            optimizer.step()
            probs = torch.sigmoid(outputs)
            pred = (probs > 0.5).float().cpu().detach().numpy()
            y_true = labels.cpu().detach().numpy()
            df = instances_overall_metrics(pred, y_true)
            total_acc += df['Accuracy']
            total_pre += df['Precision']
            total_rec += df['Recall']
            denom = (df['Precision'] + df['Recall'])
            total_f1 += (2 * df['Precision'] * df['Recall'] / denom) if denom > 0 else 0.0
            if args.max_train_batches and args.max_train_batches > 0 and (bidx + 1) >= args.max_train_batches:
                break
        # aggregate mean metrics across workers using true number of local batches
        if args.max_train_batches and args.max_train_batches > 0:
            local_batches = float(min(len(train_loader), args.max_train_batches))
        else:
            local_batches = float(len(train_loader))
        metric_tensor = torch.tensor([total_acc, total_pre, total_rec, total_f1, local_batches], device=device, dtype=torch.float64)
        if world_size > 1 and torch.cuda.is_available():
            dist.all_reduce(metric_tensor, op=dist.ReduceOp.SUM)
        denom_batches = max(metric_tensor[4].item(), 1.0)
        mean_acc = metric_tensor[0].item() / denom_batches
        mean_pre = metric_tensor[1].item() / denom_batches
        mean_rec = metric_tensor[2].item() / denom_batches
        mean_f1 = metric_tensor[3].item() / denom_batches
        if rank == 0:
            gate_info = ''
            try:
                mm = ddp_model.module if isinstance(ddp_model, torch.nn.parallel.DistributedDataParallel) else ddp_model
                if hasattr(mm, 'fusion_gate_logit'):
                    gate_info = f", gate={torch.sigmoid(mm.fusion_gate_logit).item():.3f}"
            except Exception:
                pass
            print(f'epoch {e+1}: [rank {rank}] train acc:{mean_acc}, pre:{mean_pre}, rec:{mean_rec}, f1:{mean_f1}{gate_info}')
        stop_flag = torch.tensor([0], device=device, dtype=torch.int32)
        if rank == 0:
            search_thr = (e + 1) >= args.threshold_search_start
            val_acc, val_pre, val_rec, val_f1, tuned_thr, val_extra = evaluate_with_threshold(
                ddp_model, val_loader, esm_tokenizer, device, args.max_length, threshold=best_thr, search_threshold=search_thr, opt_metric=args.opt_metric, f1_mode=args.f1_mode, threshold_mode=args.threshold_mode, max_batches=(args.max_eval_batches if args.max_eval_batches and args.max_eval_batches > 0 else None)
            )
            best_thr = tuned_thr
            if args.opt_metric == 'acc':
                cur_score = val_acc
            elif args.opt_metric == 'f1':
                cur_score = val_f1
            else:
                cur_score = 0.35 * val_acc + 0.65 * val_f1

            if scheduler is not None:
                try:
                    scheduler.step(cur_score)
                except Exception:
                    pass

            if isinstance(best_thr, np.ndarray):
                thr_info = f"mean={best_thr.mean():.3f}, min={best_thr.min():.3f}, max={best_thr.max():.3f}"
            else:
                thr_info = f"{best_thr:.3f}"
            print(
                f"epoch {e+1}: [rank {rank}] val acc:{val_acc}, pre:{val_pre}, rec:{val_rec}, f1:{val_f1}, "
                f"hloss:{val_extra['hloss']}, aupoc:{val_extra['aupoc_macro']}, auc:{val_extra['auc_macro']}, mcc:{val_extra['mcc_macro']}, thr:{thr_info}"
            )

            improved = cur_score > (best_score - 5e-3)
            # improved = True
            prev_best_score = best_score
            print(f'epoch {e+1}: [rank {rank}] score_check cur={cur_score:.6f}, best={prev_best_score:.6f}, improved={improved}')
            if improved:
                best_score = cur_score
                max_acc = max(max_acc, val_acc)
                no_improve = 0
                try:
                    test_acc, test_pre, test_rec, test_f1, _, test_extra = evaluate_with_threshold(
                        ddp_model, test_loader, esm_tokenizer, device, args.max_length, threshold=best_thr, search_threshold=False, f1_mode=args.f1_mode, threshold_mode=args.threshold_mode, max_batches=(args.max_eval_batches if args.max_eval_batches and args.max_eval_batches > 0 else None)
                    )
                    print(
                        f"epoch {e+1}: [rank {rank}] test acc:{test_acc}, pre:{test_pre}, rec:{test_rec}, f1:{test_f1}, "
                        f"hloss:{test_extra['hloss']}, aupoc:{test_extra['aupoc_macro']}, auc:{test_extra['auc_macro']}, mcc:{test_extra['mcc_macro']}, thr:{thr_info}"
                    )
                except Exception as e_test:
                    print(f'epoch {e+1}: [rank {rank}] [ERROR] test evaluation failed: {e_test}')

                # Export per-class metrics CSV for the current best model.
                # This writes args.csv_save_path and will be refreshed whenever best model updates.
                try:
                    _ = run_validate_on_device(args, test_loader, ddp_model, phase='test', save_csv=True, device=device)
                    print(f'epoch {e+1}: [rank {rank}] saved per-class CSV to: {args.csv_save_path}')
                except Exception as e_csv:
                    print(f'epoch {e+1}: [rank {rank}] [ERROR] save_csv failed: {e_csv}')

                # save model (unwrap DDP)
                if isinstance(ddp_model, torch.nn.parallel.DistributedDataParallel):
                    torch.save(ddp_model.module.state_dict(), args.save_path)
                else:
                    torch.save(ddp_model.state_dict(), args.save_path)
                if args.smoke_test:
                    print('[SMOKE_TEST] One mini-train/eval pass completed successfully.')
                    stop_flag[0] = 1
            else:
                no_improve += 1
                if no_improve >= args.early_stop_patience:
                    print(f'epoch {e+1}: [rank {rank}] early stopping (no improvement for {no_improve} epochs)')
                    stop_flag[0] = 1

        if world_size > 1 and torch.cuda.is_available():
            dist.broadcast(stop_flag, src=0)
            dist.barrier()
        if stop_flag.item() == 1:
            break

    # cleanup
    if world_size > 1 and torch.cuda.is_available():
        dist.destroy_process_group()


if __name__ == '__main__':
    # one-time run header
    print(f"Run started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"train_dataset:{len(train_dataset)}\n val_dataset:{len(val_dataset)} \n test_dataset:{len(test_dataset)}")
    print(f"GPU ids: {args.gpu_ids}")
    # determine requested GPU ids
    requested_gpu_ids = [int(x) for x in args.gpu_ids.split(',')] if args.gpu_ids else list(range(torch.cuda.device_count()))
    world_size = len(requested_gpu_ids) if torch.cuda.is_available() else 1
    if world_size > 1:
        # choose an available TCP port for DDP rendezvous
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            s.bind(('127.0.0.1', int(args.master_port)))
        except OSError:
            s.bind(('127.0.0.1', 0))
            args.master_port = int(s.getsockname()[1])
            print(f"[INFO] master_port in use, switched to {args.master_port}")
        finally:
            s.close()
        # spawn processes
        mp.spawn(main_worker, args=(world_size, requested_gpu_ids), nprocs=world_size, join=True)
    else:
        # single-process fallback: run the same worker function (no distributed init)
        main_worker(0, 1, requested_gpu_ids)

pass
