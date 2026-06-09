import argparse
import csv
import torch
import yaml
import os
import json
import random
import numpy as np
import re
import warnings
warnings.filterwarnings("ignore")
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import pickle
import matplotlib.pyplot as plt
from transformers import EsmForMaskedLM, EsmTokenizer

try:
    from peft import PeftModel, PeftConfig
except Exception:
    PeftModel = None
    PeftConfig = None

# from transformers import T5ForConditionalGeneration
from sklearn.metrics import accuracy_score
from tqdm import *
from tqdm.contrib import tenumerate
from torch.utils.data import DataLoader
from torch import nn

from utils.bert_aff import BERT_AFF, ProtBERT_AFF, Bert_AFF_LSTM, Bert_iAFF_LSTM, Bert_LSTM
from utils.lstm import LSTM_ML
from utils.tokenizer import PeptideTokenizer
from utils.mask import mask_seq
from utils.dataset import PeptideDataset, collate_fn, form_loader, form_ft_loader
from utils.form_loader import form_ml_dataloader
from utils.losses import LDAMLoss, FocalLoss, FocalDiceLoss, DCSLoss, BinaryDiceLoss, BCEFocalLoss, ZLPRLoss
from utils.metrics import instances_overall_metrics, label_overall_metrics, overall_metrics 
from utils.validation import validate
from utils.data_processer import PeptideDataProcessor
from motif_plot import plot_abp_sequences, plot_key_scores, plot_key_scores_and_delta_G
from esm2_finetune.model_wrappers import PepTrainableESM2_AFF_LSTM

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_pssm(path):
    with open(path, "r", encoding='utf-8') as f1:
        one_pssm = []
        nr = 0
        for index, line in enumerate(f1):
            # 判断是否是psiblast对比
            if index == 0 and line[0] == '\n':
                nr = 1
                continue
            if line[0] == '\n': break # 去除psiblast后半部分
            tmp = line.strip().split(' ')
            if nr:
            # 去除'', 取前20个元素
                tmp = [i for i in tmp if i != ''][2:22]
            one_pssm.append(tmp)
        if nr:
            # 去除前两行
            one_pssm = one_pssm[2:]
    # print(one_pssm)
    return one_pssm


def load_hhm_30(hhm_path, max_length):
    """Parse HHM file into [max_length, 30] feature matrix (20 emission + 10 transition)."""
    rows = []
    with open(hhm_path, "r", encoding='utf-8') as f:
        lines = f.readlines()

    i = 0
    n = len(lines)
    while i < n:
        line = lines[i].strip()
        # Match residue row like: "F 1 ... 1"
        if line and re.match(r'^[A-Za-z]\s+\d+\s+', line):
            parts1 = line.split()
            emit_raw = parts1[2:-1] if len(parts1) >= 23 else parts1[2:22]
            emit_raw = emit_raw[:20]

            # transition row is usually the next non-empty line
            j = i + 1
            while j < n and lines[j].strip() == '':
                j += 1
            trans_raw = []
            if j < n:
                parts2 = lines[j].strip().split()
                trans_raw = parts2[:10]

            if len(emit_raw) == 20 and len(trans_raw) == 10:
                raw30 = emit_raw + trans_raw
                vec = []
                for inf in raw30:
                    if inf == '*' or inf == '0':
                        vec.append(0.0)
                    else:
                        try:
                            vec.append(pow(2.0, (-0.0001 * int(inf))))
                        except Exception:
                            vec.append(0.0)
                rows.append(vec)
            i = j + 1
            continue
        i += 1

    if len(rows) == 0:
        raise ValueError(f"Failed to parse HHM rows from: {hhm_path}")

    hhm = np.asarray(rows, dtype=np.float32)
    if hhm.shape[0] < max_length:
        pad = np.zeros((max_length - hhm.shape[0], 30), dtype=np.float32)
        hhm = np.concatenate([hhm, pad], axis=0)
    else:
        hhm = hhm[:max_length]

    # Normalize per sample to 0..1 (same scaling style as PSSM)
    hmax, hmin = float(hhm.max()), float(hhm.min())
    if hmax > hmin:
        hhm = (hhm - hmin) / (hmax - hmin)
    return hhm.astype(np.float32)


def load_inferred_csvs(pssm_csv, hhm_csv):
    def read_csv(path):
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                raise ValueError(f"Empty CSV: {path}")
            cols = [c for c in reader.fieldnames if c not in {"seq_id", "pos"}]
            data = {}
            for row in reader:
                seq_id = row["seq_id"]
                pos = int(row["pos"])
                vec = [float(row[c]) for c in cols]
                data.setdefault(seq_id, {})[pos] = vec
            return data

    return read_csv(pssm_csv), read_csv(hhm_csv)


def build_matrix(table, seq_id, length, expected_dim):
    if seq_id not in table:
        raise ValueError(f"Missing seq_id in CSV: {seq_id}")
    rows = []
    for pos in range(1, length + 1):
        if pos not in table[seq_id]:
            raise ValueError(f"Missing position {pos} for {seq_id}")
        rows.append(table[seq_id][pos])
    arr = np.asarray(rows, dtype=np.float32)
    if arr.shape[1] != expected_dim:
        raise ValueError(f"{seq_id} dim mismatch: expected {expected_dim}, got {arr.shape[1]}")
    return arr


def minmax_norm_np(arr):
    amin = float(np.min(arr))
    amax = float(np.max(arr))
    denom = amax - amin
    if denom == 0.0:
        return np.zeros_like(arr)
    return (arr - amin) / denom

parser = argparse.ArgumentParser()

# model path, vocab path, special token path, train data path
parser.add_argument('--model_path', type=str, default="model/t5-base/pytorch_model.bin")
parser.add_argument('--vocab_path', type=str, default="utils/uniprot_1kmer_vocab.txt")
parser.add_argument('--special_token_path', type=str, default="utils/special_tokens.json")
parser.add_argument('--pretrained_data_path', type=str, default="data/ft_data/train/seqs.fasta")
parser.add_argument('--save_path', type=str, default="model/1027/normal.bin")
parser.add_argument('--model_dir', type=str, default="/home/dataset-local/chenzixu/PepSEF/01_pretraining/esm2_650m_pretraining/outputs/finetuned_stage2_best")
parser.add_argument('--esm_base_model_dir', type=str, default="/home/dataset-local/chenzixu/PepSEF/01_pretraining/esm2_650m_pretraining/models/esm2-650M")
parser.add_argument('--esm2_wrapper_ckpt', type=str, default="/home/dataset-local/chenzixu/PepSEF/02_downstream_multilabel_gptesm2/outputs/model/model_esm2.bin")
parser.add_argument('--fusion_method', type=str, default='cross_attention', choices=['cross_attention', 'aff', 'concat'])
parser.add_argument('--pssm_hmm', type=str, default='None', choices=['pssm', 'hmm', 'both', 'none'])
parser.add_argument('--pssm_dropout', type=float, default=0.1)
parser.add_argument('--fusion_alpha_init', type=float, default=-1.2)
parser.add_argument('--hhm_root', type=str, default='data/hhm')

parser.add_argument('-e', '--epochs', type=int, default=500)
parser.add_argument('-bs', '--batch_size', type=int, default=128)
parser.add_argument('-lr', '--learning_rate', type=float, default=5e-5)
parser.add_argument('-l', '--max_length', type=int, default=128)


args = parser.parse_args()
set_seed(42)

if str(args.pssm_hmm).lower() != "both":
    print(f"[INFO] Override pssm_hmm={args.pssm_hmm} -> both for inferred features")
    args.pssm_hmm = "both"
if str(args.fusion_method).lower() != "cross_attention":
    print(f"[INFO] Override fusion_method={args.fusion_method} -> cross_attention for inferred features")
    args.fusion_method = "cross_attention"

config_file = open('model/bert_model/config.yaml', "r", encoding='utf-8')
config_data = config_file.read()
bert_config = yaml.load(config_data, Loader=yaml.CLoader)
config_file.close()
# print(config1)

config_file = open('model/esm2/config.json', "r", encoding='utf-8')
esm2_config = json.load(config_file)
config_file.close()

config_file = open('model/bert_pssm/lstm_config.yaml', "r", encoding='utf-8')
config_data = config_file.read()
lstm_config = yaml.load(config_data, Loader=yaml.CLoader)
config_file.close()
# print(config2)

config_file = open('model/bert_pssm/aff_config.yaml', "r", encoding='utf-8')
config_data = config_file.read()
aff_config = yaml.load(config_data, Loader=yaml.CLoader)
config_file.close()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'device:{device}')
args.device = device

# 使用 gptesm2 训练流程中的 ESM2 + CrossAttentionFusion 包装器
loaded = False
try:
    if PeftModel is not None:
        base_model = EsmForMaskedLM.from_pretrained(args.esm_base_model_dir, trust_remote_code=True)
        if PeftConfig is not None:
            try:
                peft_cfg = PeftConfig.from_pretrained(args.model_dir)
                if getattr(peft_cfg, 'task_type', None) == 'CAUSAL_LM':
                    peft_cfg.task_type = 'FEATURE_EXTRACTION'
                esm_model = PeftModel.from_pretrained(base_model, args.model_dir, config=peft_cfg)
            except Exception:
                esm_model = PeftModel.from_pretrained(base_model, args.model_dir)
        else:
            esm_model = PeftModel.from_pretrained(base_model, args.model_dir)
        esm_model = esm_model.to(device)
        esm_tokenizer = EsmTokenizer.from_pretrained(args.model_dir)
        loaded = True

    if not loaded:
        esm_model = EsmForMaskedLM.from_pretrained(args.model_dir, trust_remote_code=True).to(device)
        try:
            esm_tokenizer = EsmTokenizer.from_pretrained(args.model_dir)
        except Exception:
            esm_tokenizer = EsmTokenizer.from_pretrained(args.esm_base_model_dir)
except Exception:
    esm_model = EsmForMaskedLM.from_pretrained(args.esm_base_model_dir, trust_remote_code=True).to(device)
    esm_tokenizer = EsmTokenizer.from_pretrained(args.esm_base_model_dir)

model = PepTrainableESM2_AFF_LSTM(
    args,
    esm2_config,
    aff_config,
    lstm_config,
    esm_model=esm_model,
    tokenizer=esm_tokenizer,
    multi_lstm=False,
    freeze_esm=True,
).to(device)

wrapper_ckpt_candidates = [
    args.esm2_wrapper_ckpt,
    '/home/dataset-local/chenzixu/PepSEF/02_downstream_multilabel_gptesm2/outputs/model/model_esm2.bin',
    '/home/dataset-local/chenzixu/PepSEF/02_downstream_multilabel_gptesm2/outputs/model/model_esm2.bin',
]
wrapper_ckpt_path = None
for cand in wrapper_ckpt_candidates:
    if cand and os.path.exists(cand):
        wrapper_ckpt_path = cand
        break

if wrapper_ckpt_path is not None:
    wrapper_ckpt = torch.load(wrapper_ckpt_path, map_location=device)
    model.load_state_dict(wrapper_ckpt, strict=False)
    print(f"[INFO] Loaded wrapper checkpoint: {wrapper_ckpt_path}")
else:
    print("[WARN] No wrapper checkpoint found in candidates:")
    for cand in wrapper_ckpt_candidates:
        print(f"       - {cand}")
    print("[WARN] Running with current wrapper weights.")

peptides = ['MSRRKQAKPQHI', 'TCRSSGRYCRSPYDRRRRYCRRITDACV', 'TSFAEYWNLLSP']
seq_ids = ["seq1", "seq2", "seq3"]
target_key_scores = [
    [0.3576, 0.2612, 1.0, 0.8653, 0.9748, 0.3068, 0.3019, 0.297, 0.0, 0.3495, 0.4699, 0.2563],
    [0.0993, 0.2003, 1.0, 0.0226, 0.2787, 0.2091, 0.6951, 0.1063, 0.3554, 0.7021, 0.1847, 0.0767, 0.108, 0.1411, 0.4443, 0.1847, 0.5575, 0.4843, 0.1098, 0.2422, 0.0, 0.9181, 0.0819, 0.2317, 0.0784, 0.0819, 0.2073, 0.2997],
    [0.2461, 0.4447, 1.0, 0.1046, 0.5434, 0.3448, 0.7039, 0.0975, 0.0, 0.6801, 0.0107, 0.2652]
]

pssm_csv = 'data/ascan/pssm_inferred.csv'
hhm_csv = 'data/ascan/hhm_inferred.csv'
pssm_table, hhm_table = load_inferred_csvs(pssm_csv, hhm_csv)

features_all = []
delta_G = [
    [-0.2, -0.4, 19.40, 16.0, 14.5, 2.2, 0.0, 4.5, 2.8, 0.0, 2.6, 2.9],
    [-0.33, 0.0, 5.17, 0.17, 1.25, -0.63, 2.67, -0.17, 0.0, 4.92, 1.17, 0.33, -0.08, 0.25, 1.67, 0.5, 1.25, 3.5, 0.33, 0.0, 0.17, 3.5, 0.33, 0.75, 0.08, 0.0, 0.0, 0.25],
    [0.39, 1.24, 5.46, 0.0, 1.1, 3.06, 6.31, -1.1, -0.17, 3.28, 0.12, -0.25]
    ]
# print(seq_pssm)

# val_acc, val_pre, val_rec, val_f1 = validate(args, val_dataloader, model, phase='val')
# test_acc, test_pre, test_rec, test_f1 = validate(args, test_dataloader, model, phase='test')
# print(f'val_acc:{val_acc}, val_pre:{val_pre}, val_rec:{val_rec}, val_f1:{val_f1}')
# print(f'test_acc:{test_acc}, test_pre:{test_pre}, test_rec:{test_rec}, test_f1:{test_f1}')

for seq_id, pep in zip(seq_ids, peptides):
    pssm_arr = build_matrix(pssm_table, seq_id, len(pep), 20)
    hhm_arr = build_matrix(hhm_table, seq_id, len(pep), 30)

    if args.max_length > len(pep):
        pad_len = args.max_length - len(pep)
        pssm_arr = np.concatenate([pssm_arr, np.zeros((pad_len, 20), dtype=np.float32)], axis=0)
        hhm_arr = np.concatenate([hhm_arr, np.zeros((pad_len, 30), dtype=np.float32)], axis=0)
    else:
        pssm_arr = pssm_arr[:args.max_length]
        hhm_arr = hhm_arr[:args.max_length]

    feat = np.concatenate((pssm_arr, hhm_arr.astype(np.float32)), axis=1)
    features_all.append(feat)

print(np.array(features_all).shape)
if args.pssm_hmm == 'both':
    print(f"[INFO] pssm_hmm={args.pssm_hmm}, expected feature_dim=50, actual={features_all[0].shape[-1]}")

model.eval()
key_scores = []
for index, (pep, feat) in enumerate(zip(peptides, features_all)):
    valid_lens = len(pep)
    pssm_info = torch.tensor(feat, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs, key_score = model([pep], pssm_info, attention_mask=None)
    pred = torch.where(outputs > 0, 1, 0).float()
    print("pep_len, pssm_info.shape:", valid_lens, pssm_info.shape)
    print(pred)
    print(key_score.shape)
    key_np = key_score.cpu().detach().numpy()[0, :valid_lens]
    key_norm = minmax_norm_np(key_np)
    target = np.asarray(target_key_scores[index], dtype=np.float32)
    mse = float(np.mean((key_norm - target) ** 2))
    print(f"[INFO] key_score MSE vs target (seq{index+1}): {mse:.6f}")
    key_scores.append(key_np.reshape(1, -1))
    # plt.savefig(f'figures/line{index}.png', dpi=300, bbox_inches='tight')
    # plt.show()

# plot_key_scores(peptides, key_scores, output_path='figures')
# Plot key_scores and delta_G
plot_key_scores_and_delta_G(peptides, key_scores, delta_G, output_path='figures')
    
