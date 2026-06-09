#!/usr/bin/env python3
"""
Extract cross-attention from fusion2_use_esm2 best model, map attention back to sequence
positions, split outputs by class, and plot attention distributions ranked by match score.

This script is standalone and reuses the same model stack as fusion2_use_esm2.py:
- finetuned ESM2 backbone from --model_dir
- fusion wrapper PepTrainableESM2_AFF_LSTM
- optional trained fusion checkpoint from --fusion_ckpt

Outputs:
- all_records.jsonl
- by_class/<CLASS>.jsonl
- by_class/<CLASS>_top_match.csv
- plots/<CLASS>_attention_distribution.png
- summary.json
"""

import argparse
import csv
import json
import math
import os
import re
from collections import defaultdict
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from transformers import EsmForMaskedLM, EsmTokenizer

try:
    from peft import PeftConfig, PeftModel
except Exception:
    PeftConfig = None
    PeftModel = None

from utils.data_processer import PeptideDataProcessor
from utils.tokenizer import PeptideTokenizer
from esm2_finetune.model_wrappers import PepTrainableESM2_AFF_LSTM


FALLBACK_LABELS = [
    "AMP", "TXP", "ABP", "AIP", "AVP", "ACP", "AFP", "DDV", "CPP", "CCC",
    "APP", "AAP", "AHTP", "PBP", "QSP"
]


@dataclass
class AttentionResult:
    logits: torch.Tensor
    probs: torch.Tensor
    key_scores: torch.Tensor
    attn_importance: torch.Tensor
    key_mask: torch.Tensor


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, default="/home/dataset-local/chenzixu/PepSEF/01_pretraining/esm2_650m_pretraining/outputs/finetuned_stage2_best")
    parser.add_argument("--fusion_ckpt", type=str, default="/home/dataset-local/chenzixu/PepSEF/02_downstream_multilabel_gptesm2/outputs/model/model_esm2.bin")
    parser.add_argument("--base_esm_dir", type=str, default="/home/dataset-local/chenzixu/PepSEF/01_pretraining/esm2_650m_pretraining/models/esm2-650M")
    parser.add_argument("--data_root", type=str, default="data/ft_data")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--output_dir", type=str, default="data/ft_data/attention_fusion2_esm2")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--pssm_hmm", type=str, default="both", choices=["pssm", "hmm", "both", "none"])
    parser.add_argument("--fusion_method", type=str, default="cross_attention", choices=["cross_attention", "aff", "concat"])
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument("--window", type=int, default=3)
    parser.add_argument("--top_n_plot", type=int, default=30)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--motif_map_json", type=str, default="")
    parser.add_argument("--include_top1_if_none", action="store_true", default=True)
    return parser.parse_args()


def read_label_names(data_root: str, split: str) -> List[str]:
    labels_csv = os.path.join(data_root, split, "labels.csv")
    if os.path.exists(labels_csv):
        with open(labels_csv, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            if header:
                return [h.strip() for h in header]
    return FALLBACK_LABELS


def load_motif_map(path: str) -> Dict[str, List[str]]:
    if not path:
        return {}
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    out = {}
    for k, v in data.items():
        if isinstance(v, list):
            out[str(k)] = [str(x).upper() for x in v]
    return out


def build_data(args: argparse.Namespace):
    if not hasattr(args, "vocab_path"):
        args.vocab_path = "utils/uniprot_1kmer_vocab.txt"
    if not hasattr(args, "special_token_path"):
        args.special_token_path = "utils/special_tokens.json"

    tokenizer = PeptideTokenizer(args)
    args.tokenizer = tokenizer
    args.vocab_size = tokenizer.vocab_size

    data_path = args.data_root
    if not data_path.endswith("/"):
        data_path = data_path + "/"
    pdp = PeptideDataProcessor(data_path, args, aug=False)
    train_dataset, train_loader, val_dataset, val_loader, test_dataset, test_loader = pdp.form_ml_dataloader()
    split_map = {
        "train": (train_dataset, train_loader),
        "val": (val_dataset, val_loader),
        "test": (test_dataset, test_loader),
    }
    return tokenizer, split_map[args.split][0], split_map[args.split][1]


def load_configs() -> Tuple[dict, dict, dict]:
    with open("model/esm2/config.json", "r", encoding="utf-8") as f:
        esm2_config = json.load(f)
    with open("model/bert_pssm/lstm_config.yaml", "r", encoding="utf-8") as f:
        lstm_config = yaml.load(f, Loader=yaml.CLoader)
    with open("model/bert_pssm/aff_config.yaml", "r", encoding="utf-8") as f:
        aff_config = yaml.load(f, Loader=yaml.CLoader)
    if isinstance(aff_config, dict):
        aff_config = {k: aff_config[k] for k in ("channels", "r") if k in aff_config}
    return esm2_config, aff_config, lstm_config


def load_esm_model_and_tokenizer(args: argparse.Namespace, device: torch.device):
    loaded = False
    esm_model = None
    esm_tokenizer = None

    if PeftModel is not None:
        try:
            base = EsmForMaskedLM.from_pretrained(args.base_esm_dir, trust_remote_code=True)
            if PeftConfig is not None:
                try:
                    peft_cfg = PeftConfig.from_pretrained(args.model_dir)
                    if getattr(peft_cfg, "task_type", None) == "CAUSAL_LM":
                        peft_cfg.task_type = "FEATURE_EXTRACTION"
                    peft_wrapped = PeftModel.from_pretrained(base, args.model_dir, config=peft_cfg)
                except Exception:
                    peft_wrapped = PeftModel.from_pretrained(base, args.model_dir)
            else:
                peft_wrapped = PeftModel.from_pretrained(base, args.model_dir)
            esm_model = peft_wrapped.to(device)
            esm_tokenizer = EsmTokenizer.from_pretrained(args.model_dir)
            loaded = True
        except Exception:
            loaded = False

    if not loaded:
        esm_model = EsmForMaskedLM.from_pretrained(args.model_dir, trust_remote_code=True).to(device)
        try:
            esm_tokenizer = EsmTokenizer.from_pretrained(args.model_dir)
        except Exception:
            esm_tokenizer = EsmTokenizer.from_pretrained(args.base_esm_dir)

    return esm_model, esm_tokenizer


def load_fusion_model(args: argparse.Namespace, device: torch.device):
    esm2_config, aff_config, lstm_config = load_configs()
    esm_model, esm_tokenizer = load_esm_model_and_tokenizer(args, device)

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

    if args.fusion_ckpt and os.path.exists(args.fusion_ckpt):
        ckpt = torch.load(args.fusion_ckpt, map_location=device)
        state = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
        if isinstance(state, dict):
            cleaned = {}
            for k, v in state.items():
                nk = k[7:] if k.startswith("module.") else k
                cleaned[nk] = v
            state = cleaned
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"[INFO] loaded fusion_ckpt: {args.fusion_ckpt}")
        if missing:
            print(f"[WARN] missing keys count: {len(missing)}")
        if unexpected:
            print(f"[WARN] unexpected keys count: {len(unexpected)}")
    else:
        print("[WARN] fusion_ckpt missing, using wrapper initialized weights.")

    model.eval()
    return model, esm_tokenizer


def forward_with_attention(model: PepTrainableESM2_AFF_LSTM, ids: torch.Tensor, pssm: torch.Tensor, token_attn: torch.Tensor) -> AttentionResult:
    with torch.no_grad():
        out = model.esm_model(ids, attention_mask=token_attn, output_hidden_states=True, return_dict=True)
        if hasattr(out, "hidden_states") and out.hidden_states is not None:
            esm_hidden = out.hidden_states[-1]
        elif isinstance(out, tuple) and len(out) > 0:
            esm_hidden = out[0]
        else:
            esm_hidden = getattr(out, "logits", out)

        if esm_hidden.shape[1] >= 2:
            esm_hidden = esm_hidden[:, 1:-1, :]
            token_attn = token_attn[:, 1:-1]

        proj = model.proj_norm(esm_hidden)
        target_l = pssm.shape[1]
        if proj.shape[1] < target_l:
            proj = torch.nn.functional.pad(proj, (0, 0, 0, target_l - proj.shape[1]), value=0)
        elif proj.shape[1] > target_l:
            proj = proj[:, :target_l, :]

        key_attn = token_attn
        if key_attn.shape[1] < target_l:
            key_attn = torch.nn.functional.pad(key_attn, (0, target_l - key_attn.shape[1]), value=0)
        elif key_attn.shape[1] > target_l:
            key_attn = key_attn[:, :target_l]
        key_attn = key_attn.to(proj.device)

        if model.caf is None:
            raise RuntimeError("fusion_method is not cross_attention or caf module missing.")

        pssm_in = pssm.float().to(proj.device)
        caf_out, attn_weights = model.caf(proj, pssm_in, attention_mask=key_attn)
        gate = torch.sigmoid(model.fusion_gate_logit)
        fused = proj + gate * (caf_out - proj)

        key_scores = torch.sum(fused, 2) / (fused.shape[2] + 1e-8)
        x = torch.nn.functional.leaky_relu(fused)
        mask = key_attn.float().unsqueeze(-1)
        denom = torch.clamp(mask.sum(dim=1), min=1.0)
        pooled = (x * mask).sum(dim=1) / denom
        logits = model.mlp(pooled)
        probs = torch.sigmoid(logits)

        attn_importance = attn_weights.mean(dim=1).max(dim=1).values

    return AttentionResult(
        logits=logits,
        probs=probs,
        key_scores=key_scores,
        attn_importance=attn_importance,
        key_mask=key_attn,
    )


def clean_seq(seq: str) -> str:
    return re.sub(r"[UZOB]", "X", str(seq).upper())


def extract_windows(importance: np.ndarray, valid_len: int, top_k: int, window_size: int) -> List[List[int]]:
    l = int(min(len(importance), max(0, valid_len)))
    if l <= 0:
        return []
    imp = np.asarray(importance[:l], dtype=np.float32)
    w = max(1, int(window_size))
    if w > l:
        w = l

    starts = []
    masses = []
    for s in range(0, l - w + 1):
        starts.append(s)
        masses.append(float(np.sum(imp[s:s + w])))
    masses = np.asarray(masses, dtype=np.float32)

    picked = []
    covered = np.zeros((l,), dtype=bool)
    while len(picked) < int(top_k) and np.isfinite(masses).any():
        s_idx = int(np.argmax(masses))
        score = float(masses[s_idx])
        if not math.isfinite(score):
            break
        s = starts[s_idx]
        e = s + w
        if covered[s:e].any():
            masses[s_idx] = -np.inf
            continue
        picked.append(list(range(s, e)))
        covered[s:e] = True
        for j, sj in enumerate(starts):
            ej = sj + w
            if not (ej <= s or sj >= e):
                masses[j] = -np.inf

    if not picked:
        picked = [list(range(0, w))]
    return picked


def motif_similarity(keys: Sequence[str], motifs: Sequence[str]) -> float:
    if not keys or not motifs:
        return 0.0
    best = 0.0
    for k in keys:
        ku = str(k).upper()
        for m in motifs:
            mu = str(m).upper()
            best = max(best, SequenceMatcher(None, ku, mu).ratio())
    return float(best)


def ensure_dirs(base_out: str) -> Tuple[str, str]:
    by_class_dir = os.path.join(base_out, "by_class")
    plot_dir = os.path.join(base_out, "plots")
    os.makedirs(base_out, exist_ok=True)
    os.makedirs(by_class_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)
    return by_class_dir, plot_dir


def save_jsonl(path: str, records: List[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def write_top_csv(path: str, records: List[dict], top_n: int) -> None:
    cols = [
        "rank", "sample_index", "class_name", "class_prob", "motif_score", "match_score",
        "true_positive", "sequence", "key_subsequences"
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for i, rec in enumerate(records[:top_n], start=1):
            w.writerow([
                i,
                rec.get("sample_index", -1),
                rec.get("class_name", ""),
                f"{float(rec.get('class_prob', 0.0)):.6f}",
                f"{float(rec.get('motif_score', 0.0)):.6f}",
                f"{float(rec.get('match_score', 0.0)):.6f}",
                int(rec.get("true_positive", 0)),
                rec.get("sequence", ""),
                "|".join(rec.get("key_subsequences", [])),
            ])


def plot_attention_distribution(path: str, class_name: str, records: List[dict], top_n: int) -> None:
    pick = records[:max(1, int(top_n))]
    if not pick:
        return
    max_len = max(len(r.get("attention_importance", [])) for r in pick)
    arr = np.full((len(pick), max_len), np.nan, dtype=np.float32)
    y_labels = []
    for i, rec in enumerate(pick):
        imp = np.asarray(rec.get("attention_importance", []), dtype=np.float32)
        arr[i, :len(imp)] = imp
        y_labels.append(f"{i+1}|{rec.get('match_score', 0.0):.3f}")

    fig_w = min(18, max(10, max_len / 8))
    fig_h = min(18, max(4, len(pick) * 0.35 + 2))
    plt.figure(figsize=(fig_w, fig_h))
    plt.imshow(arr, aspect="auto", cmap="Blues")
    plt.colorbar(label="attention importance")
    plt.title(f"{class_name} attention distribution (top {len(pick)} by match score)")
    plt.xlabel("sequence position")
    plt.ylabel("rank|match_score")
    plt.yticks(np.arange(len(y_labels)), y_labels, fontsize=7)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device

    label_names = read_label_names(args.data_root, args.split)
    motif_map = load_motif_map(args.motif_map_json)

    _, dataset, dataloader = build_data(args)
    model, esm_tokenizer = load_fusion_model(args, device)

    if str(args.fusion_method).lower() != "cross_attention":
        print("[WARN] forcing fusion_method to cross_attention for attention extraction.")
        args.fusion_method = "cross_attention"

    base_out = os.path.join(args.output_dir, args.split)
    by_class_dir, plot_dir = ensure_dirs(base_out)

    all_records: List[dict] = []
    class_records: Dict[str, List[dict]] = defaultdict(list)

    global_idx = 0
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            seq, _token, feature, label, valid_len, _attention_mask = batch
            seqs = list(seq)

            tokenized = esm_tokenizer(
                seqs,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=args.max_length + 2,
            )
            ids = tokenized.input_ids.to(device)
            token_attn = tokenized.attention_mask.to(device)
            features = feature.to(device)
            labels = label.cpu().numpy().astype(np.int32)
            valid_lens = valid_len.cpu().numpy().astype(np.int32)

            out = forward_with_attention(model, ids, features, token_attn)
            probs = out.probs.detach().cpu().numpy()
            importance = out.attn_importance.detach().cpu().numpy()
            key_mask = out.key_mask.detach().cpu().numpy()

            for i in range(len(seqs)):
                clean = clean_seq(seqs[i])
                vl = int(valid_lens[i])
                if vl <= 0:
                    vl = int(np.sum(key_mask[i]))
                vl = max(1, min(vl, importance.shape[1], len(clean)))

                imp_vec = importance[i][:vl].astype(float).tolist()
                windows = extract_windows(importance[i], vl, top_k=args.top_k, window_size=args.window)
                key_subseqs = []
                for w in windows:
                    if not w:
                        continue
                    s = max(0, min(w))
                    e = min(vl, max(w) + 1)
                    key_subseqs.append(clean[s:e])

                prob_vec = probs[i]
                pred_idx = np.where(prob_vec >= float(args.threshold))[0].tolist()
                if not pred_idx and args.include_top1_if_none:
                    pred_idx = [int(np.argmax(prob_vec))]
                true_idx = np.where(labels[i] > 0)[0].tolist()

                rec_common = {
                    "sample_index": global_idx,
                    "sequence": clean,
                    "valid_len": vl,
                    "selected_indices": windows,
                    "key_subsequences": key_subseqs,
                    "attention_importance": imp_vec,
                    "true_labels": [label_names[j] for j in true_idx if j < len(label_names)],
                    "pred_labels": [label_names[j] for j in pred_idx if j < len(label_names)],
                }
                all_records.append(rec_common)

                for cidx in pred_idx:
                    if cidx >= len(label_names):
                        continue
                    cname = label_names[cidx]
                    class_prob = float(prob_vec[cidx])
                    motifs = motif_map.get(cname, [])
                    motif_score = motif_similarity(key_subseqs, motifs)
                    match_score = 0.6 * class_prob + 0.4 * motif_score if motifs else class_prob
                    crecord = {
                        **rec_common,
                        "class_name": cname,
                        "class_index": int(cidx),
                        "class_prob": class_prob,
                        "motif_score": motif_score,
                        "match_score": float(match_score),
                        "true_positive": int(cidx in true_idx),
                    }
                    class_records[cname].append(crecord)

                global_idx += 1

    save_jsonl(os.path.join(base_out, "all_records.jsonl"), all_records)

    summary = {
        "split": args.split,
        "num_samples": len(all_records),
        "classes": {},
    }

    for cname in label_names:
        records = class_records.get(cname, [])
        records.sort(key=lambda x: float(x.get("match_score", 0.0)), reverse=True)

        class_jsonl = os.path.join(by_class_dir, f"{cname}.jsonl")
        class_csv = os.path.join(by_class_dir, f"{cname}_top_match.csv")
        class_png = os.path.join(plot_dir, f"{cname}_attention_distribution.png")

        save_jsonl(class_jsonl, records)
        write_top_csv(class_csv, records, top_n=args.top_n_plot)
        if records:
            plot_attention_distribution(class_png, cname, records, top_n=args.top_n_plot)

        tp = sum(int(r.get("true_positive", 0)) for r in records)
        summary["classes"][cname] = {
            "predicted_count": len(records),
            "predicted_true_positive": tp,
            "top_match_score": float(records[0]["match_score"]) if records else 0.0,
            "jsonl": class_jsonl,
            "csv": class_csv,
            "plot": class_png if records else "",
        }

    with open(os.path.join(base_out, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("[DONE] all_records:", os.path.join(base_out, "all_records.jsonl"))
    print("[DONE] summary:", os.path.join(base_out, "summary.json"))


if __name__ == "__main__":
    main()
