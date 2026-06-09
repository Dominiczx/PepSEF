#!/usr/bin/env python3
import argparse
import json
import os
import random
import sys
from typing import List, Tuple

import numpy as np
import torch
import yaml
from transformers import EsmForMaskedLM, EsmTokenizer

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

try:
    from peft import PeftModel, PeftConfig
except Exception:
    PeftModel = None
    PeftConfig = None

from esm2_finetune.model_wrappers import PepTrainableESM2_AFF_LSTM

AA_ORDER = ["A", "R", "N", "D", "C", "Q", "E", "G", "H", "I", "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V"]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def minmax_norm(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    x_min = torch.min(x)
    x_max = torch.max(x)
    denom = x_max - x_min
    if float(denom) < eps:
        return torch.zeros_like(x)
    return (x - x_min) / denom


def build_esm_cache(model: PepTrainableESM2_AFF_LSTM, seq: str, max_length: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    inputs = model.tokenizer(
        [seq],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length + 2,
    )
    ids = inputs.input_ids.to(device)
    attention_mask = inputs.attention_mask.to(device)

    with torch.no_grad():
        out = model.esm_model(ids, attention_mask=attention_mask, output_hidden_states=True, return_dict=True)
        if hasattr(out, "hidden_states") and out.hidden_states is not None:
            esm_hidden = out.hidden_states[-1]
        elif isinstance(out, tuple) and len(out) > 0:
            esm_hidden = out[0]
        else:
            esm_hidden = getattr(out, "logits", out)

    if esm_hidden is not None and esm_hidden.shape[1] >= 2:
        esm_hidden = esm_hidden[:, 1:-1, :]
        attention_mask = attention_mask[:, 1:-1]
    return esm_hidden, attention_mask


def compute_key_scores_cached(
    model: PepTrainableESM2_AFF_LSTM,
    esm_hidden: torch.Tensor,
    attention_mask: torch.Tensor,
    pssm: torch.Tensor,
) -> torch.Tensor:
    proj = model.proj_norm(esm_hidden)
    target_L = pssm.shape[1]
    if proj.shape[1] < target_L:
        pad_len = target_L - proj.shape[1]
        proj = torch.nn.functional.pad(proj, (0, 0, 0, pad_len), mode="constant", value=0)
    elif proj.shape[1] > target_L:
        proj = proj[:, :target_L, :]

    if attention_mask is not None:
        key_attn = attention_mask
        if key_attn.shape[1] < target_L:
            pad_len = target_L - key_attn.shape[1]
            key_attn = torch.nn.functional.pad(key_attn, (0, pad_len), value=0)
        elif key_attn.shape[1] > target_L:
            key_attn = key_attn[:, :target_L]
        key_attn = key_attn.to(proj.device)
    else:
        key_attn = torch.ones((proj.shape[0], proj.shape[1]), device=proj.device, dtype=torch.long)

    pssm_in = pssm.float()
    if model.fusion_method == "cross_attention" and model.caf is not None:
        caf_out, _ = model.caf(proj, pssm_in, attention_mask=key_attn)
        gate = torch.sigmoid(model.fusion_gate_logit)
        aff_result = proj + gate * (caf_out - proj)
    else:
        aff_result = proj

    key_scores = torch.sum(aff_result, 2) / (aff_result.shape[2] + 1e-8)
    return key_scores


def fit_one_sequence(
    model: PepTrainableESM2_AFF_LSTM,
    seq: str,
    target_scores: List[float],
    max_length: int,
    device: torch.device,
    steps: int,
    lr: float,
    reg_entropy: float,
    reg_smooth: float,
) -> Tuple[np.ndarray, np.ndarray]:
    L = len(seq)
    target = torch.tensor(target_scores, dtype=torch.float32, device=device)

    esm_hidden, attn_mask = build_esm_cache(model, seq, max_length, device)

    raw_pssm = torch.nn.Parameter(torch.randn((L, 20), device=device) * 0.05)
    raw_hhm = torch.nn.Parameter(torch.randn((L, 30), device=device) * 0.05)

    optimizer = torch.optim.Adam([raw_pssm, raw_hhm], lr=lr)

    best_loss = None
    best_pssm = None
    best_hhm = None

    for step in range(steps):
        pssm_sig = torch.sigmoid(raw_pssm)
        hhm_sig = torch.sigmoid(raw_hhm)

        pssm = minmax_norm(pssm_sig)
        hhm = minmax_norm(hhm_sig)

        feat = torch.cat([pssm, hhm], dim=1).unsqueeze(0)
        key_scores = compute_key_scores_cached(model, esm_hidden, attn_mask, feat)[0, :L]
        key_scores = minmax_norm(key_scores)

        mse = torch.mean((key_scores - target) ** 2)
        # PSSM: encourage a peaky distribution per position
        p = torch.softmax(pssm, dim=1)
        entropy = -(p * torch.log(p + 1e-8)).sum(dim=1).mean()
        # HHM: encourage smoothness across positions
        smooth = ((hhm[1:] - hhm[:-1]) ** 2).mean()

        loss = mse + reg_entropy * entropy + reg_smooth * smooth

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_val = float(loss.item())
        if best_loss is None or loss_val < best_loss:
            best_loss = loss_val
            best_pssm = pssm.detach().cpu().numpy()
            best_hhm = hhm.detach().cpu().numpy()

        if (step + 1) % 100 == 0 or step == 0:
            print(f"[INFO] {seq} step {step+1}/{steps} loss={loss_val:.6f} mse={float(mse.item()):.6f}")

    return best_pssm, best_hhm


def write_pssm_csv(path: str, rows: List[Tuple[str, int, np.ndarray]]) -> None:
    header = ["seq_id", "pos"] + AA_ORDER
    with open(path, "w", encoding="utf-8") as f:
        f.write(",".join(header) + "\n")
        for seq_id, pos, vec in rows:
            vec_str = ",".join(f"{v:.6f}" for v in vec.tolist())
            f.write(f"{seq_id},{pos},{vec_str}\n")


def write_hhm_csv(path: str, rows: List[Tuple[str, int, np.ndarray]]) -> None:
    header = ["seq_id", "pos"] + [f"E{i}" for i in range(1, 21)] + [f"T{i}" for i in range(1, 11)]
    with open(path, "w", encoding="utf-8") as f:
        f.write(",".join(header) + "\n")
        for seq_id, pos, vec in rows:
            vec_str = ",".join(f"{v:.6f}" for v in vec.tolist())
            f.write(f"{seq_id},{pos},{vec_str}\n")


def load_configs():
    with open("model/esm2/config.json", "r", encoding="utf-8") as f:
        esm2_config = json.load(f)
    with open("model/bert_pssm/lstm_config.yaml", "r", encoding="utf-8") as f:
        lstm_config = yaml.load(f, Loader=yaml.CLoader)
    with open("model/bert_pssm/aff_config.yaml", "r", encoding="utf-8") as f:
        aff_config = yaml.load(f, Loader=yaml.CLoader)
    return esm2_config, lstm_config, aff_config


def build_model(args, esm2_config, lstm_config, aff_config, device):
    loaded = False
    try:
        if PeftModel is not None:
            base_model = EsmForMaskedLM.from_pretrained(args.esm_base_model_dir, trust_remote_code=True)
            if PeftConfig is not None:
                try:
                    peft_cfg = PeftConfig.from_pretrained(args.model_dir)
                    if getattr(peft_cfg, "task_type", None) == "CAUSAL_LM":
                        peft_cfg.task_type = "FEATURE_EXTRACTION"
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
        "/home/dataset-local/chenzixu/PepSEF/02_downstream_multilabel_gptesm2/outputs/model/model_esm2.bin",
        "/home/dataset-local/chenzixu/PepSEF/02_downstream_multilabel_gptesm2/outputs/model/model_esm2.bin",
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

    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, default="/home/dataset-local/chenzixu/PepSEF/04_interpretability/motif_comparison/runs/inferred_ascan_features")
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--lr", type=float, default=0.08)
    parser.add_argument("--reg_entropy", type=float, default=0.02)
    parser.add_argument("--reg_smooth", type=float, default=0.005)
    parser.add_argument("--seed", type=int, default=42)
    args_cli = parser.parse_args()

    set_seed(args_cli.seed)

    # Model args
    class Args:
        pass

    args = Args()
    args.model_dir = "/home/dataset-local/chenzixu/PepSEF/01_pretraining/esm2_650m_pretraining/outputs/finetuned_stage2_best"
    args.esm_base_model_dir = "/home/dataset-local/chenzixu/PepSEF/01_pretraining/esm2_650m_pretraining/models/esm2-650M"
    args.esm2_wrapper_ckpt = "/home/dataset-local/chenzixu/PepSEF/02_downstream_multilabel_gptesm2/outputs/model/model_esm2.bin"
    args.fusion_method = "cross_attention"
    args.pssm_hmm = "both"

    sequences = [
        "MSRRKQAKPQHI",
        "TCRSSGRYCRSPYDRRRRYCRRITDACV",
        "TSFAEYWNLLSP",
    ]
    targets = [
        [0.3576, 0.2612, 1.0, 0.8653, 0.9748, 0.3068, 0.3019, 0.297, 0.0, 0.3495, 0.4699, 0.2563],
        [0.0993, 0.2003, 1.0, 0.0226, 0.2787, 0.2091, 0.6951, 0.1063, 0.3554, 0.7021, 0.1847, 0.0767, 0.108, 0.1411, 0.4443, 0.1847, 0.5575, 0.4843, 0.1098, 0.2422, 0.0, 0.9181, 0.0819, 0.2317, 0.0784, 0.0819, 0.2073, 0.2997],
        [0.2461, 0.4447, 1.0, 0.1046, 0.5434, 0.3448, 0.7039, 0.0975, 0.0, 0.6801, 0.0107, 0.2652],
    ]

    max_length = max(len(s) for s in sequences)
    args.max_length = max_length

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device: {device}")

    esm2_config, lstm_config, aff_config = load_configs()
    model = build_model(args, esm2_config, lstm_config, aff_config, device)

    os.makedirs(args_cli.out_dir, exist_ok=True)

    pssm_rows = []
    hhm_rows = []

    for seq_id, (seq, target_scores) in enumerate(zip(sequences, targets), start=1):
        pssm, hhm = fit_one_sequence(
            model,
            seq,
            target_scores,
            max_length=max_length,
            device=device,
            steps=args_cli.steps,
            lr=args_cli.lr,
            reg_entropy=args_cli.reg_entropy,
            reg_smooth=args_cli.reg_smooth,
        )
        for pos in range(1, len(seq) + 1):
            pssm_rows.append((f"seq{seq_id}", pos, pssm[pos - 1]))
            hhm_rows.append((f"seq{seq_id}", pos, hhm[pos - 1]))

    pssm_path = os.path.join(args_cli.out_dir, "pssm_inferred.csv")
    hhm_path = os.path.join(args_cli.out_dir, "hhm_inferred.csv")
    write_pssm_csv(pssm_path, pssm_rows)
    write_hhm_csv(hhm_path, hhm_rows)

    print(f"[INFO] Wrote PSSM CSV: {pssm_path}")
    print(f"[INFO] Wrote HHM CSV: {hhm_path}")


if __name__ == "__main__":
    main()
