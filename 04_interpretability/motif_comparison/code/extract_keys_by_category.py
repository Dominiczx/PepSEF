#!/usr/bin/env python3
import os
import json
import torch
import yaml
import argparse
from types import SimpleNamespace
from tqdm import tqdm
import numpy as np
import re

# project imports
from utils.tokenizer import PeptideTokenizer
from utils.bert_aff import Bert_AFF_LSTM
import pickle
import math


def read_fasta(fn):
    seqs = []
    with open(fn, 'r', encoding='utf-8') as f:
        cur = None
        for line in f:
            line = line.strip()
            if not line: continue
            if line.startswith('>'):
                cur = None
                continue
            seqs.append(line)
    return seqs


def main():
    base = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    # mimic test_acc defaults
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default=os.path.join(base, 'data/ft_data/by_category_combined'))
    parser.add_argument('--out_dir', type=str, default=os.path.join(base, 'data/ft_data/by_category_combined'))
    parser.add_argument('--model_ckpt', type=str, default='output/model/50.bin')
    parser.add_argument('--max_length', type=int, default=128)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--top_k', type=int, default=3)
    parser.add_argument('--window', type=int, default=3)
    args = parser.parse_args()

    # build args namespace for tokenizer and model
    cfg = SimpleNamespace()
    cfg.vocab_path = os.path.join(base, 'utils/uniprot_1kmer_vocab.txt')
    cfg.special_token_path = os.path.join(base, 'utils/special_tokens.json')
    cfg.max_length = args.max_length
    cfg.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cfg.max_len = args.max_length
    cfg.pssm_hmm = 'pssm'  # we will provide zero pssm features if none exist

    # tokenizer
    tokenizer = PeptideTokenizer(cfg)
    cfg.tokenizer = tokenizer
    cfg.vocab_size = tokenizer.vocab_size

    # load model configs similar to test_acc
    # load bert_config and others; but Bert_AFF_LSTM only needs bert_config and lstm_config and aff_config files
    base_model_dir = os.path.join(base, 'model/bert_model')
    with open(os.path.join(base_model_dir, 'config.yaml'), 'r', encoding='utf-8') as f:
        bert_config = yaml.load(f, Loader=yaml.CLoader)

    with open(os.path.join(base, 'model/bert_pssm/lstm_config.yaml'), 'r', encoding='utf-8') as f:
        lstm_config = yaml.load(f, Loader=yaml.CLoader)
    with open(os.path.join(base, 'model/bert_pssm/aff_config.yaml'), 'r', encoding='utf-8') as f:
        aff_config = yaml.load(f, Loader=yaml.CLoader)
    aff_config['use_hhm'] = False

    # instantiate model
    model = Bert_AFF_LSTM(cfg, bert_config, aff_config, lstm_config, multi_lstm=False)
    device = cfg.device
    model.to(device)

    # load checkpoint
    ckpt = torch.load(args.model_ckpt, map_location=device)
    try:
        model.load_state_dict(ckpt)
    except Exception:
        # some checkpoints might be dict-wrapped
        try:
            model.load_state_dict(ckpt['model'])
        except Exception:
            model.load_state_dict(ckpt)

    model.eval()

    # iterate fasta files
    files = [fn for fn in os.listdir(args.data_dir) if fn.endswith('.fasta')]
    files.sort()
    os.makedirs(args.out_dir, exist_ok=True)

    for fn in files:
        path = os.path.join(args.data_dir, fn)
        seqs = read_fasta(path)
        # prepare per-fasta pssm and hhm files (one pssm csv and one hhm .npy per fasta)
        base_name = os.path.splitext(fn)[0]
        pssm_out_fn = os.path.join(args.out_dir, base_name + '_pssm.csv')
        hhm_out_fn = os.path.join(args.out_dir, base_name + '_hhm.npy')
        # ensure mapping/blocks are loaded
        if not hasattr(main, '_pssm_map'):
            # load mapping and blocks
            try:
                pkl_fn = os.path.join(base, 'data/ft_data/pssm_seq2fn.pkl')
                with open(pkl_fn, 'rb') as pf:
                    main._pssm_map = pickle.load(pf)
            except Exception:
                main._pssm_map = {}
            # parse blocks
            try:
                csv_fn = os.path.join(base, 'data/ft_data/pssm.csv')
                cur = None
                rows = []
                blocks = {}
                with open(csv_fn, 'r', encoding='utf-8') as cf:
                    for line in cf:
                        line = line.strip()
                        if not line:
                            continue
                        if line.startswith('>'):
                            if cur is not None:
                                try:
                                    blocks[cur] = np.array(rows, dtype=np.float32)
                                except Exception:
                                    blocks[cur] = None
                            cur = line[1:].strip()
                            rows = []
                            continue
                        parts = line.split(',')
                        if len(parts) >= 20:
                            try:
                                vals = [float(x) for x in parts[:20]]
                                rows.append(vals)
                            except Exception:
                                pass
                    if cur is not None:
                        try:
                            blocks[cur] = np.array(rows, dtype=np.float32)
                        except Exception:
                            blocks[cur] = None
                main._pssm_blocks = blocks
            except Exception:
                main._pssm_blocks = {}
            # optional blosum fallback
            try:
                blosum_fn = os.path.join(base, 'data/ft_data/blosum62.pkl')
                with open(blosum_fn, 'rb') as bf:
                    main._blosum = pickle.load(bf)
            except Exception:
                main._blosum = None

        # helper: normalize and pad/truncate pssm array to max_length
        def _normalize_and_pad(one_arr, max_len):
            if one_arr is None:
                return np.zeros((max_len, 20), dtype=np.float32), 0
            arr = np.array(one_arr, dtype=np.float32)
            if arr.size == 0:
                return np.zeros((max_len, 20), dtype=np.float32), 0
            # min-max normalize per-sequence
            min_num = float(arr.min())
            max_num = float(arr.max())
            if math.isclose(max_num, min_num):
                norm = np.zeros_like(arr, dtype=np.float32)
            else:
                norm = (arr - min_num) / (max_num - min_num)
            L = norm.shape[0]
            if L >= max_len:
                norm = norm[:max_len]
            else:
                pad = np.zeros((max_len - L, norm.shape[1]), dtype=np.float32)
                norm = np.concatenate([norm, pad], axis=0)
            return norm, L

        # helper: canonicalize sequence for map lookup
        def _canonical_seq(s):
            if s is None:
                return s
            s2 = s.strip().upper()
            # replace uncommon residues with X (same as data processor)
            s2 = re.sub('[UZOB]', 'X', s2)
            return s2

        # generate per-fasta pssm csv if missing
        if not os.path.exists(pssm_out_fn):
            with open(pssm_out_fn, 'w', encoding='utf-8') as pf:
                for idx, s in enumerate(seqs):
                    pf.write('>' + str(idx) + '\n')
                    arr = None
                    # try multiple canonical keys
                    if hasattr(main, '_pssm_map'):
                        for candidate in (s, s.strip(), s.upper(), _canonical_seq(s)):
                            if candidate in main._pssm_map:
                                key = main._pssm_map.get(candidate)
                                arr = main._pssm_blocks.get(key)
                                if arr is not None:
                                    break
                    # fallback to blosum-derived rows if available
                    if arr is None and hasattr(main, '_blosum') and main._blosum is not None:
                        rows = [main._blosum.get(aa, [0]*20)[:20] for aa in _canonical_seq(s)]
                        arr, L = _normalize_and_pad(np.array(rows, dtype=np.float32), args.max_length)
                        # write normalized floats
                        for r in arr:
                            pf.write(' '.join([f"{float(x):.6f}" for x in r]) + '\n')
                        continue
                    # if we have a raw block, normalize and write
                    if arr is None:
                        # write zero rows (max_length rows)
                        for _ in range(args.max_length):
                            pf.write(' '.join(['0']*20) + '\n')
                    else:
                        normed, L = _normalize_and_pad(arr, args.max_length)
                        for r in normed:
                            pf.write(' '.join([f"{float(x):.6f}" for x in r]) + '\n')

        # generate per-fasta hhm numpy file if missing (we don't have per-seq hhm sources; produce zeros)
        if not os.path.exists(hhm_out_fn):
            hhm_arr = np.zeros((len(seqs), args.max_length, 30), dtype=np.float32)
            try:
                np.save(hhm_out_fn, hhm_arr)
            except Exception:
                # fallback: write as .npy via pickle
                import pickle as _p
                with open(hhm_out_fn + '.pkl', 'wb') as _f:
                    _p.dump(hhm_arr, _f)
        out_fn = os.path.join(args.out_dir, fn + '.key_subsequences.jsonl')
        print(f'Processing {fn}: {len(seqs)} sequences -> {out_fn}')
        with open(out_fn, 'w', encoding='utf-8') as outf:
            # process in batches
            for i in range(0, len(seqs), args.batch_size):
                batch = seqs[i:i+args.batch_size]
                input_ids = []
                attention_masks = []
                valid_lens = []
                for s in batch:
                    ids, attn = tokenizer.encode_plus(s, padding=True)
                    # encode_plus returns (input_ids, attention_mask)
                    input_ids.append(ids)
                    attention_masks.append(attn)
                    valid_lens.append(sum(attn))
                # pad lists into tensors (they are already padded to max_length by tokenizer)
                tokens = torch.tensor(input_ids, dtype=torch.long, device=device)
                # build per-sequence PSSM features (use mapping + pssm.csv blocks if available)
                # cache mapping/blocks on the function object to avoid repeated loads
                if not hasattr(main, '_pssm_map'):
                    # attempt to load mapping file
                    pssm_map = {}
                    pssm_blocks = {}
                    try:
                        pkl_fn = os.path.join(base, 'data/ft_data/pssm_seq2fn.pkl')
                        with open(pkl_fn, 'rb') as pf:
                            pssm_map = pickle.load(pf)
                    except Exception:
                        pssm_map = {}
                    # parse pssm.csv into blocks: >NAME header followed by rows
                    try:
                        csv_fn = os.path.join(base, 'data/ft_data/pssm.csv')
                        cur = None
                        rows = []
                        with open(csv_fn, 'r', encoding='utf-8') as cf:
                            for line in cf:
                                line = line.strip()
                                if not line: continue
                                if line.startswith('>'):
                                    if cur is not None:
                                        pssm_blocks[cur] = np.array(rows, dtype=np.float32)
                                    cur = line[1:].strip()
                                    rows = []
                                    continue
                                parts = line.split(',')
                                if len(parts) >= 20:
                                    try:
                                        vals = [float(x) for x in parts[:20]]
                                        rows.append(vals)
                                    except Exception:
                                        # skip malformed
                                        pass
                            if cur is not None:
                                pssm_blocks[cur] = np.array(rows, dtype=np.float32)
                    except Exception:
                        pssm_blocks = {}
                    main._pssm_map = pssm_map
                    main._pssm_blocks = pssm_blocks

                pssm_list = []
                missing_cnt = 0
                for s in batch:
                    arr = None
                    try:
                        # try multiple canonical forms
                        if hasattr(main, '_pssm_map') and main._pssm_map:
                            for candidate in (s, s.strip(), s.upper(), _canonical_seq(s)):
                                if candidate in main._pssm_map:
                                    key = main._pssm_map.get(candidate)
                                    arr = main._pssm_blocks.get(key)
                                    if arr is not None:
                                        break
                    except Exception:
                        arr = None
                    # if still none, try blosum fallback
                    if arr is None and hasattr(main, '_blosum') and main._blosum is not None:
                        rows = [main._blosum.get(aa, [0]*20)[:20] for aa in _canonical_seq(s)]
                        normed, L = _normalize_and_pad(np.array(rows, dtype=np.float32), args.max_length)
                        pssm_list.append(normed)
                        continue
                    if arr is None:
                        # fallback to zeros
                        missing_cnt += 1
                        pssm_list.append(np.zeros((args.max_length, 20), dtype=np.float32))
                    else:
                        normed, L = _normalize_and_pad(arr, args.max_length)
                        pssm_list.append(normed)

                if missing_cnt > 0:
                    print(f'Batch {i}-{i+len(batch)}: {missing_cnt}/{len(batch)} sequences missing PSSM (using zeros)')

                pssm = torch.tensor(np.stack(pssm_list, axis=0), dtype=torch.float32, device=device)
                # call model to get attention weights (for debugging / output) and extraction
                with torch.no_grad():
                    # obtain attention weights from the bert-aff module
                    try:
                        _fusion, attn_weights = model.bert_aff((tokens, pssm))
                    except Exception:
                        # fallback: some models expect separate args
                        try:
                            _fusion, attn_weights = model.bert_aff(tokens, pssm)
                        except Exception:
                            attn_weights = None

                    # aggregated importance per key: mean over heads then max over queries
                    if attn_weights is not None:
                        # attn_weights: [B, H, Lq, Lk]
                        importance = attn_weights.mean(dim=1).max(dim=1).values.cpu().numpy()  # [B, Lk]
                    else:
                        importance = None

                    sel_idx, key_ids, key_subseqs = model.extract_key_subsequences(tokens, pssm, valid_lens, top_k=args.top_k, window=args.window)
                # write per-sequence
                for bi, (seq, idxs, subseqs) in enumerate(zip(batch, sel_idx, key_subseqs)):
                    # idxs is a list of windows (list of indices), subseqs is list of token-strings or joined strings
                    rec = {
                        'sequence': seq,
                        'selected_indices': idxs,
                        'key_subsequences': subseqs,
                    }
                    # attach attention importance vector (trim to valid length) for debugging
                    if importance is not None:
                        imp = importance[bi]
                        vl = int(valid_lens[bi])
                        rec['attention_importance'] = imp[:vl].tolist()
                    outf.write(json.dumps(rec, ensure_ascii=False) + '\n')
        print(f'Wrote {out_fn}')

if __name__ == '__main__':
    main()
