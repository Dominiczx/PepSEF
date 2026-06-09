import numpy as np
import pandas as pd
import pickle
import csv
import os
import re
import random
import torch
import math
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN, BorderlineSMOTE
import torch.utils
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
from utils.dataset import FTDataset
from utils.data_augmentation import replacement_dict, replacement_alanine, global_random_shuffling, local_random_shuffling, sequence_revsersion, sequence_subsampling

class PeptideDataProcessor:
    """蛋白质序列数据处理器，用于生成机器学习数据加载器
    
    Attributes:
        path (str): 数据存储路径
        args (argparse.Namespace): 配置参数
        tokenizer: 序列编码器
        aug (bool): 是否进行数据增强
    """
    def __init__(self, path, args, aug):
        self.path = path
        self.args = args
        self.aug = aug
        
        # 初始化数据容器
        self.seq_pssm_nr = None
        self.seq_list_ori = []
        self.seq_list_aug = []
        self.all_labels = []
        self.all_swissprot = []
        self.set_len = []

    def _load_pssm_mapping(self):
        """加载PSSM文件映射关系"""
        with open(self.path + 'pssm_seq2fn.pkl', "rb") as f:
            self.seq_pssm_nr = pickle.load(f)

    def _process_partition(self, partition):
        """处理单个数据分片（train/val/test）
        
        Args:
            partition (str): 数据分片名称
        """
        # 加载序列文件
        self._load_sequence(partition)                

        # 加载标签文件
        self._load_labels(partition)
        
        # 加载PSSM/HHM特征（根据配置选择）
        pssm_hmm = getattr(self.args, 'pssm_hmm', None)
        if pssm_hmm is not None:
            pssm_hmm = str(pssm_hmm).strip().lower()
        if pssm_hmm in ('pssm', 'both'):
            self._load_pssms(partition)
        if pssm_hmm in ('hmm', 'both'):
            self._load_hhm(partition)

        # 处理数据增强
        if self.aug:
            self._data_augmentation(partition)


    def _load_sequence(self, partition):
        """加载多肽序列"""
        seqs, ids, attention_masks, lengths = [], [], [], []
        with open(os.path.join(self.path, partition, "seqs.fasta"), "r", encoding='utf-8') as f:
            for i, line in enumerate(f):
                if line[0] == ">": continue
                seq = line.strip()
                encoded_id, attention_mask = self.args.tokenizer.encode_plus(seq, padding=True)
                seqs.append(seq)
                ids.append(encoded_id)
                attention_masks.append(attention_mask)
                lengths.append(min(len(seq), self.args.max_length))
        setattr(self, f"{partition}_seqs", seqs)
        setattr(self, f"{partition}_ids", ids)
        setattr(self, f"{partition}_attention_masks", attention_masks)
        setattr(self, f"{partition}_lengths", lengths)

    def _load_labels(self, partition):
        """加载分类标签"""
        labels = []
        with open(os.path.join(self.path, partition, 'labels.csv')) as f:
            reader = csv.reader(f, delimiter=',', )
            for i, row in enumerate(reader):
                if i == 0: continue  # 跳过表头
                labels.append([int(j) for j in row])
        setattr(self, f"{partition}_labels", labels)

    def _load_pssms(self, partition):
        pssms = []
        lengths = []
        with open(os.path.join(self.path, partition, partition + '_pssm.csv'), "r", encoding='utf-8') as f:
            one_pssm = []
            one_length = 0
            for index, line in enumerate(f):
                if line[0] == '>':
                    if index == 0: continue
                    normalized_pssm, seq_length = self._normalize_pssm(one_pssm)
                    pssms.append(normalized_pssm)
                    lengths.append(seq_length)
                    one_pssm = []
                    one_length = 0
                    continue
                else:
                    tmp = line.strip().split(' ')
                    tmp = list(map(int, tmp))
                    one_pssm.append(tmp)
            normalized_pssm, seq_length = self._normalize_pssm(one_pssm)
            pssms.append(normalized_pssm)
            lengths.append(seq_length)
            setattr(self, f"{partition}_pssms", pssms)
            setattr(self, f"{partition}_lengths", lengths)

    def _normalize_pssm(self, one_pssm):
        # print(one_pssm)
        # handle empty or degenerate pssm (all values equal) safely
        if len(one_pssm) == 0:
            normalized_pssm = [[0.0] * 20 for _ in range(self.args.max_length)]
            seq_length = 0
            return normalized_pssm, seq_length

        max_num = max([max(i) for i in one_pssm])
        min_num = min([min(i) for i in one_pssm])
        denom = (max_num - min_num)
        if denom == 0:
            # avoid division by zero: if all values identical, map to zeros
            normalized_pssm = [[0.0 for _ in row] for row in one_pssm]
        else:
            normalized_pssm = [[(i - min_num) / denom for i in row] for row in one_pssm]
        seq_length = len(normalized_pssm)
        # print(np.array(normalized_pssm).shape, np.array(seq_length).shape)
        # print(seq_length)
        if self.args.max_length > len(normalized_pssm):
            normalized_pssm.extend([[0.0]*20 for i in range(self.args.max_length - len(normalized_pssm))])
        else:
            normalized_pssm = normalized_pssm[:self.args.max_length]
        return normalized_pssm, seq_length

    def _load_hhm(self, partition):
        """
        加载HHM特征。
        优先读取 data/hhm/{partition}/{Prefix}{idx}.hhm（Train/Val/Test），
        并兼容旧路径 self.path/{partition}/Train{idx}.hhm。
        """
        seqs = getattr(self, f"{partition}_seqs")
        hhm_list = []
        found_cnt = 0

        # allow override from args; default matches fusion2 old pipeline
        hhm_root = getattr(self.args, 'hhm_root', 'data/hhm')
        prefix_map = {'train': 'Train', 'val': 'Val', 'test': 'Test'}
        part_prefix = prefix_map.get(partition, partition.capitalize())

        for idx, seq in enumerate(seqs):
            # preferred path: data/hhm/{partition}/{Prefix}{idx}.hhm
            p1 = os.path.join(hhm_root, partition, f"{part_prefix}{idx}.hhm")
            # compatibility path used by some previous scripts
            p2 = os.path.join(hhm_root, partition, f"Train{idx}.hhm")
            # legacy fallback under ft_data
            p3 = os.path.join(self.path, partition, f"Train{idx}.hhm")
            hhm_path = p1 if os.path.exists(p1) else (p2 if os.path.exists(p2) else p3)

            if os.path.exists(hhm_path):
                hhm = get_hhm(hhm_path, self.args.max_length)
                found_cnt += 1
            else:
                hhm = np.zeros((self.args.max_length, 30), dtype=np.float32)
            hhm_list.append(hhm)
        setattr(self, f"{partition}_hhms", hhm_list)
        setattr(self, f"{partition}_hhm_found_count", found_cnt)
        

    def _data_augmentation(self, partition):
        """
        对训练集进行数据增强，并更新相关属性
        """
        if partition != 'train':
            return

        seqs = getattr(self, f"{partition}_seqs")
        ids = getattr(self, f"{partition}_ids")
        pssms = getattr(self, f"{partition}_pssms", None)
        hhms = getattr(self, f"{partition}_hhms", None)
        labels = getattr(self, f"{partition}_labels")
        lengths = getattr(self, f"{partition}_lengths")
        attention_masks = getattr(self, f"{partition}_attention_masks")

        # 只用replacement_alanine增强
        aug_seqs, aug_ids, aug_pssms, aug_hhms, aug_labels, aug_lengths, aug_attention_masks = [], [], [], [], [], [], []

        # 建立原始序列到pssm的映射
        seq2pssm = {seq: pssm for seq, pssm in zip(seqs, pssms)} if pssms is not None else {}
        seq2hhm = {seq: hhm for seq, hhm in zip(seqs, hhms)} if hhms is not None else {}

        # only needed when pssm features exist
        blosum = None
        if pssms is not None:
            with open('data/ft_data/blosum62.pkl', 'rb') as f:
                blosum = pickle.load(f)

        for seq, id_, label, length, attn_mask in zip(seqs, ids, labels, lengths, attention_masks):
            # replacement_alanine
            aug_seq = replacement_alanine(seq, 0.1)
            aug_id, aug_attn_mask = self.args.tokenizer.encode_plus(aug_seq, padding=True)
            # PSSM增强
            if pssms is not None:
                if aug_seq in seq2pssm:
                    aug_pssm = seq2pssm[aug_seq]
                else:
                    # 用blosum62生成新的pssm
                    aug_pssm = [blosum.get(aa, [0]*20)[:20] for aa in re.sub('[UZOB]', 'X', aug_seq)]
                    # 补齐或截断到max_length
                    if len(aug_pssm) < self.args.max_length:
                        aug_pssm.extend([[0]*20] * (self.args.max_length - len(aug_pssm)))
                    else:
                        aug_pssm = aug_pssm[:self.args.max_length]
            else:
                aug_pssm = None
            if hhms is not None:
                aug_hhm = seq2hhm.get(aug_seq, np.zeros((self.args.max_length, 30), dtype=np.float32).tolist())
            else:
                aug_hhm = None
            aug_length = min(len(aug_seq), self.args.max_length)
            aug_seqs.append(aug_seq)
            aug_ids.append(aug_id)
            if pssms is not None:
                aug_pssms.append(aug_pssm)
            if hhms is not None:
                aug_hhms.append(aug_hhm)
            aug_labels.append(label)
            aug_lengths.append(aug_length)
            aug_attention_masks.append(aug_attn_mask)

            # replacement_dict
            aug_seq = replacement_dict(seq, 0.1)
            aug_id, aug_attn_mask = self.args.tokenizer.encode_plus(aug_seq, padding=True)
            # PSSM增强
            if pssms is not None:
                if aug_seq in seq2pssm:
                    aug_pssm = seq2pssm[aug_seq]
                else:
                    # 用blosum62生成新的pssm
                    aug_pssm = [blosum.get(aa, [0]*20)[:20] for aa in re.sub('[UZOB]', 'X', aug_seq)]
                    # 补齐或截断到max_length
                    if len(aug_pssm) < self.args.max_length:
                        aug_pssm.extend([[0]*20] * (self.args.max_length - len(aug_pssm)))
                    else:
                        aug_pssm = aug_pssm[:self.args.max_length]
            else:
                aug_pssm = None
            if hhms is not None:
                aug_hhm = seq2hhm.get(aug_seq, np.zeros((self.args.max_length, 30), dtype=np.float32).tolist())
            else:
                aug_hhm = None
            aug_length = min(len(aug_seq), self.args.max_length)
            aug_seqs.append(aug_seq)
            aug_ids.append(aug_id)
            if pssms is not None:
                aug_pssms.append(aug_pssm)
            if hhms is not None:
                aug_hhms.append(aug_hhm)
            aug_labels.append(label)
            aug_lengths.append(aug_length)
            aug_attention_masks.append(aug_attn_mask)

        # 合并原始和增强数据
        all_seqs = seqs + aug_seqs
        all_ids = ids + aug_ids
        all_pssms = (pssms + aug_pssms) if pssms is not None else None
        all_hhms = (hhms + aug_hhms) if hhms is not None else None
        all_labels = labels + aug_labels
        all_lengths = lengths + aug_lengths
        all_attention_masks = attention_masks + aug_attention_masks

        setattr(self, f"{partition}_seqs", all_seqs)
        setattr(self, f"{partition}_ids", all_ids)
        if pssms is not None:
            setattr(self, f"{partition}_pssms", all_pssms)
        if hhms is not None:
            setattr(self, f"{partition}_hhms", all_hhms)
        setattr(self, f"{partition}_labels", all_labels)
        setattr(self, f"{partition}_lengths", all_lengths)
        setattr(self, f"{partition}_attention_masks", all_attention_masks)


    def form_ml_dataloader(self):
        """主处理流程入口"""
        # one-time fallback when HHM files are absent in this dataset
        hhm_mode_checked = False
        for partition in ['train', 'val', 'test']:
            self._process_partition(partition)
            if not hhm_mode_checked and self.args.pssm_hmm in ('hmm', 'both'):
                found = int(getattr(self, f"{partition}_hhm_found_count", 0))
                total = len(getattr(self, f"{partition}_seqs"))
                if found == 0:
                    old_mode = self.args.pssm_hmm
                    # no hhm files at all -> avoid adding all-zero 30 dims
                    self.args.pssm_hmm = 'pssm' if old_mode == 'both' else 'none'
                    print(f"[WARN] No HHM files detected under {os.path.join(self.path, partition)}; switch pssm_hmm: {old_mode} -> {self.args.pssm_hmm}")
                hhm_mode_checked = True
            # self._data_augmentation(partition)
            # 选择单独使用或者拼接
            # if self.use_hhm:
            #     features = getattr(self, f"{partition}_hhms")
            # else:
            #     features = getattr(self, f"{partition}_pssms")
            
            if self.args.pssm_hmm == 'pssm':
                features = np.asarray(getattr(self, f"{partition}_pssms"), dtype=np.float32)
            elif self.args.pssm_hmm == 'hmm':
                features = np.asarray(getattr(self, f"{partition}_hhms"), dtype=np.float32)
            elif self.args.pssm_hmm == 'none':
                num_samples = len(getattr(self, f"{partition}_seqs"))
                features = np.zeros((num_samples, self.args.max_length, 0), dtype=np.float32)  # shape (N, L, 0)
            elif self.args.pssm_hmm == 'both':
                pssm_arr = np.asarray(getattr(self, f"{partition}_pssms"), dtype=np.float32)
                hhm_arr = np.asarray(getattr(self, f"{partition}_hhms"), dtype=np.float32)
                # safety check: ensure same leading dims
                if pssm_arr.shape[0] != hhm_arr.shape[0] or pssm_arr.shape[1] != hhm_arr.shape[1]:
                    raise ValueError(f"PSSM and HHM shape mismatch: pssm={pssm_arr.shape}, hhm={hhm_arr.shape}")
                features = np.concatenate((pssm_arr, hhm_arr), axis=2)
            # ensure float32 and consistent shape
            features = np.asarray(features, dtype=np.float32)
            dataset = FTDataset(
                getattr(self, f"{partition}_seqs"),
                getattr(self, f"{partition}_ids"),
                features,  # 这里用features变量
                getattr(self, f"{partition}_labels"),
                getattr(self, f"{partition}_lengths"),
                getattr(self, f"{partition}_attention_masks"),
            )
            setattr(self, f"{partition}_dataset", dataset)
            dataloader = DataLoader(dataset, self.args.batch_size, shuffle=(partition == 'train'))
            setattr(self, f"{partition}_dataloader", dataloader)
              
        return self.train_dataset, self.train_dataloader, self.val_dataset, self.val_dataloader, self.test_dataset, self.test_dataloader

class PSSMProcesser:
    """
        PSSM矩阵处理和生成
    """
    def __init__(self, path):
        self.path = path
        # self.args = args

    def preprocess(self, sub_dir='test', out_fn='test_pssm.csv'):
        files = os.listdir(os.path.join(self.path, sub_dir))
        files.sort(key=lambda x:int(x.split('.')[0][len(sub_dir):]))
        all_pssm = []
        for fn in files:
            one_pssm = []
            nr = 0
            with open(os.path.join(self.path, sub_dir, fn)) as f1, \
            open(os.path.join(self.path, out_fn), 'a', encoding='utf-8') as f2:
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
                f2.write('>' + fn.split('.')[0] + '\n')
                for row in one_pssm:
                    f2.write(' '.join(row))
                    f2.write('\n')


def get_hhm(hhm_path, max_length):
    """
    读取单个hhm文件，返回shape为[max_length, 30]的numpy数组
    """
    directory = ['train', 'val', 'test']
    hhm = []
    with open(hhm_path, "r") as f:
        lines = f.readlines()
        start = False
        for line in lines:
            if line.startswith("#"):
                continue
            if line.startswith("HMM"):
                start = True
                continue
            if line.startswith("//"):
                break
            if start:
                parts = line.strip().split()
                if len(parts) >= 30:
                    row = []
                    for inf in parts[:30]:
                        # follow transform used in form_loader.get_hhm: map '*' or '0' -> 0, else pow(2, -0.0001*int(inf))
                        if inf == '*' or inf == '0':
                            row.append(0.0)
                        else:
                            try:
                                row.append(pow(2.0, (-0.0001 * int(inf))))
                            except Exception:
                                # fallback to 0.0 for unparsable tokens
                                row.append(0.0)
                    hhm.append(row)
            # 补齐或截断
            if len(hhm) < max_length:
                hhm.extend([[0.0]*30 for _ in range(max_length - len(hhm))])
            else:
                hhm = hhm[:max_length]

        # Convert to numpy and normalize per-sample to 0..1 to match PSSM scaling
        hhm = np.array(hhm, dtype=np.float32)
        if hhm.size == 0:
            return np.zeros((max_length, 30), dtype=np.float32)
        hhm_max = hhm.max()
        hhm_min = hhm.min()
        if hhm_max == hhm_min:
            # degenerate: all values same -> zeros
            hhm = np.zeros_like(hhm)
        else:
            hhm = (hhm - hhm_min) / (hhm_max - hhm_min)
        return hhm.astype(np.float32)
# pssmprocesser = PSSMProcesser('../data/new_pssm')
# pssmprocesser.preprocess(sub_dir='test', out_fn='test_pssm.csv')

    