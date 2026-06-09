import torch
import math
import pickle
import numpy as np
import csv
import os
import random
from utils.sampling import ImblancedSampling
from utils.data_augmentation import replacement_dict, replacement_alanine, global_random_shuffling, local_random_shuffling, sequence_revsersion, sequence_subsampling
from sklearn.model_selection import train_test_split
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN, BorderlineSMOTE

def form_ml_dataloader(path, args, tokenizer):
    with open(path + "/seqs.fasta", "r", encoding='utf-8') as f:
        train_list = []
        for i, line in enumerate(f):
            if line[0] == ">": continue
            seq = line.strip()
            ids = tokenizer.encode(seq, padding=True)
            train_list.append(ids)

    with open(path + '/labels.csv') as f:
        reader = csv.reader(f, delimiter=',', )
        train_labels = []
        for i, row in enumerate(reader):
            if i == 0:
                header = row
                continue
            train_labels.append([int(j) for j in row])
    
    # list(zip(train_list, train_labels))[:100]
    ros = RandomOverSampler(random_state=0)
    train_label_single = np.array([[train_labels[i][j] for i in range(len(train_labels))]  for j in range(15)])
    # print(train_label_single.shape)
    train_dataloaders = []
    for i in train_label_single:
        X_resample, y_resample = ros.fit_resample(train_list, i)
        # print(np.array(X_resample).shape, y_resample.shape)
        train_dataset = LSTMDataset(X_resample, y_resample)
        train_dataloaders.append(DataLoader(train_dataset, args.batch_size, shuffle = True))
    return train_dataset, train_dataloaders

class PeptideDataset(torch.utils.data.Dataset):
    def __init__(self, args, input_ids, attention_masks, mlm_positions, labels):
        self.args = args
        self.input_ids = input_ids
        self.attention_masks = attention_masks
        self.mlm_positions = mlm_positions
        self.labels = labels

        self.length = len(self.input_ids)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        input_id = self.input_ids[idx]
        attention_mask = self.attention_masks[idx]
        mlm_position = self.mlm_positions[idx]
        label = self.labels[idx]

        return {
            'input_ids': torch.tensor(input_id, dtype=torch.long).to(self.args.device),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long).to(self.args.device),
            'mlm_position': torch.tensor(mlm_position, dtype=torch.int64).to(self.args.device),
            'labels': torch.tensor(label, dtype=torch.int64).to(self.args.device)
        }
        
def read_dataset(path, tokenizer, seq_length):
    dataset, columns = [], {}
    with open(path, mode="r", encoding="utf-8") as f:
        for line_id, line in enumerate(f):
            if line_id == 0:
                for i, column_name in enumerate(line.strip().split("\t")):
                    columns[column_name] = i
                    # print(i, column_name)
                continue
            line = line[:-1].split("\t")
            label = line[columns["label"]]
            tgt = [np.float64(x) for x in label.split(",")]
            text_a = line[columns["text_a"]]
            # break
            
            enc_seq = tokenizer.tokenize(text_a, seq_length)
            src = enc_seq['input_ids']
            seg = enc_seq['attention_mask']
            if len(src) <= seq_length:
                dataset.append((src, tgt, seg))  
            else:
                dataset.append((src[:seq_length], tgt, seg[:seq_length]))  
    return dataset 
            
def batch_loader(batch_size, src, tgt, seg, soft_tgt=None):
    instances_num = src.size()[0]
    for i in range(instances_num // batch_size):
        src_batch = src[i * batch_size : (i + 1) * batch_size, :]
        tgt_batch = tgt[i * batch_size : (i + 1) * batch_size]
        seg_batch = seg[i * batch_size : (i + 1) * batch_size, :]
        if soft_tgt is not None:
            soft_tgt_batch = soft_tgt[i * batch_size : (i + 1) * batch_size, :]
            yield src_batch, tgt_batch, seg_batch, soft_tgt_batch
        else:
            yield src_batch, tgt_batch, seg_batch, None
    if instances_num > instances_num // batch_size * batch_size:
        src_batch = src[instances_num // batch_size * batch_size :, :]
        tgt_batch = tgt[instances_num // batch_size * batch_size :]
        seg_batch = seg[instances_num // batch_size * batch_size :, :]
        if soft_tgt is not None:
            soft_tgt_batch = soft_tgt[instances_num // batch_size * batch_size :, :]
            yield src_batch, tgt_batch, seg_batch, soft_tgt_batch
        else:
            yield src_batch, tgt_batch, seg_batch, None
            
def collate_fn(batch):
    data = {}
    data['input_ids'] = [i['input_ids'] for i in batch]
    data['attention_mask'] = [i['attention_mask'] for i in batch]
    data['mlm_position'] = [i['mlm_position'] for i in batch]
    data['labels'] = [i['labels'] for i in batch]
    return data   


def form_loader(path, args, tokenizer, sampler=True):
    with open(path + "/seqs.fasta", "r", encoding='utf-8') as f:
        train_list = []
        for i, line in enumerate(f):
            if line[0] == ">": continue
            seq = line.strip()
            ids = tokenizer.encode(seq, padding=True)
            train_list.append(ids)

    with open(path + '/labels.csv') as f:
        reader = csv.reader(f, delimiter=',', )
        train_labels = []
        for i, row in enumerate(reader):
            if i == 0:
                header = row
                continue
            train_labels.append([int(j) for j in row])

    # list(zip(train_list, train_labels))[:100]
    train_dataset = LSTMDataset(train_list, train_labels)
    weights = get_weights(train_dataset)[1]
    if sampler:
        train_dataloader = DataLoader(train_dataset, args.batch_size, sampler=WeightedRandomSampler(weights, len(weights), replacement=True))
    else:
        train_dataloader = DataLoader(train_dataset, args.batch_size)
    return train_dataset, train_dataloader

class LSTMDataset(torch.utils.data.dataset.Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = label
    def __len__(self):
        return len(self.data)
    def __getitem__(self, index):
        data = torch.tensor(self.data[index])
        label = torch.tensor(self.label[index])
        return data, label

class FTDataset(torch.utils.data.dataset.Dataset):
    def __init__(self, seq, id, pssm, labels, valid_len, attention_mask):
        self.seq = seq
        self.id = id
        self.pssm = pssm
        self.labels = labels
        self.valid_len = valid_len
        self.attention_mask = attention_mask
    def __len__(self):
        return len(self.seq)
    def __getitem__(self, index):
        seq = self.seq[index]
        id = torch.tensor(self.id[index])
        pssm = torch.tensor(self.pssm[index])
        label = torch.tensor(self.labels[index])
        valid_len = torch.tensor(self.valid_len[index])
        attention_mask = torch.tensor(self.attention_mask[index])
        return seq, id, pssm, label, valid_len, attention_mask
    
def get_weights(dataset):
    label_sum = {}
    weights = [0] * 15
    length = float(len(dataset))
    for data, label in dataset:
        for index, l in enumerate(label):
            if l == 1:
                weights[index] += 1
    # weights = {label: count/length for label, count in label_sum.items()}
    # print(weights)
    weights = [length/i for i in weights]
    sampler_weight = [1] * int(length)
    for i, data in enumerate(dataset):
        for index, l in enumerate(data[1]):
            if l == 1:
                sampler_weight[i] *= weights[index]
    # sampler_weight = [math.log(i) for i in sampler_weight]
    # print(weights, sampler_weight[:100])
    # exit(0)
    return torch.tensor(weights, dtype=torch.float), torch.tensor(sampler_weight)            

def make_weights_for_balanced_classes_split(dataset):
	N = float(len(dataset))   
	print(f"Number of samples: {N}")
    
	for c in range(len(dataset.slide_cls_ids)):
		print(f"Class {c} has {len(dataset.slide_cls_ids[c])} samples")
	weight_per_class = [N/len(dataset.slide_cls_ids[c]) for c in range(len(dataset.slide_cls_ids))]                                                                                                     
	weight = [0] * int(N)                                           
	for idx in range(len(dataset)):   
		y = dataset.getlabel(idx)                        
		weight[idx] = weight_per_class[y]                                  
	return torch.DoubleTensor(weight)

 

def form_ft_loader(path, args, tokenizer):
    def get_ids_pssms(train_seqs): 
        train_ids = [seq2id[seq] for seq in train_seqs]
        train_pssms = [fn_pssm[seq_pssm_nr[seq]] for seq in train_seqs]
        return train_ids, train_pssms
    
    def get_n_pssm(path):
        # 根据文件路径生成归一化pssm矩阵
        pssm, fn_pssm = [], {}
        with open(path, "r") as f:
            for i, line in enumerate(f):
                if i == 0: 
                    # pssm1
                    fn = line[1:].strip()
                    continue
                if line[0] == ">":
                    # 下一个pssm之前对上一个pssm矩阵进行归一化
                    pssm = np.array(pssm)
                    max_num = pssm.max()
                    min_num = pssm.min()
                    pssm = (pssm - min_num) / (max_num - min_num)
                    # 对超过的部分截断，对不足最大长度的部分补0
                    if args.max_length-pssm.shape[0] > 0:
                        pssm = np.append(pssm, [[0] * 20 for _ in range(args.max_length-pssm.shape[0])], axis=0)
                        # print(pssm)
                        # exit(0)
                    else:
                        pssm = pssm[:args.max_length]
                    # if pssm.shape != (128,20): print(pssm.shape)
                    fn_pssm[fn] = pssm
                    pssm = []
                    fn = line[1:].strip()
                    continue
                pssm.append([int(k) for k in line.strip().split(',')])
            # 最后一条序列的pssm矩阵
            pssm = np.array(pssm) 
            max_num = pssm.max()
            min_num = pssm.min()
            pssm = (pssm - min_num) / (max_num - min_num)
            if args.max_length-pssm.shape[0] > 0:
                pssm = np.append(pssm, [[0] * 20 for _ in range(args.max_length-pssm.shape[0])], axis=0)
            else:
                pssm = pssm[:args.max_length]
            fn_pssm[fn] = pssm
        return fn_pssm
    
    # 加载seq对应pssm文件的字典{seq: pssm_file1}
    with open(path + 'pssm_seq2fn.pkl', "rb") as f:
        seq_pssm_nr = pickle.load(f)
    
    # 加载所有sequence和label
    pat = ['/train', '/val', '/test']
    seq_list_ori, seq_list_aug, train_list, all_labels, all_swissprot = [], [], [], [], []
    swissprot = []
    fn_pssm_swissprot = {}
    for pa in pat:
        # 加载序列文件
        train_labels, swissprot = [], []
        with open(path + pa + "/seqs.fasta", "r", encoding='utf-8') as f:
            for i, line in enumerate(f):
                if line[0] == ">": continue
                seq = line.strip()
                seq_list_ori.append(seq)
                # 数据增强
                # 两种替换方法为随机替换
                # seq_aug = replacement_alanine(seq=seq, p=0.1)
                # seq_list_aug.append(seq_aug)
                # ids = tokenizer.encode(seq_aug, padding=True)
                # train_list.append(ids)
                # seq_aug = replacement_alanine(seq=seq, p=0.1)
                # seq_list_aug.append(seq_aug)
                # ids = tokenizer.encode(seq_aug, padding=True)
                # train_list.append(ids)
                
                # seq_aug = global_random_shuffling(seq=seq)
                # seq_aug = local_random_shuffling(seq=seq)
                # # seq_aug = sequence_revsersion(seq=seq)
                # seq_aug = sequence_subsampling(seq=seq)
                # seq_aug = seq
                seq_list_aug.append(seq)
                ids = tokenizer.encode(seq, padding=True)
                train_list.append(ids)
        # 加载标签文件
        with open(path + pa + '/labels.csv') as f:
            reader = csv.reader(f, delimiter=',', )
            for i, row in enumerate(reader):
                if i == 0:
                    header = row
                    continue
                train_labels.append([int(j) for j in row])
        # 加载pssm文件
        fn_pssm = get_n_pssm(path + pa + '/pssm.csv')
        # fn_pssm_swissprot.update(fn_pssm)
        swissprot = list(fn_pssm.values())
        # print(np.array(swissprot).shape)
        train_labels = np.repeat(np.array(train_labels), 3, axis=0).tolist()
        swissprot = np.repeat(np.array(swissprot), 3, axis=0) # 把每个label, pssm复制三遍 两个为增强序列，一个是原序列
        all_labels.extend(train_labels)
        all_swissprot.extend(swissprot)
        # print(len(train_list), len(all_labels), len(all_swissprot))
    
    seq2id = dict(zip(seq_list_aug, train_list))
    seq2label = dict(zip(seq_list_aug, all_labels))  
    seq2pssm = dict(zip(seq_list_aug, all_swissprot))
    # print(len(seq2id), len(seq2label), len(seq2pssm))
    # print(len(seq_list_ori), len(seq_list_aug))  10237 30711
    
    # 加载处理好的pssm_nr文件
    fn_pssm_nr = get_n_pssm('./data/ft_data/pssm.csv')
        
    # 按照seq对应pssm文件的字典进行切分
    # seqs, ids, pssms, labels = [], [], [], []
    # for seq, pssm_file in seq_pssm_nr.items():
    #     if seq in seq2label:
    #         seqs.append(seq)
    #         ids.append(seq2id[seq])
    #         # pssms.append(fn[pssm_file])
    #         labels.append(seq2label[seq])
    # print(len(seqs), len(labels))
    
    # 加在blosum62字典用于作为进行数据增强的序列的PSSM矩阵
    with open('data/ft_data/blosum62.pkl', 'rb') as f:
        blosum = pickle.load(f)
        
    # 生成序列对应pssm矩阵的字典
    # print(list(seq2pssm.items())[:10])
    for seq, pssm in seq2pssm.items():
        if seq in seq_pssm_nr:
            seq2pssm[seq] = fn_pssm_nr[seq_pssm_nr[seq]]
        elif seq not in seq_list_ori:
            pssm_tmp = np.array([])
            for s in seq:
                pssm_tmp = np.append(pssm_tmp, blosum[s])
            max_num = pssm_tmp.max()
            min_num = pssm_tmp.min()
            pssm_tmp = (pssm - min_num) / (max_num - min_num)
            if args.max_length-pssm.shape[0] > 0:
                pssm_tmp = np.append(pssm_tmp, [[0] * 20 for _ in range(args.max_length-pssm.shape[0])], axis=0)
            else:
                pssm = pssm[:args.max_length]
            seq2pssm[seq] = pssm_tmp
                
    
    random.seed(123)
    # shuffler = random.shuffle(list(range(10234)))
    # # print(list(seq2id.values())[:5], list(seq2pssm.values())[:5], list(seq2label.values())[:5])
    # ids = [list(seq2id.values())[i] for i in shuffler]
    # pssms = [list(seq2pssm.values())[i] for i in shuffler]
    # labels = [list(seq2label.values())[i] for i in shuffler]
    ids, pssms, labels = list(seq2id.values()), list(seq2pssm.values()), list(seq2label.values())
    # print(len(ids), len(pssms), len(labels))

    train_len, val_len, test_len = round(4/5 * len(ids)), round(1/10 * len(ids)), round(1/10 * len(ids))
    # print(len(ids), train_len, val_len, test_len)
    train_ids, train_pssms, train_labels = list(ids)[:train_len], list(pssms)[:train_len], list(labels)[:train_len]
    val_ids, val_pssms, val_labels = list(ids)[train_len:train_len+val_len], list(pssms)[train_len:train_len+val_len], list(labels)[train_len:train_len+val_len]
    test_ids, test_pssms, test_labels = list(ids)[train_len+val_len:], list(pssms)[train_len+val_len:], list(labels)[train_len+val_len:]
    # print(len(train_ids), len(val_ids), len(test_ids))
    # exit(0)

    # with open('data/ft_data/train.csv', "w", encoding='utf-8') as f:
    #     writer = csv.writer(f)
    #     for index, ids in enumerate(train_ids):
    #         f.write(f'>seq{index}\n')
    #         f.write(">>id:\n")
    #         print(ids)
    #         writer.writerow(ids)
    #         f.write(">>pssm:\n")
    #         for pssm in train_pssms[index]:
    #             writer.writerow(pssm)
    #         f.write(">>label:\n")
    #         writer.writerow(train_labels[index])
    # exit(0)            
            
    
    # train_seqs, tmp_seqs, train_labels, tmp_labels = train_test_split(seqs, labels, test_size=0.2, random_state=123)
    # val_seqs, test_seqs, val_labels, test_labels = train_test_split(tmp_seqs, tmp_labels, test_size=0.5, random_state=123)
    
    # train_ids, train_pssms = get_ids_pssms(train_seqs)
    # # print(np.array(train_ids).shape, np.array(train_pssms).shape)
    # val_ids, val_pssms = get_ids_pssms(val_seqs)
    # test_ids, test_pssms = get_ids_pssms(test_seqs)
    train_ids_pssms = list(zip(train_ids, train_pssms))
    
    # ros = RandomOverSampler(random_state=123)
    # ros = BorderlineSMOTE(kind='borderline-2', n_jobs=-1, random_state=42)
    ros = ImblancedSampling(train_labels, 1/2)
    # ros = ADASYN(random_state=123)
    train_label_single = np.array([[train_labels[i][j] for i in range(len(train_labels))]  for j in range(15)])
    # print(train_label_single.shape)
    train_dataloaders = []
    for i in train_label_single:
        try:
            X_resample, y_resample = ros.fit_resample(train_ids_pssms, i)
        except:
            X_resample, y_resample = train_ids_pssms, i
        # print(np.array(X_resample).shape, y_resample.shape)
        train_ids = [x[0] for x in X_resample]
        train_pssms = [x[1] for x in X_resample]
        train_dataset = FTDataset(train_ids, train_pssms, y_resample)
        # print(len(train_dataset))
        train_dataloaders.append(DataLoader(train_dataset, args.batch_size, shuffle = True))
        
    val_dataset = FTDataset(val_ids, val_pssms, val_labels)
    val_dataloader = DataLoader(val_dataset, args.batch_size)
    test_dataset = FTDataset(test_ids, test_pssms, test_labels)
    test_dataloader = DataLoader(test_dataset, args.batch_size)
    return train_dataset, train_dataloaders, val_dataset, val_dataloader, test_dataset, test_dataloader

