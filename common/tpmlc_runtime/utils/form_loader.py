import numpy as np
import pandas as pd
import pickle
import csv
import os
import random
import torch
import math
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN, BorderlineSMOTE
import torch.utils
from torch.utils.data.dataloader import DataLoader
from utils.sampling import ImblancedSampling
from torch.utils.data.sampler import WeightedRandomSampler
from utils.data_augmentation import replacement_dict, replacement_alanine, global_random_shuffling, local_random_shuffling, sequence_revsersion, sequence_subsampling

def form_struc_loader(path, args, aug):
    pat = ['/train_npz', '/val_npz', '/test_npz']
    for pa in pat:
        files = os.listdir(path + pa)
        for index, f in enumerate(files):
            tmp = np.load(path + pa + "/" + f)
            dist, omega, phi, theta = tmp['dist'], tmp['omega'], tmp['phi'], tmp['theta']
            print(tmp.files)
            print(dist.shape, omega.shape, phi.shape, theta.shape)
            info = np.concatenate([dist, omega, phi, theta], axis=2)
            print(info.shape)
            if index == 4:
                exit(0)

def get_hhm(path, args):
    files = os.listdir('data/hhm')
    fn_hhm = {}
    for file in files:
        tmp_hhm = []
        flag = 0
        with open(os.path.join('data/hhm', file)) as f:
            for index, line in enumerate(f):
                if line[0] == '#':
                    flag = 1
                    begin = index + 5
                if flag == 1 and index == begin and line[:2] != '//':
                    info = line.strip().split('\t')
                    info[0] = info[0].split(' ')[1]
                    # for inf in info[:-1]:
                    #     if inf == '*' or inf == '0':
                    #         inf = 0
                    #     else:
                    #         inf = pow(2, (-0.0001*int(inf)))
                    info[:-1] = [0 if inf == '*' or inf == '0' else pow(2, (-0.0001*int(inf))) for inf in info[:-1] ]
                    tmp_hhm.append(info[:-1])
                    begin += 3
            tmp_hhm = np.array(tmp_hhm)
            if args.max_length-tmp_hhm.shape[0] > 0:
                tmp_hhm = np.append(tmp_hhm, [[0] * 20 for _ in range(args.max_length-tmp_hhm.shape[0])], axis=0)
            else:
                tmp_hhm = tmp_hhm[:args.max_length]
            # print(np.array(tmp_hhm).shape)
            fn = file.split('.')[0]
            fn_hhm[fn] = tmp_hhm
    return fn_hhm

def form_hhm_loader(path, args, aug=False):
    pat = ['/train', '/val', '/test']
    seq_list_ori, seq_list_aug, train_list, all_labels, all_swissprot = [], [], [], [], []
    set_len = []
    swissprot = []
    for pa in pat:
        # 加载序列文件
        train_labels, swissprot = [], []
        with open('data/ft_data' + pa + "/seqs.fasta", "r", encoding='utf-8') as f:
            for i, line in enumerate(f):
                if line[0] == ">": continue
                seq = line.strip()
                seq_list_ori.append(seq)
                seq_list_aug.append(seq)
                ids = args.tokenizer.encode(seq, padding=True)
                train_list.append(ids)
        # 加载标签文件
        with open('data/ft_data' + pa + '/labels.csv') as f:
            reader = csv.reader(f, delimiter=',', )
            for i, row in enumerate(reader):
                if i == 0:
                    header = row
                    continue
                train_labels.append([int(j) for j in row])

        if aug:
            train_labels = np.repeat(np.array(train_labels), 2, axis=0).tolist()
            swissprot = np.repeat(np.array(swissprot), 2, axis=0) # 把每个label, pssm复制?遍 两个为增强序列，一个是原序列
        all_labels.extend(train_labels)
        all_swissprot.extend(swissprot)
        # print(len(seq_list_aug), len(train_list), len(all_labels), len(all_swissprot))
        set_len.append(len(seq_list_aug)-sum(set_len))
    
    # print(len(all_iu), len(all_labels))
    fn_hhm = get_hhm('data/hhm', args)

    seqs, hhms, labels = train_list, all_scr, all_labels
    # print(len(ids), len(pssms), len(labels))

    train_len, val_len, test_len = round(4/5 * len(all_labels)), round(1/10 * len(all_labels)), round(1/10 * len(all_labels))
    
    train_seq, train_scrs, train_labels = list(seqs)[:train_len], list(scrs)[:train_len], list(labels)[:train_len]
    val_seq, val_scrs, val_labels = list(seqs)[train_len:train_len+val_len], list(scrs)[train_len:train_len+val_len], list(labels)[train_len:train_len+val_len]
    test_seq, test_scrs, test_labels = list(seqs)[train_len+val_len:], list(scrs)[train_len+val_len:], list(labels)[train_len+val_len:]

    train_dataset = SCRDataset(train_seq, train_scrs, train_labels)
    # print(len(train_dataset))
    train_dataloader = DataLoader(train_dataset, args.batch_size, shuffle = False)
        
    val_dataset = SCRDataset(val_seq, val_scrs, val_labels)
    val_dataloader = DataLoader(val_dataset, args.batch_size)
    test_dataset = SCRDataset(test_seq, test_scrs, test_labels)
    test_dataloader = DataLoader(test_dataset, args.batch_size)
    return train_dataset, train_dataloader, val_dataset, val_dataloader, test_dataset, test_dataloader

def form_iu_loader(path, args, aug=False):
    upper_path = ['/long', '/short', '/glob']
    iu_dict = {up:[] for up in upper_path}
    pat = ['/train', '/val', '/test']
    for up in upper_path:
        all_iu = []
        for pa in pat:
            files = os.listdir(path + up + pa)
            for index, file in enumerate(files):
                tmp_iu = []
                with open(os.path.join(path + up + pa, file)) as f:
                    for line in f:
                        if line[0].isdigit():
                            # print(line)
                            try:
                                pos, pep, value = line.strip().split('\t')
                                tmp_iu.append([float(value)])
                            except:
                                print(file)
                                exit(0)
                    if len(tmp_iu) < 256:
                        tmp_iu.extend([[0]] * (256-len(tmp_iu)))
                    else:
                        tmp_iu = tmp_iu[:256]
                    # print(np.array(tmp_iu).shape)
                all_iu.append(np.array(tmp_iu))
            iu_dict[up] = all_iu
    
    triple_dict = {}
    # for up, pa in iu_dict.items():
    #     # print(np.array(list(pa.values())).shape)
    #     print(up)
    #     triple_dict[up] = np.concatenate(list(pa.values()), axis=2)
    #     print(triple_dict[up].shape)
    all_iu = np.concatenate(list(iu_dict.values()), axis=2)
    print(all_iu.shape)
            

    pat = ['/train', '/val', '/test']
    seq_list_ori, seq_list_aug, train_list, all_labels, all_swissprot = [], [], [], [], []
    set_len = []
    swissprot = []
    fn_pssm_swissprot = {}
    for pa in pat:
        # 加载序列文件
        train_labels, swissprot = [], []
        # 加载标签文件
        with open('data/ft_data' + pa + '/labels.csv') as f:
            reader = csv.reader(f, delimiter=',', )
            for i, row in enumerate(reader):
                if i == 0:
                    header = row
                    continue
                train_labels.append([int(j) for j in row])

        if aug:
            train_labels = np.repeat(np.array(train_labels), 2, axis=0).tolist()
            swissprot = np.repeat(np.array(swissprot), 2, axis=0) # 把每个label, pssm复制?遍 两个为增强序列，一个是原序列
        all_labels.extend(train_labels)
        all_swissprot.extend(swissprot)
        # print(len(seq_list_aug), len(train_list), len(all_labels), len(all_swissprot))
        set_len.append(len(seq_list_aug)-sum(set_len))
    
    # print(len(all_iu), len(all_labels))

    ius, labels = all_iu, all_labels
    # print(len(ids), len(pssms), len(labels))

    train_len, val_len, test_len = round(4/5 * len(all_labels)), round(1/10 * len(all_labels)), round(1/10 * len(all_labels))
    
    train_ius, train_labels = list(ius)[:train_len], list(labels)[:train_len]
    val_ius, val_labels = list(ius)[train_len:train_len+val_len], list(labels)[train_len:train_len+val_len]
    test_ius, test_labels = list(ius)[train_len+val_len:], list(labels)[train_len+val_len:]

    train_dataset = IUDataset(train_ius, train_labels)
    # print(len(train_dataset))
    train_dataloader = DataLoader(train_dataset, args.batch_size, shuffle = False)
        
    val_dataset = IUDataset(val_ius, val_labels)
    val_dataloader = DataLoader(val_dataset, args.batch_size)
    test_dataset = IUDataset(test_ius, test_labels)
    test_dataloader = DataLoader(test_dataset, args.batch_size)
    return train_dataset, train_dataloader, val_dataset, val_dataloader, test_dataset, test_dataloader

def form_scr_loader(path, args, aug=False):
    all_scr = []
    scr_dict = {'C':1, 'H':2, 'E':3}
    pat = ['/train', '/val', '/test']
    for pa in pat:
        with open('data/scratch_output' + pa + '/' + pa[1:] + '.out.ss') as f:
            for index, line in enumerate(f):
                if line[0] != '>':
                    tmp_scr = [scr_dict[i] for i in line.strip()]
                    if len(tmp_scr) < 128:
                        tmp_scr += [0] * (128 - len(tmp_scr))
                    else:
                        tmp_scr = tmp_scr[:128]
                    all_scr.append(tmp_scr)
            

    pat = ['/train', '/val', '/test']
    seq_list_ori, seq_list_aug, train_list, all_labels, all_swissprot = [], [], [], [], []
    set_len = []
    swissprot = []
    fn_pssm_swissprot = {}
    for pa in pat:
        # 加载序列文件
        train_labels, swissprot = [], []
        with open('data/ft_data' + pa + "/seqs.fasta", "r", encoding='utf-8') as f:
            for i, line in enumerate(f):
                if line[0] == ">": continue
                seq = line.strip()
                seq_list_ori.append(seq)
                seq_list_aug.append(seq)
                ids = args.tokenizer.encode(seq, padding=True)
                train_list.append(ids)
        # 加载标签文件
        with open('data/ft_data' + pa + '/labels.csv') as f:
            reader = csv.reader(f, delimiter=',', )
            for i, row in enumerate(reader):
                if i == 0:
                    header = row
                    continue
                train_labels.append([int(j) for j in row])

        if aug:
            train_labels = np.repeat(np.array(train_labels), 2, axis=0).tolist()
            swissprot = np.repeat(np.array(swissprot), 2, axis=0) # 把每个label, pssm复制?遍 两个为增强序列，一个是原序列
        all_labels.extend(train_labels)
        all_swissprot.extend(swissprot)
        # print(len(seq_list_aug), len(train_list), len(all_labels), len(all_swissprot))
        set_len.append(len(seq_list_aug)-sum(set_len))
    
    # print(len(all_iu), len(all_labels))

    seqs, scrs, labels = train_list, all_scr, all_labels
    # print(len(ids), len(pssms), len(labels))

    train_len, val_len, test_len = round(4/5 * len(all_labels)), round(1/10 * len(all_labels)), round(1/10 * len(all_labels))
    
    train_seq, train_scrs, train_labels = list(seqs)[:train_len], list(scrs)[:train_len], list(labels)[:train_len]
    val_seq, val_scrs, val_labels = list(seqs)[train_len:train_len+val_len], list(scrs)[train_len:train_len+val_len], list(labels)[train_len:train_len+val_len]
    test_seq, test_scrs, test_labels = list(seqs)[train_len+val_len:], list(scrs)[train_len+val_len:], list(labels)[train_len+val_len:]

    train_dataset = SCRDataset(train_seq, train_scrs, train_labels)
    # print(len(train_dataset))
    train_dataloader = DataLoader(train_dataset, args.batch_size, shuffle = False)
        
    val_dataset = SCRDataset(val_seq, val_scrs, val_labels)
    val_dataloader = DataLoader(val_dataset, args.batch_size)
    test_dataset = SCRDataset(test_seq, test_scrs, test_labels)
    test_dataloader = DataLoader(test_dataset, args.batch_size)
    return train_dataset, train_dataloader, val_dataset, val_dataloader, test_dataset, test_dataloader


def form_ml_dataloader(path, args, tokenizer, aug):
    # 加载seq对应pssm文件的字典{seq: pssm_file1}
    with open(path + 'pssm_seq2fn.pkl', "rb") as f:
        seq_pssm_nr = pickle.load(f)
    
    # 加载所有sequence和label
    pat = ['/train', '/val', '/test']
    seq_list_ori, seq_list_aug, train_list, all_labels, all_swissprot = [], [], [], [], []
    set_len = []
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
                seq_list_aug.append(seq)
                ids = tokenizer.encode(seq, padding=True)
                train_list.append(ids)
                if aug:
                    # 数据增强
                    # 两种替换方法为随机替换
                    # seq_aug = replacement_alanine(seq=seq, p=0.1)
                    # seq_list_aug.append(seq_aug)
                    # ids = tokenizer.encode(seq_aug, padding=True)
                    # train_list.append(ids)
                    seq_aug = replacement_dict(seq=seq, p=0.1)
                    seq_list_aug.append(seq_aug)
                    ids = tokenizer.encode(seq_aug, padding=True)
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
        # fn_pssm = get_n_pssm(path + pa + '/pssm.csv', args)
        fn_pssm = get_hhm('data/hhm', args)
        # fn_pssm_swissprot.update(fn_pssm)
        swissprot = list(fn_pssm.values())
        # print(np.array(swissprot).shape)

        if aug:
            train_labels = np.repeat(np.array(train_labels), 2, axis=0).tolist()
            swissprot = np.repeat(np.array(swissprot), 2, axis=0) # 把每个label, pssm复制?遍 两个为增强序列，一个是原序列
        all_labels.extend(train_labels)
        all_swissprot.extend(swissprot)
        # print(len(seq_list_aug), len(train_list), len(all_labels), len(all_swissprot))
        set_len.append(len(seq_list_aug)-sum(set_len))
    
    seq2id = [list(i) for i in zip(seq_list_aug, train_list)]
    seq2label = [list(i) for i in zip(seq_list_aug, all_labels)]
    seq2pssm = [list(i) for i in zip(seq_list_aug, all_swissprot)]
    # print(len(seq2id), len(seq2label), len(seq2pssm))
    # print(len(seq_list_ori), len(seq_list_aug))  10237 30711
    
    # 加载处理好的pssm_nr文件
    fn_pssm_nr = get_n_pssm('./data/ft_data/pssm.csv', args)
    # fn_pssm_nr = get_hhm('data/hhm', args)
    
    # 加在blosum62字典用于作为进行数据增强的序列的PSSM矩阵
    with open('data/ft_data/blosum62.pkl', 'rb') as f:
        blosum = pickle.load(f)

    # 生成序列对应pssm矩阵的字典
    # print(list(seq2pssm.items())[:10])
    for index, seq_pssm in enumerate(seq2pssm):
        seq, pssm = seq_pssm[0], seq_pssm[1]
        # print(len(seq2pssm), len(fn_pssm_nr))
        # 9285 5222
        # seq2pssm: [['seq', array()], ['seq', array()]]
        # fn_pssm_nr: {file_name: array}
        # seq_pssm_nr: {seq: file_name}
        if seq in seq_pssm_nr:
            seq2pssm[index][1] = fn_pssm_nr[seq_pssm_nr[seq]]
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
            seq2pssm[index][1] = pssm_tmp
    ids, pssms, labels = [i[1] for i in seq2id], [i[1] for i in seq2pssm],[i[1] for i in seq2label]
    print(len(ids), len(pssms), len(labels))

    # ros = RandomOverSampler(random_state=123)
    # ros = BorderlineSMOTE(kind='borderline-2', n_jobs=-1, random_state=42)
    ros = ImblancedSampling(train_labels, 1/2)
    # ros = ADASYN(random_state=123)

    train_len, val_len, test_len = set_len[0], set_len[1], set_len[2]
    # print(set_len)
    # print(len(ids), train_len, val_len, test_len)
    train_ids, train_pssms, train_labels = list(ids)[:train_len], list(pssms)[:train_len], list(labels)[:train_len]
    val_ids, val_pssms, val_labels = list(ids)[train_len:train_len+val_len], list(pssms)[train_len:train_len+val_len], list(labels)[train_len:train_len+val_len]
    test_ids, test_pssms, test_labels = list(ids)[train_len+val_len:], list(pssms)[train_len+val_len:], list(labels)[train_len+val_len:]
    print(len(train_ids), len(val_ids), len(test_ids))
    print(len(train_pssms), len(val_pssms), len(test_pssms))
    exit(0)

    train_ids_pssms = list(zip(train_ids, train_pssms))

    train_dataset = FTDataset(train_ids, train_pssms, train_labels)
    # print(len(train_dataset))
    train_dataloader = DataLoader(train_dataset, args.batch_size, shuffle = False)
        
    val_dataset = FTDataset(val_ids, val_pssms, val_labels)
    val_dataloader = DataLoader(val_dataset, args.batch_size)
    test_dataset = FTDataset(test_ids, test_pssms, test_labels)
    test_dataloader = DataLoader(test_dataset, args.batch_size)
    return train_dataset, train_dataloader, val_dataset, val_dataloader, test_dataset, test_dataloader
    
        

def form_ft_loader(path, args, tokenizer):
    def get_ids_pssms(train_seqs): 
        train_ids = [seq2id[seq] for seq in train_seqs]
        train_pssms = [fn_pssm[seq_pssm_nr[seq]] for seq in train_seqs]
        return train_ids, train_pssms
    
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
        fn_pssm = get_n_pssm(path + pa + '/pssm.csv', args)
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
    print(len(train_ids), len(val_ids), len(test_ids))
    # print(len(train_pssms), len(val_pssms), len(test_pssms))
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

class FTDataset(torch.utils.data.dataset.Dataset):
    def __init__(self, seq, pssm, labels):
        self.seq = seq
        self.pssm = pssm
        self.labels = labels
    def __len__(self):
        return len(self.seq)
    def __getitem__(self, index):
        seq = torch.tensor(self.seq[index])
        pssm = torch.tensor(self.pssm[index])
        label = torch.tensor(self.labels[index])
        return seq, pssm, label
    
class IUDataset(torch.utils.data.dataset.Dataset):
    def __init__(self, iu, labels):
        self.iu = iu
        self.labels = labels
    def __len__(self):
        return len(self.iu)
    def __getitem__(self, index):
        iu = torch.tensor(self.iu[index])
        label = torch.tensor(self.labels[index])
        return iu, label

class SCRDataset(torch.utils.data.dataset.Dataset):
    def __init__(self, seq, scr, labels):
        self.seq = seq
        self.scr = scr
        self.labels = labels
    def __len__(self):
        return len(self.seq)
    def __getitem__(self, index):
        seq = torch.tensor(self.seq[index])
        scr = torch.tensor(self.scr[index])
        label = torch.tensor(self.labels[index])
        return seq, scr, label
    
def get_n_pssm(path, args):
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