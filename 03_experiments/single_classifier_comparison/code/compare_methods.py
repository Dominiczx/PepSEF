import argparse
import torch
import yaml
import os
import json
import random
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import pickle
import matplotlib.pyplot as plt

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
from utils.bert_aff import ESM2_AFF_LSTM

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

parser = argparse.ArgumentParser()

# model path, vocab path, special token path, train data path
parser.add_argument('--model_path', type=str, default="model/t5-base/pytorch_model.bin")
parser.add_argument('--vocab_path', type=str, default="utils/uniprot_1kmer_vocab.txt")
parser.add_argument('--special_token_path', type=str, default="utils/special_tokens.json")
parser.add_argument('--pretrained_data_path', type=str, default="data/ft_data/train/seqs.fasta")
parser.add_argument('--save_path', type=str, default="model/1027/normal.bin")

parser.add_argument('-e', '--epochs', type=int, default=500)
parser.add_argument('-bs', '--batch_size', type=int, default=128)
parser.add_argument('-lr', '--learning_rate', type=float, default=5e-5)
parser.add_argument('-l', '--max_length', type=int, default=128)
parser.add_argument('--pssm_hmm', type=str, default='hmm', choices=['pssm', 'hmm', 'both', 'none'])
parser.add_argument('--pssm_dropout', type=float, default=0.1)
parser.add_argument('--fusion_alpha_init', type=float, default=-1.2)


args = parser.parse_args()
set_seed(123)

config_file = open('model/bert_model/config.yaml', "r", encoding='utf-8')
config_data = config_file.read()
bert_config = yaml.load(config_data, Loader=yaml.CLoader)
config_file.close()
# print(config1)

config_file = open('model/esm2/config.json', "r", encoding='utf-8')
esm2_config = json.load(config_file)
config_file.close()

# config_file = open('model/protbert/config.json', "r", encoding='utf-8')
# protbert_config = json.load(config_file)
# config_file.close()

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
args.save = False
tokenizer = PeptideTokenizer(args)
args.tokenizer = tokenizer
args.vocab_size = tokenizer.vocab_size
# my model
# Map CUDA-trained checkpoints to the active device so the comparison can run
# on CPU when both GPUs are already occupied by full training jobs.
checkpoint = torch.load('./output/model/39_hhm.bin', map_location=device)
model = Bert_AFF_LSTM(args, bert_config, aff_config, lstm_config, multi_lstm=False).to(device)
# The 39_hhm checkpoint was saved with the older three-linear-layer MLP
# without LayerNorm modules, so restore that head only for this comparison.
if 'mlp.2.weight' in checkpoint and 'mlp.1.weight' not in checkpoint:
    model.mlp = nn.Sequential(
        nn.Linear(256, 1024),
        nn.LeakyReLU(),
        nn.Linear(1024, 512),
        nn.LeakyReLU(),
        nn.Linear(512, 15),
    ).to(device)
model.load_state_dict(checkpoint)
#esm2 model
# model = ESM2_AFF_LSTM(args, esm2_config, aff_config, lstm_config, multi_lstm=False).to(device)

criterion = FocalDiceLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
path = 'data/ft_data/'
pdp = PeptideDataProcessor(path, args, aug=False)
train_dataset, train_dataloader, val_dataset, val_dataloader, test_dataset, test_dataloader = pdp.form_ml_dataloader()

with open('data/ft_data/blosum62.pkl', 'rb') as f:
    blosum = pickle.load(f)



val_acc, val_pre, val_rec, val_f1 = validate(args, val_dataloader, model, phase='val')
test_acc, test_pre, test_rec, test_f1 = validate(args, test_dataloader, model, phase='test')
print(f'val_acc:{val_acc}, val_pre:{val_pre}, val_rec:{val_rec}, val_f1:{val_f1}')
print(f'test_acc:{test_acc}, test_pre:{test_pre}, test_rec:{test_rec}, test_f1:{test_f1}')
