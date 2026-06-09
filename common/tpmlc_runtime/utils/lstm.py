import torch
import csv
import torch.nn as nn
import numpy as np
import pandas as pd

from sklearn.metrics import hamming_loss
from torch.autograd import Variable
from torch.nn.init import xavier_normal
from torch.utils.data.dataloader import DataLoader

class NET(nn.Module):
    def __init__(self, encoder, multi_label_classifier) -> None:
        super(NET, self).__init__()
        self.encoder = encoder
        self.classifier = multi_label_classifier
        
    def forward(self, x):
        encoded_X = self.encoder(x)
        output = self.classifier(encoded_X)
        return output


def conv(batch_norm, c_in, c_out, ks=3, sd=1, pad=0):
    if batch_norm:
        return nn.Sequential(
            nn.Conv2d(c_in, c_out, kernel_size=ks, stride=sd, padding=(ks-1)//2, bias=False),
            nn.BatchNorm2d(c_out),
            nn.ReLU(),
        )
    else:
        return nn.Sequential(
            # nn.Conv2d(c_in, c_out, kernel_size=ks, stride=sd, padding=(ks-1)//2, bias=True),
            nn.Conv2d(c_in, c_out, kernel_size=ks, stride=sd, padding=pad, bias=True),
            nn.ReLU(),
        )


def fc(c_in, c_out, activation=None):
    if activation == 'relu':
        return nn.Sequential(
            nn.Linear(c_in, c_out),
            nn.ReLU(),
        )
    elif activation == 'sigmoid':
        return nn.Sequential(
            nn.Linear(c_in, c_out),
            nn.Sigmoid(),
        )
    else:
        return nn.Linear(c_in, c_out)
    
class LSTM_ML2(nn.Module):
    def __init__(self, in_features, hidden_size, layer_num, phase='Train', batch_norm=False) -> None:
        super(LSTM_ML2, self).__init__()
        self.fcs = nn.ModuleList([
            LSTM_ML(in_features, hidden_size, layer_num, phase, batch_norm)
            for _ in range(15)
        ])
    def forward(self, x):
        outputs = []
        for i, fc in enumerate(self.fcs):          # N x D2
            output = fc(x)    
            outputs.append(output)
        x = torch.cat(outputs, dim=-1)
        return x


class LSTM_ML(nn.Module):
    def __init__(self, in_features, hidden_size, layer_num, multi_lstm=False, phase='Train', batch_norm=False):
        super(LSTM_ML, self).__init__()
        self.phase = phase
        self.multi_lstm = multi_lstm
        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)
        self.batch_norm = batch_norm
        self.conv1 = conv(self.batch_norm, 1, 256, ks=[3, in_features], pad=0)
        self.lstm = nn.LSTM(input_size=in_features,
                            hidden_size=hidden_size,
                            num_layers=layer_num,
                            batch_first=True,
                            dropout=0.5,
                            bidirectional=True)
        self.gru = nn.GRU(input_size=in_features,
                          hidden_size=hidden_size,
                          num_layers=layer_num,
                          batch_first=True,
                          dropout=0.5,
                          bidirectional=True)
        self.fc1 = fc(hidden_size*2, 128, activation='relu')
        self.fc2 = fc(128, 15)
        self.fc3 = fc(128,1)
        self.sigmoid = nn.Sigmoid()
        self.fcs = nn.ModuleList([nn.Sequential(
                fc(hidden_size*2, 256, activation='relu'),
                nn.Linear(256, 1),
                # nn.Sigmoid()
            )
            for _ in range(15)
        ])
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_normal(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        # N, T, D1 = tuple(x.size())
        x = self.dropout1(x)
        # # x, _ = self.lstm(x)                 # N x T x D2
        # x, _ = self.gru(x)                  # N x T x D2
        # x = x[:, -1, :]                     # N x 1 x D2 (last time step)
        # x = x.view(N, -1)                   # N x D2
        
        N, T, D1 = tuple(x.size())
        x, _ = self.lstm(x)                 # N x T x D2
        x = torch.sum(x, 1) / T             # N x D2
        # x = self.dropout2(x)
        # N, T, D1 = tuple(x.size())
        # x = x.view(N, 1, T, D1)
        # x = self.conv1(x)                   # N x 256 x T x 1
        # x = torch.transpose(x, 2, 1)        # N x T x 256 x 1
        # x = x.contiguous()
        # x = x.view(N, -1, 256)              # N x T x 256
        # x, _ = self.lstm(x)                 # N x T x D2
        # x = x[:, -1, :]                     # N x 1 x D2 (last time step)
        # x = x.view(N, -1)                   # N x D2
        
        # 分类
        if self.multi_lstm:
            x = self.fc1(x)
            x = self.fc3(x)
        elif not self.multi_lstm:
            x = self.fc1(x)
            x = self.fc2(x)
        # outputs = []
        # for i, fc in enumerate(self.fcs):          # N x D2
        #     output = fc(x)    
        #     outputs.append(output)
        # x = torch.cat(outputs, dim=-1)

        # if self.phase == 'Train':
        return x
        # else:
        # return self.sigmoid(x)

    def weight_parameters(self):
        return [param for name, param in self.named_parameters() if 'weight' in name]

    def bias_parameters(self):
        return [param for name, param in self.named_parameters() if 'bias' in name]
    

def instances_overall_metrics(y_pred: np.array, y_true: np.array, threshold=0.5, save = None, show = True):
    """
    计算样本层面的整体评价指标
    """
    y_pred_cls = np.zeros_like(y_pred, dtype=np.int)
    y_pred_cls[y_pred > threshold] = 1    # 预测类别

    n, m = y_true.shape

    # Hamming Loss
    HLoss = hamming_loss(y_true, y_pred_cls)

    # Accuracy
    ACC = 0
    for i in range(n):
        ACC += (np.sum((y_pred_cls[i] == 1) & (y_true[i] == 1)) / np.sum((y_pred_cls[i] == 1) | (y_true[i] == 1)))
    ACC /= n

    # Precision
    Precision = 0
    for i in range(n):
        if (np.sum((y_pred_cls[i] == 1) & (y_true[i] == 1)) == 0): continue
        Precision += (np.sum((y_pred_cls[i] == 1) & (y_true[i] == 1)) / np.sum(y_pred_cls[i] == 1) )
    Precision /= n

    # Recall
    Recall = 0
    for i in range(n):
        Recall += (np.sum((y_pred_cls[i] == 1) & (y_true[i] == 1)) / np.sum(y_true[i] == 1))
    Recall /= n

    # Absolute ture
    AT = 0
    for i in range(n):
        if(np.all(y_pred_cls[i] == y_true[i])):
            AT += 1
    AT /= n

    df = pd.DataFrame({'HLoss': [HLoss], 'Accuracy': [ACC], 'Precision': [Precision], 'Recall': [Recall], 'Absolute true': [AT]})
    if show:
        print(df)

    if save is not None:
        df.to_csv(save)

    return df
        

def main():
    net = LSTM_ML(in_features=300, hidden_size=64, layer_num=2)
    print(net)
    while True:
        input = Variable(torch.randn(32, 250, 300))
        output = net(input)
        print(output.size())


if __name__ == '__main__':
    main()