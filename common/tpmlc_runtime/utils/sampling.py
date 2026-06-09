import torch
import math
import random

class ImblancedSampling():
    def __init__(self, labels, q) -> None:
        self.labels = labels
        self.q = q
        
    def get_weights(self):
        if self.q == 1/2:
            self.weights = self.get_square_sampling_weight()
        
    def get_square_sampling_weight(self):
        count = [0] * 15
        for label in self.labels:
            for index, l in enumerate(label):
                if l == 1:
                    count[index] += 1
        
        # 平方根采样
        tmp = [math.sqrt(i) for i in count]          
        prob = [i / sum(tmp) for i in tmp]
        
        weights = [0] * len(self.labels)
        for i, label in enumerate(self.labels):
            for index, l in enumerate(label):
                if l == 1:
                    weights[i] *= prob[index]
        weights = [1 / w for w in weights]
        return weights
    
    def fit_resample(self, id_pssm, single_label):
        self.get_weights()
        random.seed(123)
        id_pssm_label = [(id_pssm[i][0], id_pssm[i][1], single_label[i]) for i in range(len(id_pssm))]
        sampled = random.choices(id_pssm_label, self.weights, k=len(id_pssm))
        sampled_id_pssm = [(i[0], i[1]) for i in sampled]
        sampled_label = [i[2] for i in sampled]
        return sampled_id_pssm, sampled_label