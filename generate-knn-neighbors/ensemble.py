# -*- coding: utf-8 -*-
from __future__ import print_function
import os
from PIL import Image
import pandas as pd
import logging
import random
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
import numpy as np
import torch.nn as nn
# from utils import *
# os.environ["CUDA_VISIBLE_DEVICES"] = '2,3'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

"""Seed and GPU setting"""
seed = 109
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
torch.cuda.manual_seed(seed)

cudnn.benchmark = True
cudnn.deterministic = True

class EnsembleModel(nn.Module):
    def __init__(self, class_num, n):
        super(EnsembleModel, self).__init__()
        self.class_num = class_num
        self.w = nn.Parameter(
            torch.empty((n, 1), dtype=torch.float64, requires_grad=True)
        )
        torch.nn.init.uniform_(self.w, 0, 1)
    
    def forward(self, inputs):
        x = torch.matmul(inputs, self.w)
        #print(self.w)
        result = torch.sum(x, 2).reshape((-1, self.class_num))
        
        return result
    
def train(model, inputs, labels, test_inputs, test_labels, epochs = 100, lr = 0.01):
    inputs = torch.tensor(inputs, dtype=torch.float64)
    labels = torch.tensor(labels, dtype=torch.long)
    CELoss = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
    data_len = inputs.shape[0]
    bs = 64
    batch_num = data_len // bs + (1 if data_len % bs else 0)
    suffle_data = True
    for epoch in range(epochs):
        model.train()
        if suffle_data:
            index = np.random.permutation(np.arange(data_len))
        else:
            index = np.arange(data_len)
        acc = 0.
        for i in range(batch_num):
            input_batch = inputs[index][bs * i: bs * i + bs].to(device)
            label_batch = labels[index][bs * i: bs * i + bs].to(device)
            output = model(input_batch)
            # _, pred = torch.max(output, 1)
            optimizer.zero_grad()
            loss = CELoss(output, label_batch)
            loss.backward()
            optimizer.step()
        scheduler.step()
        demo(model, inputs, labels)
        demo(model, test_inputs, test_labels)
            
def demo(model, inputs, labels):
    inputs = torch.tensor(inputs, dtype=torch.float64)
    labels = torch.tensor(labels, dtype=torch.long)
    data_len = inputs.shape[0]
    bs = 64
    batch_num = data_len // bs + (1 if data_len % bs else 0)
    model.eval()
    acc = 0.
    right = 0.
    all = 0.
    with torch.no_grad():
        for i in range(batch_num):
            input_batch = inputs[bs * i: bs * i + bs].to(device)
            label_batch = labels[bs * i: bs * i + bs].to(device)
            output = model(input_batch)
            _, pred = torch.max(output.data, 1)
            all += output.shape[0]
            right += pred.eq(label_batch.data).cpu().sum()
        print("{:.4f} / {:.4f} = {:.4f}".format(right, all, right / all))



# val_paths = ['/mnt/hdd0/data/liyajuan/ltw/PMG/datasets/cars_t10/val_softmax.npy', '/mnt/hdd0/data/liyajuan/ltw/PMG/datasets/cars_t10/r50_val.npy', '/mnt/hdd0/data/liyajuan/ltw/PMG/datasets/cars_t10/r101_val.npy', '/mnt/hdd0/data/liyajuan/ltw/PMG/datasets/cars_t10/r152_val.npy']
# test_paths = ['/mnt/hdd0/data/liyajuan/ltw/PMG/datasets/cars_t10/test_softmax.npy', '/mnt/hdd0/data/liyajuan/ltw/PMG/datasets/cars_t10/r50_test.npy', '/mnt/hdd0/data/liyajuan/ltw/PMG/datasets/cars_t10/r101_test.npy', '/mnt/hdd0/data/liyajuan/ltw/PMG/datasets/cars_t10/r152_test.npy',]
# val_label_path = '/mnt/hdd0/data/liyajuan/ltw/PMG/datasets/cars_t10/val_true.npy'
# test_label_path = '/mnt/hdd0/data/liyajuan/ltw/PMG/datasets/cars_t10/test_true.npy'


data_file = {
    "2013tw":{
        "val_label_path": [r'G:\DNN提取规则\2013tw\paths_dev.csv'],
        "test_label_path": [r'G:\DNN提取规则\2013tw\paths_test.csv'],
        "val_paths":["G:\DNN提取规则\\2013tw\\roberta-base\prob_file\logits_dev_prediction.csv.npy",],
        "test_paths":["G:\DNN提取规则\\2013tw\\roberta-base\prob_file\logits_test_prediction.csv.npy"],

    },
    # "2016tw":{
    #     "val_paths":["G:\DNN提取规则\\2016tw\\roberta-base\prob_file\logits_dev_prediction.csv.npy",],
    #     "test_paths": ["G:\DNN提取规则\\2016tw\\xlnet-base-cased\prob_file\logits_test_prediction.csv.npy"],
    #     "val_label_path": [r'G:\DNN提取规则\2013tw\paths_dev.csv'],
    #     "test_label_path": [r'G:\DNN提取规则\2013tw\paths_test.csv']
    # },
    # "sst":{
    #     "val_paths": ["G:\DNN提取规则\\sst\\roberta-base\prob_file\logits_dev_prediction.csv.npy"],
    #     "test_paths": ["G:\DNN提取规则\\sst\\xlnet-base-cased\prob_file\logits_test_prediction.csv.npy"],
    #     "val_label_path": [r'G:\DNN提取规则\2013tw\paths_dev.csv'],
    #     "test_label_path": [r'G:\DNN提取规则\2013tw\paths_test.csv']
    # }
}

for data_name, data_dict in data_file.items():
    class_num = 2
    c = class_num
    print('load model')
    model = EnsembleModel(class_num, 2)
    model.to(device)
    print('model success')
    for data_type, path_list in data_dict.items():
        if data_type =="val_label_path":
            val_labels = pd.read_csv(path_list[0]).values[:]
            val_n = len(val_labels)
            print(val_labels)
        if data_type =="test_label_path":
            test_labels = pd.read_csv(path_list[0], encoding="utf-8").values[:]
            test_n = len(test_labels)
            print(test_labels)
        if data_type == "val_paths":
            m = len(path_list)
            val_inputs = np.zeros([val_n, c])
            for path in path_list:
                inp = np.load(path).reshape([val_n, c])
                val_inputs = np.concatenate((val_inputs, inp), axis=1)
        if data_type == "test_paths":
            test_inputs = np.zeros([test_n, c])
            for path in path_list:
                inp = np.load(path).reshape([test_n, c])
                test_inputs = np.concatenate((test_inputs, inp), axis=1)

    train(model, val_inputs[:, :], val_labels, test_inputs[:, :], test_labels)
    demo(model, test_inputs[:, :], test_labels)



