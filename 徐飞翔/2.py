# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import torch
from torch import nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms

EPOCH = 40
BATCH_SIZE = 16
TIME_STEP = 5
INPUT_SIZE = 8
LR = 0.001
DATASIZE = 427700


def csv2train(filename):
    df = pd.read_csv(filename, index_col=0)
    groups = df.groupby('Date')

    train_data = []
    print("loading data...")
    for date,group in groups:

        group["VolDiff"] = group["Volume"] - group["Volume"].shift(1)
        group_new = group.fillna(0)

        dataset = np.array(group_new.loc[:,['MidPrice', 'LastPrice', 'Volume',
        	'BidPrice1','BidVolume1', 'AskPrice1', 'AskVolume1','VolDiff']])

        #nomalization
        dataset.astype(np.float32)
        mean = dataset.mean(axis=0)
        std = dataset.std(axis=0)

        dataset = (dataset - mean)/std

        size = len(dataset)-30
        size = size - (size%BATCH_SIZE)
        for i in range(0,size,1):
            train_data.append((dataset[i:i+10],dataset[i+10:i+30,0].mean()))

    return train_data,0,1

def csv2test(filename):
    df = pd.read_csv(filename, index_col=0)
    groups = df.groupby('Date')

    test_data = []
    test_data_means = []
    test_data_stds = []
    for date, group in groups:

        group["VolDiff"] = group["Volume"] - group["Volume"].shift(1)
        group_new = group.fillna(0)

        dataset = np.array(group_new.loc[:, ['MidPrice', 'LastPrice', 'Volume',
        	'BidPrice1', 'BidVolume1', 'AskPrice1', 'AskVolume1', 'VolDiff']])

        dataset.astype(np.float32)
        dataset.astype(np.float32)
        mean = dataset.mean(axis=0)
        std = dataset.std(axis=0)

        dataset = (dataset - mean)/std

        mean_midprice = mean[0]
        std_midprice = std[0]

        size = len(dataset)
        for i in range(0, size, 10):
            test_data.append(dataset[i:i+10])
            test_data_means.append(mean_midprice)
            test_data_stds.append(std_midprice)

    return test_data, test_data_means,test_data_stds

def res2csv(dataset):
    fout = open("result.csv","w")
    fout.write("caseid,midprice\n")
    for i in range(len(dataset)):
        if(i<142):
            continue
        fout.write("%d,%f\n"%(i+1,dataset[i]))
    fout.close()

class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()

        self.rnn = nn.LSTM(         # if use nn.RNN(), it hardly learns
            input_size=INPUT_SIZE,
            hidden_size=64,         # rnn hidden unit
            num_layers=2,           # number of rnn layer
            # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
            batch_first=True,
        )

        self.out = nn.Linear(64, 1)

    def forward(self, x):
        r_out, (h_n, h_c) = self.rnn(x, None)

        out = self.out(r_out[:, -1, :])
        return out

train_data,train_mean,train_std = csv2train("train_data.csv")


train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=False)

rnn = RNN()
#print(rnn)

# optimize all cnn parameters
optimizer = torch.optim.Adam(rnn.parameters(), lr=0.00001,betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
# the target label is not one-hotted
loss_func = nn.MSELoss()

# training and testing
for epoch in range(EPOCH):
    for step, (b_x, b_y) in enumerate(train_loader):
        b_x = b_x.view(-1, 10, INPUT_SIZE).float()
        #print(b_x)
        b_y = b_y.float()

        output = rnn(b_x)
        loss = loss_func(output.view(BATCH_SIZE), b_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step%50 == 0:
            print('Epoch: ', epoch, ' loss: %.4f' %loss.data.numpy())


test_data,test_data_means,test_data_stds = csv2test("test_data.csv")


test_data = torch.tensor(test_data).float()
test_output = rnn(test_data)
test_output = test_output.detach().numpy().reshape(-1)

test_data_means = np.array(test_data_means)
test_data_stds = np.array(test_data_stds)

test_output = (test_output*test_data_stds)+test_data_means

res2csv(test_output)
