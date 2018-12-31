import torch
import os, re
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import datetime

import torch.utils.data as Data

INPUT_SIZE = 8
seq_len = 10
BATCH_SIZE = 16
LR = 1e-4
device = torch.device("cuda:0")
EPOCH = 2
KEEP_ON = True
save_dir = "data/model_DNN"
st = 0


def csv2train(filename):
    df = pd.read_csv(filename, index_col=0)
    groups = df.groupby('Date')

    train_data = []
    train_tag = []
    for date, group in groups:
        print("load data for date:", date)
        # cal diff of volume
        col_name = group.columns.tolist()
        col_name.insert(col_name.index("Volume"), "VolDiff")
        group = group.reindex(columns=col_name)
        # print(group.columns)

        group.loc[:, ["VolDiff"]] = group["Volume"] - group["Volume"].shift(1)

        group_new = group.fillna(0)
        # print(group_new.columns)

        # trans to numpy
        dataset = np.array(group_new.loc[:, ['MidPrice', 'LastPrice', 'Volume',
                                             'BidPrice1', 'BidVolume1', 'AskPrice1', 'AskVolume1', 'VolDiff']])

        # norm data in size of group
        dataset.astype(np.float32)
        mean = dataset.mean(axis=0)
        std = dataset.std(axis=0)
        # print("mean:\n",mean,'\nstd:\n',std)
        dataset = (dataset - mean)/std

        #add to data set with dropout improper additional data
        size = len(dataset)-30
        size = size - (size % BATCH_SIZE)
        for i in range(0, size, 1):
            train_data.append(dataset[i:i+10])
            train_tag.append(dataset[i+10:i+30, 0].mean())
    train_data = torch.FloatTensor(train_data)
    train_tag = torch.FloatTensor(train_tag)

    return (train_data, train_tag), 0, 1


def csv2test(filename):
    df = pd.read_csv(filename, index_col=0)
    groups = df.groupby('Date')

    test_data = []
    test_data_means = []
    test_data_stds = []
    for date, group in groups:
        # print("load data for date:", date)
        # cal diff of volume
        col_name = group.columns.tolist()
        col_name.insert(col_name.index("Volume"), "VolDiff")
        group = group.reindex(columns=col_name)
        # print(group.columns)
        group["VolDiff"] = group["Volume"] - group["Volume"].shift(1)
        group_new = group.fillna(0)
        # print(group_new.columns)

        # trans to numpy
        dataset = np.array(group_new.loc[:, ['MidPrice', 'LastPrice', 'Volume',
                                             'BidPrice1', 'BidVolume1', 'AskPrice1', 'AskVolume1', 'VolDiff']])

        # norm data in size of group
        dataset.astype(np.float32)
        dataset.astype(np.float32)
        mean = dataset.mean(axis=0)
        std = dataset.std(axis=0)
        # print("mean:\n", mean, '\nstd:\n', std)
        dataset = (dataset - mean)/std

        mean_midprice = mean[0]
        std_midprice = std[0]

        #add to data set with dropout improper additional data
        size = len(dataset)
        for i in range(0, size, 10):
            test_data.append(dataset[i:i+10])
            test_data_means.append(mean_midprice)
            test_data_stds.append(std_midprice)

    return test_data, test_data_means, test_data_stds


def res2csv(dataset):
    now = datetime.datetime.now()
    fout = open("data/DNN_test_predict.csv", "w")
    fout.write("caseid,midprice\n")
    for i in range(len(dataset)):
        if(i < 142):
            continue
        fout.write("%d,%f\n" % (i+1, dataset[i]))
    fout.close()


class OurDNN(nn.Module):
    def __init__(self, seq_len):
        super(OurDNN, self).__init__()
        self.seq_len = seq_len
        self.fc1 = nn.Linear(seq_len, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, inputs):
        x = inputs
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def adjust_learning_rate(LR, optimizer, epoch):
    lr = LR * (0.1 ** (epoch // 4))
    for group in optimizer.param_groups:
        group['lr'] = lr



# Prepare dataset
# trainset = initDatabase("train")
# testset = initDatabase("test")

# Prepare model
net = OurDNN(seq_len)

if KEEP_ON:
    res = os.listdir(save_dir)
    for netFile in res:
        last = int(re.sub("\D","",netFile))
        if last > st:
            st = last
    rnn = torch.load(save_dir + "/" + str(st) + ".pkl", \
                    map_location = {"cuda:2":"cuda:0"})

net.to(device)
optimizer = optim.Adam(net.parameters(),
                        lr=LR, weight_decay=0.01)
crit = nn.MSELoss()



train_data, train_mean, train_std = csv2train("data/train_data.csv")
print(len(train_data[0]))

train_dataset = Data.TensorDataset(train_data[0], train_data[1])
train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=False)


for epoch in range(st, EPOCH):
    LOSS = 0
    b_num = 0
    adjust_learning_rate(LR, optimizer, epoch)
    for i, (inputs, targets) in enumerate(train_loader):
        b_num += 1

        inputs = inputs[:, :, 0]
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = net(inputs)
        optimizer.zero_grad()
        loss = crit(outputs.view(-1), targets)
        loss.backward()
        LOSS += loss.data.cpu()
        #acc += torch.sqrt(loss)
        optimizer.step()
        if i % 100 == 0:
            print("train epoch %d, batch %d, loss: %.10f" %
                    (epoch, i, LOSS / b_num))
    torch.save(net, save_dir + "/" + str(epoch + 1) + ".pkl")


net.cpu()

#load test data
test_data, test_data_means, test_data_stds = csv2test("data/test_data.csv")
#predict
test_data = torch.tensor(test_data).float()
test_data = test_data[:, :, 0]
test_output = net(test_data)
test_output = test_output.detach().numpy().reshape(-1)
test_data_means = np.array(test_data_means)
test_data_stds = np.array(test_data_stds)

#re scaler
test_output = (test_output*test_data_stds)+test_data_means

print(test_output.shape)
res2csv(test_output)
