import torch
import re, os
from torch import nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import numpy as np
import pandas as pd

# Hyper Parameters
# train the training data n times, to save time, we just train 1 epoch
EPOCH = 50
BATCH_SIZE = 32
TIME_STEP = 10          # rnn time step / image height
INPUT_SIZE = 8         # rnn input size / image width
LR = 0.001               # learning rate
DATASIZE = 427700
KEEP_ON = False
save_dir = "data/model_LSTM"
st = 0

def csv2train(filename):
    df = pd.read_csv(filename, index_col=0)
    groups = df.groupby('Date')

    train_data = []
    for date, group in groups:
        print("load data for data:", date)
        # cal diff of volume
        group["VolDiff"] = group["Volume"] - group["Volume"].shift(1)
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
            train_data.append((dataset[i:i+10], dataset[i+10:i+30, 0].mean()))

    return train_data, 0, 1


def csv2test(filename):
    df = pd.read_csv(filename, index_col=0)
    groups = df.groupby('Date')

    test_data = []
    test_data_means = []
    test_data_stds = []
    for date, group in groups:
        print("load data for data:", date)
        # cal diff of volume
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
    fout = open("data/LSTM_test_predict.csv", "w")
    fout.write("caseid,midprice\n")
    for i in range(len(dataset)):
        if(i < 142):
            continue
        fout.write("%d,%f\n" % (i+1, dataset[i]))
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
        # x shape (batch, time_step, input_size)
        # r_out shape (batch, time_step, output_size)
        # h_n shape (n_layers, batch, hidden_size)
        # h_c shape (n_layers, batch, hidden_size)
        # None represents zero initial hidden state
        r_out, (h_n, h_c) = self.rnn(x, None)

        # choose r_out at the last time step
        out = self.out(r_out[:, -1, :])
        return out

if __name__ == '__main__':

    train_data, train_mean, train_std = csv2train("data/train_data.csv")
    print(len(train_data))
    train_loader = torch.utils.data.DataLoader(
        dataset=train_data, batch_size=BATCH_SIZE, shuffle=False)
    device = torch.device("cuda:0")
    rnn = RNN()
    print(rnn)

    #load net_parameters

    if KEEP_ON:
        res = os.listdir(save_dir)
        for netFile in res:
            last = int(re.sub("\D","",netFile))
            if last > st:
                st = last
        rnn = torch.load(save_dir + "/" + str(st) + ".pkl", \
						map_location = {"cuda:2":"cuda:0"})
    
    rnn.to(device)
    # optimize all cnn parameters
    optimizer = torch.optim.Adam(rnn.parameters(), lr=0.00001, betas=(
        0.9, 0.999), eps=1e-08, weight_decay=0)
    # the target label is not one-hotted
    loss_func = nn.MSELoss()
    # training and testing
    for epoch in range(st, EPOCH):
        for step, (inputs, targets) in enumerate(train_loader):        # gives batch data
            # reshape x to (batch, time_step, input_size)
            optimizer.zero_grad()
            inputs = inputs.view(-1, 10, INPUT_SIZE).float().to(device)
            targets = targets.float().to(device)
            output = rnn(inputs) 
            loss = loss_func(output.view(BATCH_SIZE), targets)
            loss.backward()                                 # backpropagation, compute gradients
            optimizer.step()    
            if step % 50 == 0:                            # update gradients
                print('[train] Epoch: %d, batch: %d, loss: %.8f' % (epoch + 1,  (step + 1), loss))
        # save net parameters
        torch.save(rnn, save_dir + "/" + str(epoch + 1) + ".pkl")

    print("over")

    #load test data
    test_data, test_data_means, test_data_stds = csv2test("data/test_data.csv")

    #predict
    test_data = torch.tensor(test_data).float().to(device)
    test_output = rnn(test_data)
    test_output = test_output.detach().cpu().numpy().reshape(-1)

    test_data_means = np.array(test_data_means)
    test_data_stds = np.array(test_data_stds)

    print(test_output.shape)
    print(test_data_means.shape)
    print(test_data_stds.shape)
    #re scaler
    test_output = (test_output*test_data_stds)+test_data_means

    print(test_output.shape)
    res2csv(test_output)
