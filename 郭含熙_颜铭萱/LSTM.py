from torch import nn
import torch
import torch.utils.data as Data
import pandas as pd
import csv
import numpy as np

EPOCH = 5  # train the training data n times, to sa
BATCH_SIZE = 32  # Number of data the network process at one time
TIME_STEP = 10  # rnn time step
INPUT_SIZE = 7  # rnn input size
LR = 0.1  # learning rate


# Processing data
def load_csv_data(filename):
    file = pd.read_csv(filename)
    data = file.loc[:, ['LastPrice', 'Volume', 'BidPrice1', 'BidVolume1', 'AskPrice1', 'AskVolume1', 'MidPrice']].values
    predict = file.loc[:, ['LastPrice', 'Volume', 'BidPrice1', 'BidVolume1', 'AskPrice1', 'AskVolume1', 'MidPrice']].values
    data = data.astype(np.float32)
    predict = predict.astype(np.float32)
    return data, predict


def load_csv_test_data(filename):
    file = pd.read_csv(filename)
    data = file.loc[:, ['LastPrice', 'Volume', 'BidPrice1', 'BidVolume1', 'AskPrice1', 'AskVolume1', 'MidPrice']].values
    data = data.astype(np.float32)
    return data


def load_csv_test_data2(filename):
    file = pd.read_csv(filename)
    data = file.loc[:, ['LastPrice', 'Volume', 'BidPrice1', 'BidVolume1', 'AskPrice1', 'AskVolume1', 'MidPrice']].values
    data = data.astype(np.float32)
    return data


def process_data_test(data):
    for i in range(len(data)-1, 1, -1):
        data[i][1] -= data[i - 1][1]
    data[0][1] = 0
    feature = []
    for j in range(0, len(data), 10):
        data[j:j+10][0][1] = 0
        mean = np.mean(data[j:j+10], axis=0)
        std = np.mean(data[j:j+10], axis=0)
        data[j:j+10] = (data[j:j+10] - mean) / (std + 0.00000001)
        feature.append(data[j:j + 10])
    feature = torch.from_numpy(np.array(feature))
    return feature


def process_data(data, predict):
    for i in range(len(data)-1, 1, -1):
        data[i][1] -= data[i - 1][1]
    data[0][1] = 0
    data_feature = []
    data_predict = []
    for j in range(0, len(data) - 30 - 1, 1):
        tmp = data[j:j+10]
        tmp[0][1] = 0
        mean = np.mean(tmp, axis=0)
        std = np.std(tmp, axis=0)
        tmp = (tmp - mean) / (std + 0.00000001)
        data_feature.append(tmp)
        pre = predict[j+10:j+30, 6]
        data_predict.append(np.mean(pre)-predict[j+9, 6])
    data_predict = np.array(data_predict)
    data_feature = torch.from_numpy(np.array(data_feature))
    data_predict = torch.from_numpy(data_predict)
    return data_feature, data_predict

data, predict = load_csv_data('train_data.csv')
feature, predict1 = process_data(data, predict)
train = Data.TensorDataset(feature, predict1)
dataloader = Data.DataLoader(dataset=train, batch_size=BATCH_SIZE, shuffle=True)
x_test = load_csv_test_data('test_data.csv')
x_test = process_data_test(x_test)
dataloader_test = Data.DataLoader(dataset=x_test, batch_size=1, shuffle=False)


# LSTM Module
class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()
        self.rnn = nn.LSTM(
            input_size=INPUT_SIZE,
            hidden_size=64,  # rnn hidden unit
            num_layers=3,  # number of rnn layer
            batch_first=True
        )
        self.out = nn.Linear(64, 1)

    def forward(self, x):
        r_out, (h_n, h_c) = self.rnn(x, None)
        out = self.out(r_out[:, -1, :])
        return out

# Training
rnn = RNN()
loss_func = nn.MSELoss()
loss_func = loss_func.cuda()
rnn = rnn.cuda()
for k in range(1):
    rnn.train()
    i = 0
    for epoch in range(EPOCH):
        if epoch != 0:
            LR = float(LR)/10
        optimizer = torch.optim.Adam(rnn.parameters(), lr=LR)
        for step, (b_x, b_y) in enumerate(dataloader):
            b_x = b_x.cuda()
            b_y = b_y.cuda()
            output = rnn(b_x)
            loss = loss_func(output[:, 0], b_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if step % 500 == 49:
                print('Epoch: ', epoch, '| train loss: ', loss.data.cpu().numpy(), '\t', step, '\t LR=', LR, '\trnn', k)

    # Prediction and file writing
    y_pred = []
    rnn.eval()
    y_pred.append([])
    for step, test_input in enumerate(dataloader_test):
       test_input = test_input.cuda()
       output_test = rnn(test_input)
       res = output_test[:, 0][0]
       y_pred[k].append(res.cpu())
       print(res.data.cpu().numpy())

x_test = load_csv_test_data2('test_data.csv')
with open('sample.csv', 'w', newline='') as fout:
    fieldnames = ['caseid', 'midprice']
    writer = csv.DictWriter(fout, fieldnames=fieldnames)
    writer.writeheader()
    for i in range(142, len(y_pred[0])):
        tmp = 0
        for j in range(1):
            tmp += float(y_pred[j][i].data.numpy())
        tmp = tmp / 1
        writer.writerow({'caseid': str(i+1), 'midprice': float(tmp+x_test[(i+1)*10-1][6])})