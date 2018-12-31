import torch
import numpy as np
import pandas as pd
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn import preprocessing
import torch.nn.functional as F


EPOCH = 1
BATCH = 64
TIME_STEP = 1
INPUT_SIZE = 6
LR = 0.01


class DiabetesDataset(Dataset):
    def __init__(self, filepath):
        train = pd.read_csv(filepath)
        # target_array = preprocessing.scale(target_array)
        target = pd.read_csv('train_midprice.csv')[['MidPrice']]
        target = np.array(target).astype(np.float32)
        volume = train['BidVolume1'] - train['AskVolume1']
        # price = train['BidPrice1'] - train['AskPrice1']
        other = train[['Volume', 'BidPrice1', 'AskPrice1', 'BidVolume1', 'AskVolume1']]
        volume = pd.DataFrame({'MidVolume': list(volume)})
        # price = pd.DataFrame({'Price': list(price)})

        data = pd.concat([other, volume], axis=1)
        data = np.array(data)
        for i in range(INPUT_SIZE):
            data[:, i] = preprocessing.scale(data[:, i])
        data = torch.tensor(data.astype(np.float32))
        self.len = data.shape[0]
        self.x_data = data
        self.y_data = target
        self.y_data = target
        print(self.y_data.shape)

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len


class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()
        self.hidden_size = BATCH
        self.rnn = nn.LSTM(
            input_size=INPUT_SIZE,
            hidden_size=self.hidden_size,  # rnn hidden unit
            num_layers=1,  # 有几层 RNN layers
            batch_first=True,
        )
        self.out = nn.Linear(BATCH, 1)

    def forward(self, x):
        self.hidden = (torch.zeros(1, BATCH, self.hidden_size),
                       torch.zeros(1, BATCH, self.hidden_size))
        r_out, self.hidden= self.rnn(x, self.hidden)
        out = self.out(r_out[:, -1, :])
        return out


rnn = RNN()
# rnn = nn.DataParallel(rnn, device_ids=[0, 1])
print(rnn)
dataset = DiabetesDataset(filepath='train.csv')
train_loader = DataLoader(dataset=dataset, batch_size=BATCH, shuffle=True)
optimizer = torch.optim.Adam(rnn.parameters(), lr=LR, betas=(0.9, 0.99))
loss_func = nn.MSELoss(size_average=False)


def train():
    for epoch in range(EPOCH):
        for step, (data, target) in enumerate(train_loader):        # gives batch data
            if data.shape[0] != BATCH:
                continue
            data = data.view(-1, TIME_STEP, INPUT_SIZE)

            output = rnn(data)
            loss = loss_func(output, target)                   # cross entropy loss
            optimizer.zero_grad()                           # clear gradients for this training step
            loss.backward(retain_graph=True)                                 # backpropagation, compute gradients
            optimizer.step()                                # apply gradients
            if step % 10 == 0:
                print('Train Epoch:{}[{}/{} ({:.0f}%)]\tLoss:{:.6f}'.format(
                    epoch, step * len(data), len(train_loader.dataset),
                           100. * step / len(train_loader), loss.item()
                ))


if __name__ == "__main__":
    train()
    torch.save(rnn, 'net.pkl')  # 保存整个网络

