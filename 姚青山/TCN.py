import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
import matplotlib.pyplot as plt
import pandas as pd
from torch.utils.data import Dataset,DataLoader,TensorDataset
import numpy as np
import csv

TIME_STEP = 10
BATCH_SIZE = 64
EPOCH = 10
LR = 0.001

test_normal = []
class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)
        self.out = nn.Linear(num_channels[-1],1)
    def forward(self, x):
        r_out = self.network(x)
        last_out = r_out[:,:,0]
        out = self.out(last_out)
        return out

def get_dataset(dataset,mean,std,datadate):
    #normalized_data=(dataset-mean)/std
    data_X,data_Y=[],[]
    flag = True
    for i in range(len(dataset)-30-1):
        flag = True
        for j in range(i+1,i+TIME_STEP+20):
            if(datadate[j] != datadate[j-1]):
                flag = False
                break

        if(flag):
            x=dataset[i:i+TIME_STEP]
            y=dataset[i+TIME_STEP:i+TIME_STEP+20,0]
            y=np.mean(y)-x[9][0]
            mean = np.mean(x,axis=0)
            std = np.std(x,axis=0)

            x = (x-mean)/(std + 0.1)

            data_X.append(x)
            data_Y.append(y)

    data_X = torch.from_numpy(np.array(data_X))
    data_Y = torch.from_numpy(np.array(data_Y))

    return data_X,data_Y

def get_t_dataset(dataset,mean,std):
    #normalized_data=(dataset-mean)/std
    data_X= []
    for i in range(0,len(dataset),10):
        x=dataset[i:i+TIME_STEP]
        mean = np.mean(x,axis=0)
        std = np.std(x,axis=0)
        x = (x-mean)/(std+0.1)
        data_X.append(x)
        test_normal.append((mean[0],std[0]+0.1))
    data_X = torch.from_numpy(np.array(data_X))
    return data_X




f=open('train_data.csv')
df=pd.read_csv(f)     #训练集


dataset=df.loc[:,['MidPrice','Volume','BidPrice1','BidVolume1','AskPrice1','AskVolume1','LastPrice']].values
dataset = dataset.astype(np.float32)

# #Consider the time
# dataset=df.loc[:,['MidPrice','Volume','BidPrice1','BidVolume1','AskPrice1','AskVolume1','LastPrice','Time']].values
datadate=df.loc[:,['Date']].values
# for i in range(len(dataset)):
#   tmp = dataset[i,7]
#   h,m,s = tmp.split(":")
#   dataset[i,7] = int(h)*60*60 + int(m)*60 + int(s)
# dataset = dataset.astype(np.float32)

################################

f.close()

mean = np.mean(dataset,axis=0)
std = np.std(dataset,axis=0)
print(std)
train_feature,train_target = get_dataset(dataset[0:],mean,std,datadate)
train_dataset = TensorDataset(train_feature,train_target)

f2 = open('test_data.csv')
df2=pd.read_csv(f2)     #测试集
dataset2=df2.loc[:,['MidPrice','Volume','BidPrice1','BidVolume1','AskPrice1','AskVolume1','LastPrice']].values
#dataset2=df2.loc[:,['MidPrice','Volume','BidPrice1','BidVolume1','AskPrice1','AskVolume1','LastPrice','Time']].values

newset = []
for i in range(1000):
    newset.extend(dataset2[i*10:i*10+10])
newset = np.array(newset)
# print(newset)
# for i in range(len(newset)):
#   tmp = newset[i,7]
#   h,m,s = tmp.split(":")
#   newset[i,7] = int(h)*60*60 + int(m)*60 + int(s)
newset = newset.astype(np.float32)

mean2 = np.mean(newset,axis=0)
std2 = np.std(newset,axis=0)
print(mean2)
test_feature = get_t_dataset(newset,mean2,std2)
#test_dataset = TensorDataset(test_feature,test_target)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_feature, batch_size=1, shuffle=False)

TCN = TemporalConvNet(10,[128,64,32, 16])
print(TCN)
optimizer = torch.optim.Adam(TCN.parameters(), lr=LR)   # optimize all cnn parameters
loss_func = nn.MSELoss()
for epoch in range(EPOCH):
    TCN.train()
    print("epoch:",epoch)
    for step, input_data in enumerate(train_loader):
        x, y = input_data
        pred = TCN(x)
        loss = loss_func(pred.squeeze(),y)

        if step % 50 == 0:
            print(step,float(loss))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

a = []
b = []
TCN.eval()
for i,x in enumerate(test_loader):
    x = x.float()

    pred = TCN(x).squeeze()
    pred = float(x[0][9][0])*test_normal[i][1]+test_normal[i][0]+float(pred)
    a.append(pred)
    b.append(i+1)
    # rnn.eval()
    # sum_loss = 0
    # for step, input_data in enumerate(test_loader):
    #   x, y = input_data
    #   pred = rnn(x)[:,0]
    #   loss = loss_func(pred,y)
    #   sum_loss += loss

    #   if step % 500 == 0:
    #       print(step,sum_loss,pred,y)

with open('TCN5.csv','w') as fout:
        fieldnames = ['caseid','midprice']
        writer = csv.DictWriter(fout, fieldnames = fieldnames)
        writer.writeheader()
        for i in range(142,len(a)):
            writer.writerow({'caseid':str(b[i]),'midprice':float(a[i])})
