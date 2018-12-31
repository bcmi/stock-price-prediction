import torch
from torch import nn
import matplotlib.pyplot as plt
import pandas as pd
from torch.utils.data import Dataset,DataLoader,TensorDataset
import numpy as np
import csv
# Hyper Parameters
EPOCH = 25

BATCH_SIZE = 32
INPUT_SIZE = 7
learningrate = 0.0001

#result GPU 加速完成


class LSTM(nn.Module):
	def __init__(self):
		super(LSTM,self).__init__()

		self.rnn = nn.LSTM(
			input_size = INPUT_SIZE,
			hidden_size = 64,
			num_layers = 2,
			batch_first = True,
		)
		self.out = nn.Linear(64,1)

	def forward(self,x):
		r_out, (h_n,c_n) = self.rnn(x,None)
		last_out = r_out[:,-1,:]
		out = self.out(last_out)
		return out

def get_dataset(dataset,mean,std):
	normalized_data=(dataset-mean)/std   
	data_X,data_Y=[],[]
	flag = True
	for i in range(len(normalized_data)-30-1):
		x=normalized_data[i:i+10]     
		y=normalized_data[i+10:i+10+20,0]
		data_X.append(x)
		data_Y.append(np.mean(y))

	data_X = torch.from_numpy(np.array(data_X))
	data_Y = torch.from_numpy(np.array(data_Y))

	return data_X,data_Y

def get_t_dataset(dataset,mean,std):
	normalized_data=(dataset-mean)/std
	data_X= []
	for i in range(0,len(normalized_data),10):
		x=normalized_data[i:i+10]

		data_X.append(x)
	data_X = torch.from_numpy(np.array(data_X))
	return data_X

device = torch.device("cuda:0")
net = LSTM()

if torch.cuda.device_count() > 1:
    # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
    net = nn.DataParallel(net)
print("Let's use", torch.cuda.device_count(), "GPUs!")
input("Input to cotinue.")

net = net.to(device)

f=open('train_data.csv')
df=pd.read_csv(f)     #训练集

dataset=df.loc[:,['MidPrice','Volume','BidPrice1','BidVolume1','AskPrice1','AskVolume1','LastPrice']].values
dataset = dataset.astype(np.float32)

mean = np.mean(dataset,axis=0)
std = np.std(dataset,axis=0)

train_feature,train_target = get_dataset(dataset[0:],mean,std)
train_dataset = TensorDataset(train_feature,train_target)

f2 = open('test_data.csv')
df2=pd.read_csv(f2)     #测试集
dataset2=df2.loc[:,['MidPrice','Volume','BidPrice1','BidVolume1','AskPrice1','AskVolume1','LastPrice']].values


newset = []
for i in range(1000):
    newset.extend(dataset2[i*10:i*10+10])
newset = np.array(newset)

newset = newset.astype(np.float32)

mean2 = np.mean(newset,axis=0)
std2 = np.std(newset,axis=0)

test_feature = get_t_dataset(newset,mean2,std2)
#test_dataset = TensorDataset(test_feature,test_target)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_feature, batch_size=1, shuffle=False)

optimizer = torch.optim.Adam(net.parameters(), lr=learningrate)
loss_func = nn.MSELoss()


for epoch in range(EPOCH):
    net.train()
    print("epoch:",epoch)
    for step, input_data in enumerate(train_loader):
        x, y = input_data
        x,y = x.to(device), y.to(device)
        pred_y = net(x)[:,0]

        loss = loss_func(pred_y,y)

        if step % 50 == 0:
            print("{}/{} steps".format(step, len(train_loader)), loss, pred_y[0], y[0])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

a = []
b = []
net.eval()
for i,x in enumerate(test_loader):
	x = x.float()
	
	pred = net(x)[:,0]
	pred = float(pred)*std2[0]+mean2[0]
	a.append(pred)
	b.append(i+1)

with open('submit.csv','w', newline='') as fout:
    fieldnames = ['caseid','midprice']
    writer = csv.DictWriter(fout,fieldnames = fieldnames)
    writer.writeheader()
    for i in range(142,len(a)):
        writer.writerow({'caseid':str(b[i]),'midprice':float(a[i])})

# 保存模型
torch.save(net, 'net_cuda_0.pkl')

# model = torch.load('model.pkl')