import torch
from torch import nn
import matplotlib.pyplot as plt
import pandas as pd
from torch.utils.data import Dataset,DataLoader,TensorDataset
import numpy as np
import csv
# Hyper Parameters
EPOCH = 10

BATCH_SIZE = 64
TIME_STEP = 10
INPUT_SIZE =7
LR = 0.001

test_normal = []

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
		r_out, (h_n,c_n) = self.rnn(x,h_state)
		last_out = r_out[:,-1,:]
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
			if(dataset[j][7] - dataset[j-1][7] != 3):
				flag = False
				break

		if(flag):
			x=dataset[i:i+TIME_STEP,0:7]
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

rnn = LSTM()
print(rnn)


f=open('train_data.csv')
df=pd.read_csv(f)     #训练集


# dataset=df.loc[:,['MidPrice','Volume','BidPrice1','BidVolume1','AskPrice1','AskVolume1','LastPrice']].values
# dataset = dataset.astype(np.float32)

#Consider the time
dataset=df.loc[:,['MidPrice','Volume','BidPrice1','BidVolume1','AskPrice1','AskVolume1','LastPrice','Time']].values
datadate=df.loc[:,['Date']].values
for i in range(len(dataset)):
	tmp = dataset[i,7]
	h,m,s = tmp.split(":")
	dataset[i,7] = int(h)*60*60 + int(m)*60 + int(s)
dataset = dataset.astype(np.float32)

################################

f.close()

mean = np.mean(dataset,axis=0)
std = np.std(dataset,axis=0)

train_feature,train_target = get_dataset(dataset[0:],mean,std,datadate)
train_dataset = TensorDataset(train_feature,train_target)

f2 = open('test_data.csv')
df2=pd.read_csv(f2)     #测试集
dataset2=df2.loc[:,['MidPrice','Volume','BidPrice1','BidVolume1','AskPrice1','AskVolume1','LastPrice']].values
# dataset2=df2.loc[:,['MidPrice','Volume','BidPrice1','BidVolume1','AskPrice1','AskVolume1','LastPrice','Time']].values

newset = []
for i in range(1000):
	newset.extend(dataset2[i*10:i*10+10])
newset = np.array(newset)
# print(newset)
# for i in range(len(newset)):
# 	tmp = newset[i,7]
# 	h,m,s = tmp.split(":")
# 	newset[i,7] = int(h)*60*60 + int(m)*60 + int(s)
newset = newset.astype(np.float32)

mean2 = np.mean(newset,axis=0)
std2 = np.std(newset,axis=0)
print(mean2)
test_feature = get_t_dataset(newset,mean,std)
#test_dataset = TensorDataset(test_feature,test_target)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_feature, batch_size=1, shuffle=False)

optimizer = torch.optim.Adam(rnn.parameters(), lr=LR)
loss_func = nn.MSELoss()

h_state = None

for epoch in range(EPOCH):
	rnn.train()
	print("epoch:",epoch)
	for step, input_data in enumerate(train_loader):
		x, y = input_data

		pred = rnn(x)[:,0]

		loss = loss_func(pred,y)

		if step % 50 == 0:
			print(step,float(loss),pred,y)
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()


a = []
b = []
rnn.eval()
for i,x in enumerate(test_loader):
	x = x.float()

	pred = rnn(x)[:,0]
	pred = float(x[0][9][0])*test_normal[i][1]+test_normal[i][0]+float(pred)
	a.append(pred)
	b.append(i+1)
	# rnn.eval()
	# sum_loss = 0
	# for step, input_data in enumerate(test_loader):
	# 	x, y = input_data
	# 	pred = rnn(x)[:,0]
	# 	loss = loss_func(pred,y)
	# 	sum_loss += loss

	# 	if step % 500 == 0:
	# 		print(step,sum_loss,pred,y)

with open('s20.csv','w') as fout:
        fieldnames = ['caseid','midprice']
        writer = csv.DictWriter(fout, fieldnames = fieldnames)
        writer.writeheader()
        for i in range(142,len(a)):
            writer.writerow({'caseid':str(b[i]),'midprice':float(a[i])})
