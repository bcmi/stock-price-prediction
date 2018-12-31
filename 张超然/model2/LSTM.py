import pandas as pd
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
import numpy as np
import csv
import os
from keras.callbacks import EarlyStopping
from keras.models import load_model
from keras.layers.core import Activation
import sklearn
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler

def load_data(train_path=""):
	train = pd.read_csv(train_path,header=1) # 读入数据
	data=train.iloc[:,3:].values
	return data
def ave(nums):
	l=len(nums)
	sum=0
	for i in nums:
		sum+=i
	res=sum/l
	return res
def presess(data):
	inp=[]
	outp=[]
	for i in range(0,430009):
		tmp=[]
		for j in range(0,10):
			tmp.append(data[i+j])
		inp.append(tmp)
		#if(i%1000==0):
			#print(tmp)
		tmp=[]
		for j in range(10,30):
			tmp.append(data[i+j,0])
			r=ave(tmp)
		outp.append(r)
		#if(i%1000==0):
			#print(r)
	return inp,outp
train=load_data("train_data.csv")
train_in,train_out=presess(train)
#i=pd.DataFrame(columns=["MidPrice","LastPrice","Volume","BidPrice1","BidVolume1","AskPrice1","AskVolume1"],data=train_in)
#o=pd.DataFrame(columns=["MidPrice"],data=train_out)
#i.to_csv('E:\Artificial Intelligence\train_in_pressessed.csv')
#o.to_csv('E:\Artificial Intelligence\train_out_pressessed.csv')
cut=int(len(train_in)*0.99)
x_train=np.array(train_in[:cut],ndmin=2)
y_train=np.array(train_out[:cut])
x_val=np.array(train_in[cut:],ndmin=2)
y_val=np.array(train_out[cut:])
#print(x_val)
#print(y_val)
scalers={}
for i in range(x_train.shape[2]):
	scalers[i]=StandardScaler()
	x_train[:,i,:]=scalers[i].fit_transform(x_train[:,i,:])
for i in range(x_val.shape[2]):
	x_val[:,i,:]=scalers[i].transform(x_val[:,i,:])
model =Sequential()
model.add(LSTM(128,input_shape=(x_train.shape[1],x_train.shape[2]),return_sequences=True))
model.add(Dropout(0.5))
model.add(LSTM(128,return_sequences=True))
model.add(LSTM(128,return_sequences=False))
model.add(Activation('relu'))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam', metrics=['mse'])
callback=EarlyStopping(monitor="loss",patience=10,verbose=1,mode='auto')
model.fit(x_train, y_train, epochs=5,validation_data=(x_val,y_val), batch_size=128, callbacks=[callback],shuffle=True)
model.summary()
model.save('lstm_model.h5')
model = load_model('lstm_model.h5')

test=load_data("test_data.csv")
test_in=[]
for i in range(1430,9990,10):
	tmp=[]
	for j in range(0,10):
		#print(test[i+j])
		tmp.append(test[i+j])
	test_in.append(tmp)
test_x=np.array(test_in)
for i in range(test_x.shape[2]):
	test_x[:,i,:]=scalers[i].transform(test_x[:,i,:])
res=model.predict(test_x)
o=pd.DataFrame(columns=["midprice"],data=res)
o.to_csv('out_pressessed.csv')
#print (res)

