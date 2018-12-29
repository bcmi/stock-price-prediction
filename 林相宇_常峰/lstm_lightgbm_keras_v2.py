# -*- coding: utf-8 -*-
import sys
import numpy as np
import pandas as pd
import csv
import math
import keras
from keras.models import Sequential
from keras.layers import Dense, LSTM, Activation, Dropout
from keras.layers.recurrent import SimpleRNN
from keras import optimizers
from sklearn import preprocessing
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from keras.callbacks import EarlyStopping
# Load data

seqlen = 10
batch_size = 512
feature_size = 7
#sc= preprocessing.StandardScaler()
sc = preprocessing.MinMaxScaler()
label_mmm = None
def load_csv_data(filename):
	file = pd.read_csv(filename)
	data = file[['MidPrice','LastPrice','BidPrice1','BidVolume1','AskPrice1','AskVolume1']]
	labels = file[['MidPrice']]
	data = feature_engineering(data, train = True)
	data = np.array(data)
	labels = np.array(labels)
	return data, labels

# Load data
def load_csv_test(filename):
	file = pd.read_csv(filename)
	data = file[['MidPrice','LastPrice','BidPrice1','BidVolume1','AskPrice1','AskVolume1']]
	data = feature_engineering(data)
	data = np.array(data)
	return data
	
def feature_engineering(features,train = False):
	deltavolume = features['BidVolume1']-features['AskVolume1']
	new_col = pd.DataFrame(deltavolume,columns = ["DeltaVolume"])
	res = pd.concat([features,new_col],axis = 1)
	volumes = res[['BidVolume1','AskVolume1',"DeltaVolume"]]
	volumes = np.array(volumes)
	if(train):
		volumes = sc.fit_transform(volumes)
	else:
		volumes = sc.transform(volumes)
	new_vol = pd.DataFrame(volumes,columns = ['BidVolume1','AskVolume1',"DeltaVolume"])
	res[['BidVolume1','AskVolume1',"DeltaVolume"]] = new_vol
	return res



def train_data_gen(features,labels):
	L = len(labels) - 30
	f = list()
	l = list()
	for i in range(L):
		label_for_case = np.mean(labels[i + 10: i + 30]) - features[i + 9][0]
		l.append(label_for_case)
	for i in range(L):
		pivot = features[i + 9][0]
		pivot_array = np.array([pivot,pivot,pivot,0,pivot,0,0])
		feature_for_case = features[i: i + 10] - pivot_array
		f.append(feature_for_case)
	return np.array(f),np.array(l)

def train_data_gen_lgbm(features,labels):
	L = len(labels) - 30
	f = list()
	l = list()
	for i in range(L):
		label_for_case = np.mean(labels[i + 10: i + 30]) - features[i + 9][0]
		l.append(label_for_case)
	for i in range(L):
		pivot = features[i + 9][0]
		pivot_array = np.array([pivot,pivot,pivot,0,pivot,0,0])
		feature_for_case = features[i: i + 10] - pivot_array
		f.append(feature_for_case.reshape(70))
	return np.array(f),np.array(l)

def test_data_gen(features):
	L = len(features)//10
	dat = list()
	pvs = list()
	for i in range(L):
		pvs.append(features[i*10 + 9][0])
	for i in range(L):
		pivot = features[i*10 + 9][0]
		pivot_array = np.array([pivot,pivot,pivot,0,pivot,0,0])
		feature_for_case = features[i*10: i*10 + 10] - pivot_array
		dat.append(feature_for_case)
	return np.array(dat),pvs

def test_data_gen_lgbm(features):
	L = len(features)//10
	dat = list()
	pvs = list()
	for i in range(L):
		pvs.append(features[i*10 + 9][0])
	for i in range(L):
		pivot = features[i*10 + 9][0]
		pivot_array = np.array([pivot,pivot,pivot,0,pivot,0,0])
		feature_for_case = features[i*10: i*10 + 10] - pivot_array
		dat.append(feature_for_case.reshape(70))
	return np.array(dat),pvs

def build_model():
	model = Sequential()
	model.add(LSTM(units=32, input_shape=(seqlen, feature_size), return_sequences=False))
	model.add(Dense(1))
	model.compile(loss="mse", optimizer='nadam')
	return model


def build_model2():
	d = 0.2
	model = Sequential()
	model.add(LSTM(16, input_shape=(seqlen, feature_size), return_sequences=True))
	model.add(Dropout(d))
	model.add(LSTM(16, input_shape=(seqlen, feature_size), return_sequences=False))
	model.add(Dropout(d))           
	model.add(Dense(1))
	model.compile(loss='mse',optimizer='nadam')
	return model

params = {
	'task': 'train',
	'objective': 'regression',
	'early_stopping_rounds':200,
	'n_estimators':20000,
	'metric': 'mse',
	'num_leaves': 31,
	'learning_rate': 0.01,
}
			
if __name__ == '__main__':
	features, labels = load_csv_data('train_data.csv')
	features_test = load_csv_test('test_data.csv')

	features1, labels1= train_data_gen(features,labels)
	test1,pvs1= test_data_gen(features_test)
	model = build_model()
	filepath = "best.hdf5"
	checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True,mode='min')
	callbacks_list = [checkpoint]
	history = model.fit(x=features1, y=labels1, batch_size=batch_size, verbose = 1,epochs=50,validation_split=0.3,shuffle=True,callbacks=callbacks_list)
	trainScore = model.evaluate(features1, labels1, verbose=0)
	model = load_model("best.hdf5")
	predict = model.predict(test1)
	print(predict)
	for i in range(len(predict)):
		predict[i] = predict[i] + pvs1[i]
	print('Train Score:'+ str(math.sqrt(trainScore)))
	with open("predictlstm.csv", "w", newline='') as csvfile:
		writer = csv.writer(csvfile)
		writer.writerow(
			["caseid", "midprice"])
		for i in range(142, len(predict)):
			writer.writerow([i+1, float(predict[i])])

	print(model.summary())
	import matplotlib.pyplot as plt
	plt.plot(history.history['loss'])
	plt.plot(history.history['val_loss'])
	plt.title("model loss")
	plt.ylabel("loss")
	plt.xlabel("epoch")
	plt.legend(["train","test"],loc="upper left")
	plt.show()

	features2, labels2= train_data_gen_lgbm(features,labels)
	test2,pvs2= test_data_gen_lgbm(features_test)
	filepath = "best.h5"
	split = int(len(labels)*0.8)
	import lightgbm as lgb
	lgb_train = lgb.Dataset(features2[:split], labels2[:split])
	lgb_val = lgb.Dataset(features2[split:], labels2[split:])
	gbm = lgb.train(params, lgb_train, 50,  verbose_eval=100,valid_sets=[lgb_train, lgb_val],early_stopping_rounds = 200)
	predict = gbm.predict(test2,num_iteration=gbm.best_iteration)
	for i in range(len(predict)):
		predict[i] = predict[i] + pvs2[i]
	with open("predictlgbm.csv", "w", newline='') as csvfile:
		writer = csv.writer(csvfile)
		writer.writerow(
			["caseid", "midprice"])
		for i in range(142, len(predict)):
			writer.writerow([i+1, float(predict[i])])
