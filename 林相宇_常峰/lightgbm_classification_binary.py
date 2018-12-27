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
	new_vol = pd.DataFrame(volumes,columns = ['BidVolume1','AskVolume1',"DeltaVolume"])
	res[['BidVolume1','AskVolume1',"DeltaVolume"]] = new_vol
	return res




def train_data_gen_lgbm(features,labels):
	L = len(labels) - 30
	f = list()
	l = list()
	for i in range(L):
		label_for_case = np.mean(labels[i + 10: i + 30]) - features[i + 9][0]
		if(label_for_case >= 0):
			label_for_case = 1
		elif(label_for_case < 0):
			label_for_case = 0
		l.append(label_for_case)
	for i in range(L):
		pivot = features[i + 9][0]
		pivot_array = np.array([pivot,pivot,pivot,0,pivot,0,0])
		feature_for_case = features[i: i + 10] - pivot_array
		f.append(feature_for_case.reshape(70))
	return np.array(f),np.array(l)


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


params = {
	'task': 'train',
	'objective': 'binary',
	'metrics':'auc',
	'early_stopping_rounds':100,
	'n_estimators':500,
	'learning_rate': 0.05,
}
			
if __name__ == '__main__':
	features, labels = load_csv_data('train_data.csv')
	features_test = load_csv_test('test_data.csv')
	displacement = 0.0005

	features2, labels2= train_data_gen_lgbm(features,labels)
	test2,pvs2= test_data_gen_lgbm(features_test)
	filepath = "best.h5"
	split = int(len(labels)*0.8)
	import lightgbm as lgb
	lgb_train = lgb.Dataset(features2[:split], labels2[:split])
	lgb_val = lgb.Dataset(features2[split:], labels2[split:])
	gbm = lgb.train(params, lgb_train, 50,  verbose_eval=100,valid_sets=[lgb_train, lgb_val],early_stopping_rounds = 200)
	predict = gbm.predict(test2,num_iteration=gbm.best_iteration)
	res = list()
	for i in range(len(predict)):
		if(predict[i] < 0.5):
			res.append(pvs2[i] - displacement)
		elif(predict[i] >= 0.5):
			res.append(pvs2[i] + displacement)
	with open("predictclassification.csv", "w", newline='') as csvfile:
		writer = csv.writer(csvfile)
		writer.writerow(
			["caseid", "midprice"])
		for i in range(142, len(res)):
			writer.writerow([i+1, float(res[i])])
