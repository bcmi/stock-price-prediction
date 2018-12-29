import pandas as pd
import numpy as np
import csv
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, LSTM, TimeDistributed, RepeatVector
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import load_model
from sklearn import preprocessing
import random
import time
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import matplotlib.pyplot as plt

random.seed(time.time())

class TestCaseData(object):
    def __init__(self,
                 input_size=6,
                 time_steps=10,
                ):
        self.time_steps = time_steps;
        self.input_size = input_size;
        
        # 获取训练集的MinMax分布
        df_train = pd.read_csv("train_data.csv")
        self.data_train = df_train[["LastPrice", "Volume", "BidPrice1",  
                   "BidVolume1", "AskPrice1", "AskVolume1"]].values
        self.Mins, self.Maxs = self.data_train.min(axis=0), self.data_train.max(axis=0)
        self.lower_bound, self.upper_bound = 0, 1


        df = pd.read_csv("test_data.csv")
        self.data = df[["LastPrice", "Volume", "BidPrice1",  
                   "BidVolume1", "AskPrice1", "AskVolume1"]].values

        # 按训练集的MinMax分布归一化
        data_std = (self.data - self.data.min(axis=0)) / (self.data.max(axis=0) - self.data.min(axis=0))
        self.normalized_data = data_std * (self.upper_bound - self.lower_bound) + self.lower_bound
        # scaler = preprocessing.MinMaxScaler()
        # self.normalized_data = scaler.fit_transform(self.data)

    def get_test_cases(self, time_steps=10):
        
        test_cases = []
        for i in range(1000):
            x = self.normalized_data[i * time_steps: (i+1) * time_steps, :]
            test_cases.append(x)
        
        test_cases = np.array(test_cases)
        return test_cases
    
    # def get_mean_std(self, time_steps=10):
    #     means = []
    #     stds = []
        
    #     for i in range(1000): # totally 1000 test cases
    #         x = self.data[i * time_steps: (i+1) * time_steps, :self.input_size]
    #         std = np.std(x,axis=0)
    #         mean = np.mean(x, axis=0)
    #         means.append([mean[0]])
    #         stds.append([std[0]])
    #     return np.array(means), np.array(stds)
        
def main():

    T = TestCaseData()
    X_test_cases = T.get_test_cases()
    # means, stds = T.get_mean_std()

    model = load_model('saved_model.h5')
    res = model.predict(X_test_cases)


    # res_tmp = (res - T.lower_bound) / (T.upper_bound - T.lower_bound)
    # res = res_tmp * (T.Maxs[0] - T.Mins[0]) + T.Mins[0]

    with open('res.csv','w',encoding='utf8',newline='') as fout:
        fieldnames = ['caseid','midprice']
        writer = csv.DictWriter(fout, fieldnames = fieldnames)
        writer.writeheader()
        for i in range(142, len(res)):
            writer.writerow({'caseid':str(i+1),'midprice':float(res[i][0])})
    
if __name__ == '__main__':
    main()