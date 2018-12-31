import pandas as pd
import numpy as np
import keras
import csv
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, LSTM, TimeDistributed, RepeatVector
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam, RMSprop
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import load_model
from sklearn import preprocessing
import random
import time
import matplotlib.pyplot as plt
import os

#读取训练数据
f = open("washed_train_data.csv")
df = pd.read_csv(f)
rawInput = df.iloc[:, 1:].values
train_data_x = df[["LastPrice", "Volume", "BidPrice1",
           "BidVolume1", "AskPrice1", "AskVolume1"]].values
train_data_y = df[["MidPrice"]].values
f.close()

#保证该输入数据满足条件
def find_validate(inputdata, i):
    for j in range(9):
        if inputdata[i - 1, 0] != inputdata[i, 0]:
            return False
        time1 = inputdata[i - 1, 1].split(':')
        time2 = inputdata[i, 1].split(':')
        if (int(time1[2]) + 3) % 60 != int(time2[2]):
            return False
    return True

#将volumn改写为增量
for i in range(len(train_data_x) - 1):
    train_data_x[i + 1][4] = train_data_x[i + 1][4] - train_data_x[i][4]
    if (train_data_x[i + 1][4] < 0):
        train_data_x[i + 1][4] = 0
        
# MinMax归一化数据
scaler_x = preprocessing.MinMaxScaler()
train_data_x = scaler_x.fit_transform(train_data_x)

#生成训练集
train_set_x, train_set_y = [], []
for i in range(0, len(train_data_x) - 31):
    if find_validate(rawInput, i + 1):
        x = train_data_x[i : i + 10, :]
        y = [np.mean(train_data_y[i + 10 : i + 30])]
        y = y - train_data_y[i + 9][0]
        y = np.array(y) * 1000
        train_set_x.append(x.tolist())
        train_set_y.append(y.tolist())
    if i % 1000 == 0:
        print(i)

cut_pos = int(len(train_set_x) * 0.99)
X_train, Y_train = np.array(train_set_x[:cut_pos]), np.array(train_set_y[:cut_pos])
X_val, Y_val = np.array(train_set_x[cut_pos:]), np.array(train_set_y[cut_pos:])

model = Sequential()
model.add(LSTM(128, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True))
model.add(Dropout(0.5))
model.add(LSTM(128, return_sequences=True))
model.add(LSTM(128, return_sequences=False))
model.add(Dropout(0.5))
model.add(Dense(1))
model.compile(loss="mse", optimizer="rmsprop")
callback = EarlyStopping(monitor="loss", patience=10, verbose=1, mode="auto")

model.fit(X_train, Y_train, epochs=5, batch_size=64, validation_data=(X_val, Y_val), callbacks=[callback], shuffle=True)


df_t = pd.read_csv("test_data.csv")
test_data_x = df_t[["LastPrice", "Volume", "BidPrice1",
               "BidVolume1", "AskPrice1", "AskVolume1"]].values
test_data_y = df_t[["MidPrice"]].values

for i in range(len(test_data_x) - 1):
    test_data_x[i + 1][1] = test_data_x[i + 1][1] - test_data_x[i][1]
    if (test_data_x[i + 1][1] < 0):
        test_data_x[i + 1][1] = 0
        
test_data_x = scaler_x.transform(test_data_x)
x_test = []
base_mid_price = []
for i in range(1000):
    x = test_data_x[i * 10 : (i+1) * 10, :]
    y_base = test_data_y[(i + 1) * 10 - 1][0]
    x_test.append(x)
    base.append(y_base)
x_test = np.array(x_test)

res = model.predict(x_test)

with open('result0.csv', 'w', encoding='utf8', newline='') as fout:
    fieldnames = ['caseid','midprice']
    writer = csv.DictWriter(fout, fieldnames = fieldnames)
    writer.writeheader()
    for i in range(142, len(res)):
        writer.writerow({'caseid':str(i+1),'midprice':float(res[i][0] / 1000.0 + base[i])})












