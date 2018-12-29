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
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# 使用30%的GPU memory
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.4
session = tf.Session(config=config)
# 设置session
KTF.set_session(session)

random.seed(time.time())

time_steps = 10
train_ratio = 0.99

# 读取训练集
df = pd.read_csv("train_data.csv")
train_data_x = df[["LastPrice", "Volume", "BidPrice1",
           "BidVolume1", "AskPrice1", "AskVolume1"]].values
train_data_y = df[["MidPrice"]].values
date = df[["Date"]].values

# 分割不同的日期
date_stamp = [0]
for i in range(len(date) - 1):
	if (date[i + 1] != date[i]):
		date_stamp.append(i + 1)
date_stamp.append(len(date) - 1)

# 读取要预测的数据
df_t = pd.read_csv("test_data.csv")
test_data_x = df_t[["LastPrice", "Volume", "BidPrice1",
               "BidVolume1", "AskPrice1", "AskVolume1"]].values
test_data_y = df_t[["MidPrice"]].values

for i in range(len(train_data_x) - 1):
    train_data_x[i + 1][1] = train_data_x[i + 1][1] - train_data_x[i][1]
    if (train_data_x[i + 1][1] < 0):
        train_data_x[i + 1][1] = 0
for i in range(len(test_data_x) - 1):
    test_data_x[i + 1][1] = test_data_x[i + 1][1] - test_data_x[i][1]
    if (test_data_x[i + 1][1] < 0):
        test_data_x[i + 1][1] = 0

# MinMax归一化数据
scaler_x = preprocessing.MinMaxScaler()
train_data_x = scaler_x.fit_transform(train_data_x)
test_data_x = scaler_x.transform(test_data_x)

# Z-Score归一化数据
# scaler_x_std = preprocessing.StandardScaler().fit(train_data_x)
# scaler_x_std.transform(test_data_x)

train_set_x, train_set_y = [], []
for d in range(len(date_stamp) - 1):
	for i in range(date_stamp[d], date_stamp[d + 1] - 3 * time_steps):
	    x = train_data_x[i : i + time_steps, :]
	    y = [np.mean(train_data_y[i + time_steps : i + 3 * time_steps])] # 取midprice
	    y = y - train_data_y[i + time_steps - 1][0] # 增量
	    y = np.array(y) * 1000
	    train_set_x.append(x.tolist())
	    train_set_y.append(y.tolist())

# cut_pos = int(len(train_set_x) * train_ratio)
cut_pos = int(len(train_set_x) * (1 - train_ratio))
# 拆分数据
# X_train, Y_train = np.array(train_set_x[ : cut_pos]), np.array(train_set_y[ : cut_pos])
# X_val, Y_val = np.array(train_set_x[cut_pos : ]), np.array(train_set_y[cut_pos : ])
X_train, Y_train = np.array(train_set_x[cut_pos : ]), np.array(train_set_y[cut_pos : ])
X_val, Y_val = np.array(train_set_x[ : cut_pos]), np.array(train_set_y[ : cut_pos])

# 建立模型
model = Sequential()
model.add(LSTM(128, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True))
model.add(Dropout(0.4))
# model.add(LSTM(100, return_sequences=True))
model.add(LSTM(128, return_sequences=True))
model.add(LSTM(128, return_sequences=False))
model.add(Dropout(0.4))
model.add(Dense(1))
r = RMSprop(lr=0.001, rho=0.9, epsilon=1e-06, decay=0.9995)
a = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, decay=0.9998)
model.compile(loss="mse", optimizer="adam")
model.summary()

callback = EarlyStopping(monitor="loss", patience=10, verbose=1, mode="auto")
# 训练模型
# model = load_model('saved_model.h5')
# for i in range(5):
#     model.fit(X_train, Y_train, epochs=1, batch_size=128, validation_data=(X_val, Y_val), callbacks=[callback], shuffle=True)
#     model.reset_states()

# 用于再训练
# model = load_model('saved_model.h5')

model.fit(X_train, Y_train, epochs=3, batch_size=32, validation_data=(X_val, Y_val), callbacks=[callback], shuffle=True)
model.save('saved_model.h5')

print("Predicting...")
# model = load_model('saved_model.h5')
test_cases = []
base_mid_price = []
for i in range(1000):
    x = test_data_x[i * time_steps : (i+1) * time_steps, :]
    y_base = test_data_y[(i+1) * time_steps - 1][0] # 用于加上增量计算最终结果
    test_cases.append(x)
    base_mid_price.append(y_base)
test_cases = np.array(test_cases)

res = model.predict(test_cases)

with open('res.csv', 'w', encoding='utf8', newline='') as fout:
    fieldnames = ['caseid','midprice','diff']
#     fieldnames = ['caseid','midprice']
    writer = csv.DictWriter(fout, fieldnames = fieldnames)
    writer.writeheader()
    for i in range(142, len(res)):
        writer.writerow({'caseid':str(i+1),'midprice':float(res[i][0] / 1000.0 + base_mid_price[i]), 'diff': float(res[i][0] / 1000.0)})
#         writer.writerow({'caseid':str(i+1),'midprice':float(res[i][0] / 100.0 + base_mid_price[i])})
print("Done.")