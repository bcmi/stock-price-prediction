# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
from math import sqrt
import numpy as np
from numpy import concatenate
from matplotlib import pyplot
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

dataset = read_csv('train_data.csv', header=0, index_col=0)
dataset.drop('Date', axis=1, inplace=True)
dataset.drop('Time', axis=1, inplace=True)
print(dataset.head(5))
dataset.to_csv('train.csv')
dataset = read_csv('test_data.csv', header=0, index_col=0)
dataset=dataset.loc[430039+50*142:]
dataset.drop('Date', axis=1, inplace=True)
dataset.drop('Time', axis=1, inplace=True)
print(dataset.head(5))
dataset.to_csv('test.csv')

def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    agg = concat(cols, axis=1)
    agg.columns = names
    if dropnan:
        agg.dropna(inplace=True)
    return agg



dataset = read_csv('train.csv', header=0, index_col=0)
values = dataset.values
print(values.shape)
submit_dataset = read_csv('test.csv', header=0, index_col=0)
submit_values = submit_dataset.values
print(submit_values.shape)
values = concatenate((values, submit_values), axis=0)
print(values.shape)

values = values.astype('float32')
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)
print(scaled.shape)

reframed = series_to_supervised(scaled, 10, 1)
reframed.drop(reframed.columns[[71, 72, 73, 74, 75, 76]], axis=1, inplace=True)
print(reframed.head())

values = reframed.values
n_train = 200000
train = values[:n_train, :]
test = values[n_train:430039, :]
submit = values[430039:, :]
train_X, train_y = train[:, :-1], train[:, -1]
test_X, test_y = test[:, :-1], test[:, -1]
submit_X, submit_y = submit[:, :-1], submit[:, -1]

train_X = train_X.reshape((train_X.shape[0], 10, int(train_X.shape[1]//10)))
test_X = test_X.reshape((test_X.shape[0], 10, int(test_X.shape[1]//10)))
submit_X = submit_X.reshape(submit_X.shape[0], 10, int(submit_X.shape[1]//10))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape, submit_X.shape, submit_y.shape)

model = Sequential()
model.add(LSTM(64, input_shape=(train_X.shape[1], train_X.shape[2]))) # 50
model.add(Dense(1))
model.compile(loss='mse', optimizer='adam')
history = model.fit(train_X, train_y, epochs=20, batch_size=720, validation_data=(test_X, test_y), verbose=2, shuffle=False)  # epochs迭代次数

def rmse(X, y):
    yhat = model.predict(X)
    X = X.reshape((X.shape[0], X.shape[2]*10))
    inv_yhat = concatenate((yhat, X[:, 1:7]), axis=1)
    print(type(inv_yhat))
    print(inv_yhat.shape)
    inv_yhat = scaler.inverse_transform(inv_yhat)
    inv_yhat = inv_yhat[:, 0]
    print('预测值长度：', len(inv_yhat))
    print('预测值：')
    print(inv_yhat)
    result_yhat = []
    for i in range(858):
        if i == 857:
            result_yhat.append(inv_yhat[8569])
            break
        result_yhat.append(inv_yhat[i*10])
    caseid = []
    for i in range(143, 1001):
        caseid.append(int(i))
    result_data = []
    result_data.append(caseid)
    result_data.append(result_yhat)
    result_data = np.array(result_data)
    result_data = DataFrame(result_data.transpose(), columns=['caseid', 'midprice'])
    result_data['caseid']=int(result_data['caseid'])
    result_data=result_data.set_index('caseid')
    result_data.to_csv('result.csv')
    y = y.reshape((len(y), 1))
    inv_y = concatenate((y, X[:, 1:7]), axis=1)
    inv_y = scaler.inverse_transform(inv_y)
    inv_y = inv_y[:, 0]
    rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
    print('Test RMSE: %.3f' % rmse)


rmse(submit_X, submit_y)

