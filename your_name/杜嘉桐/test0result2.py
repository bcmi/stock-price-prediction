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


def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]     # train中430039
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


# load dataset
dataset = read_csv('train.csv', header=0, index_col=0)
values = dataset.values
print(values.shape)
submit_dataset = read_csv('test.csv', header=0, index_col=0)    # 加载测试集
submit_values = submit_dataset.values    # 测试集数据矩阵
print(submit_values.shape)
values = concatenate((values, submit_values), axis=0)    # 前430039行是train数据，后8580是需要预测的数据
print(values.shape)
####print(values)

# ensure all data is float
values = values.astype('float32')
# normalize features 归一化  测试时新数据加入可能要重新计算
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)
####print(type(scaled))
print(scaled.shape)

# frame as supervised learning
reframed = series_to_supervised(scaled, 10, 1)  # 前十行没有了
# drop columns we don't want to predict
reframed.drop(reframed.columns[[71, 72, 73, 74, 75, 76]], axis=1, inplace=True)
print(reframed.head())

############################################################ 以上数据处理得到reframed，包含输入和输出 #####################################

# split into train and test sets
values = reframed.values
print('++++++++++++++++++++++++++++++++++++++++++')
print(values.shape)
n_train = 400000
train = values[:n_train, :]
test = values[n_train:430039, :]
submit = values[430039:, :]
# split into input and outputs
train_X, train_y = train[:, :-1], train[:, -1]
test_X, test_y = test[:, :-1], test[:, -1]
submit_X, submit_y = submit[:, :-1], submit[:, -1]


########################## 需要做一个submit_X ############################################
# submit_X = values[430039, :]
# print(type(values[430039, :]))
# print(values[430039, :].shape)
# print(values[430039, :])
# for i in range(1, 858):
#     submit_X = concatenate((submit_X, values[430039+10*i-1, :]))
# print('/*/*/*/*/*/*/*/*/*/*/*  submit_X  /*/*/*/*/*/*/*/*/***/*/*')
# print(submit_X.shape)
# print(submit_X)


# reshape input to be 3D [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
submit_X = submit_X.reshape(submit_X.shape[0], 1, submit_X.shape[1])
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape, submit_X.shape, submit_y.shape)
    #(400000, 1, 70)   (400000,)   (30029, 1, 70) (30029,)

############################################################## 数据分割结束，下面设计LSTM网络 #####################################

# design network
model = Sequential()
model.add(LSTM(500, input_shape=(train_X.shape[1], train_X.shape[2]))) # 50
model.add(Dense(1))
model.compile(loss='mae', optimizer='adam')
# fit network
history = model.fit(train_X, train_y, epochs=150, batch_size=72, validation_data=(test_X, test_y), verbose=2, shuffle=False)  # epochs迭代次数
# plot history
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()

def rmse(X, y):
    yhat = model.predict(X)
    X = X.reshape((X.shape[0], X.shape[2]))  # test_X (30029, 70)
    # invert scaling for forecast
    inv_yhat = concatenate((yhat, X[:, 1:7]), axis=1)  # 水平连接
    print('******************** inv_yhat *************')
    print(type(inv_yhat))
    print(inv_yhat.shape)
    inv_yhat = scaler.inverse_transform(inv_yhat)  # 反归一化得到真值
    inv_yhat = inv_yhat[:, 0]  # 第一列是预测值
    print('预测值长度：', len(inv_yhat))
    print('预测值：')
    print(inv_yhat)
    # 提取858个结果值，保存到result.csv
    result_yhat = []
    for i in range(858):
        if i == 857:
            result_yhat.append(inv_yhat[8569])
            break
        result_yhat.append(inv_yhat[i*10])
    caseid = []
    for i in range(143, 1001):
        caseid.append(i)
    result_data = []
    result_data.append(caseid)
    result_data.append(result_yhat)
    result_data = np.array(result_data)
    result_data = DataFrame(result_data.transpose(), columns=['caseid', 'midprice'])
    result_data.to_csv('result.csv')
    # invert scaling for actual
    y = y.reshape((len(y), 1))
    inv_y = concatenate((y, X[:, 1:7]), axis=1)
    inv_y = scaler.inverse_transform(inv_y)
    inv_y = inv_y[:, 0]
    # calculate RMSE
    rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
    print('Test RMSE: %.3f' % rmse)


# test_y 或 submit_y
rmse(submit_X, submit_y)

# # calculate RMSE
# yhat = model.predict(test_X)
# print('******************** yhat *************')
# print(type(yhat))
# print(yhat.shape)
# test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))      # test_X (30029, 70)
# # invert scaling for forecast
# inv_yhat = concatenate((yhat, test_X[:, 1:7]), axis=1)  # 水平连接
# print('******************** inv_yhat *************')
# print(type(inv_yhat))
# print(inv_yhat.shape)
# inv_yhat = scaler.inverse_transform(inv_yhat)  # 反归一化得到真值
# inv_yhat = inv_yhat[:, 0]  # 第一列是预测值
# print('预测值长度：', len(inv_yhat))
# print('预测值：')
# print(inv_yhat)
# # invert scaling for actual
# # test_y 或 submit_y
# test_y = test_y.reshape((len(test_y), 1))
# inv_y = concatenate((test_y, test_X[:, 1:7]), axis=1)
# inv_y = scaler.inverse_transform(inv_y)
# inv_y = inv_y[:, 0]
# # calculate RMSE
# rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
# print('Test RMSE: %.3f' % rmse)
