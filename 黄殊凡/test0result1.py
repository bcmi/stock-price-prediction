from math import sqrt
import numpy as np
from numpy import concatenate
from matplotlib import pyplot
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM


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
print(dataset.head(5))

values = values.astype('float32')
scaler=[]
for i in range(0,5):
    p=[max(values[:,i]),min(values[:,i])]
    scaler.append(p.copy())
scaled=values.copy()
for i in range(len(values)):
    for j in [1,3]:
        scaled[i,j]=(scaled[i,j]-scaler[j][1])/(scaler[j][0]-scaler[j][1])



reframed = series_to_supervised(scaled, 10, 1)
for i in [0,2,4]:
    for j in [0,1,2,3,4,5,6,7,8,10]:
        reframed[reframed.columns[i+5*j]]=(reframed[reframed.columns[i+5*j]]*10.0-reframed[reframed.columns[i+5*9]]*10.0)
reframed.drop(reframed.columns[[4,9,14,19,24,29,34,39,44,45,47,49,50,51,52,53]], axis=1, inplace=True)
#reframed.to_csv('fuck.csv')
values = reframed.values
n_train =300000
t_test = values[:430019, :]
train = t_test[:n_train, :]
test = t_test[n_train:430019, :]
submit = values[430019:, :]
# split into input and outputs
train_X, train_y = train[:, :-1], train[:, -1]
test_X, test_y = test[:, :-1], test[:, -1]
submit_X, submit_y = submit[:, :-1], submit[:, -1]


train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
submit_X = submit_X.reshape(submit_X.shape[0], 1, submit_X.shape[1])
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape, submit_X.shape, submit_y.shape)
print(train_X)
print(train_y)

model = Sequential()
model.add(LSTM(64, input_shape=(train_X.shape[1], train_X.shape[2]),dropout=0.5,return_sequences=True))
model.add(LSTM(100, return_sequences=False))
model.add(Dense(1))
model.compile(loss='mse', optimizer='adam')
history = model.fit(train_X, train_y, epochs=50, batch_size=2000, validation_data=(test_X, test_y), verbose=2, shuffle=False)
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()

def rmse(X, y):
    yhat = model.predict(X)
    X = X.reshape((X.shape[0], X.shape[2]))
    print(X.shape)
    print(yhat)
    print(scaler)
    inv_yhat=[]
    for i in range(len(yhat)):
        inv_yhat.append(yhat[i][0])
    print(type(inv_yhat))
    print('预测值长度：', len(inv_yhat))
    print('预测值：')
    print(inv_yhat)
    n=read_csv('test1.csv')
    n=n.values
    print(n)
    result_yhat = []
    for i in range(858):
        if i == 857:
            result_yhat.append(inv_yhat[8569]/10+n[i][1])
            break
        result_yhat.append(inv_yhat[i*10]/10+n[i][1])
    caseid = []
    for i in range(143, 1001):
        caseid.append(i)
    result_data = []
    result_data.append(caseid)
    result_data.append(result_yhat)
    result_data = DataFrame({'caseid':caseid, 'midprice':result_yhat})
    result_data.set_index('caseid',inplace=True)
    result_data.to_csv('result.csv')


rmse(submit_X, submit_y)
