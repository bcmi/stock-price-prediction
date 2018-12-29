import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from matplotlib import pyplot
from sklearn.preprocessing import LabelEncoder,MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Dropout
from keras.layers import LSTM
from keras.callbacks import EarlyStopping
from numpy import concatenate
from math import sqrt
from keras.regularizers import l2
from sklearn.preprocessing import scale
from keras import backend as K

def rmse(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))

def read_raw():
    dataset = pd.read_csv('train_data.csv',  parse_dates = [['Date', 'Time']], index_col=0)
    dataset.drop('Unnamed: 0', axis=1, inplace=True)

    values = dataset.values
    ams = []
    zero = [0,0,0,0,0,0,0]
    tmp = [0,0,0,0,0,0,0]
    index = []
    sum = 0
    for i in range(20):
        sum += values[i,0]
    
    for i in range(len(values)-20):
        #values[i,6] = values[i,4]-values[i,6]
        #values[i,4] = values[i,3]-values[i,5]
        
        if (i == 0):
            for j in range(len(index)):
                zero[index[j]] = values[i,index[j]]
                values[i,index[j]] = 0
            ams.append(0)
        else:
            for j in range(len(index)):
                tmp[index[j]] = zero[index[j]]
                zero[index[j]] = values[i,index[j]]
                values[i,index[j]] -= tmp[index[j]]
            sum = sum - values[i-1, 0] + values[i+19, 0]
            ams.append(sum/20.0 - values[i-1, 0])
        
    values = values[1:-20]
    ams.pop(0)
    indexes = [0]
    for i in range(len(values)-1):
        if (values[i, 2] > values[i+1, 2]):
            indexes.append(i+1)
    indexes.append(len(values))

    for i in range(len(indexes)-1):
        v = values[indexes[i]:indexes[i+1]]
        a = ams[indexes[i]:indexes[i+1]]
        d = pd.DataFrame(v)
        a = pd.DataFrame(a)
        a.to_csv('ams%s.csv'%str(i),index = False,header = False)
        d.to_csv('dataset%s.csv'%str(i),index = False,header = False)

    return len(indexes) - 1

def read_test():
    dataset = pd.read_csv('test_data.csv',  parse_dates = [['Date', 'Time']], index_col=0)
    dataset.drop('Unnamed: 0', axis=1, inplace=True)

    values = dataset.values
    zero = [0,0,0,0,0,0,0]
    tmp = [0,0,0,0,0,0,0]
    index = []
    
    for i in range(len(values)):
        #values[i,6] = values[i,4]-values[i,6]
        #values[i,4] = values[i,3]-values[i,5]
        
        if (i == 0):
            for j in range(len(index)):
                zero[index[j]] = values[i,index[j]]
                values[i,index[j]] = 0
        else:
            for j in range(len(index)):
                tmp[index[j]] = zero[index[j]]
                zero[index[j]] = values[i,index[j]]
                values[i,index[j]] -= tmp[index[j]]

    dataset = pd.DataFrame(values)

    print(dataset.shape)

    dataset.to_csv('testset.csv',index = False,header = False)

def test_data_transform():
    testset = pd.read_csv('testset.csv', header=None, index_col=None)
    test = testset.values
    scaler = MinMaxScaler(feature_range=(-1, 1))
    predict_X = []
    for i in range(0,1000):
        x = test[10*i:10*i+10,:]
        x = scaler.fit_transform(x)
        #x = x.flatten()
        predict_X.append(x)

    predict_X = np.array(predict_X)
    #predict_X = pd.DataFrame(predict_X)
    #predict_X.to_csv('predict_X.csv',index = False, header = False)
    np.save('predict_X.npy', predict_X)
    return predict_X


def train_data_transform(fileindex):
    dataset = pd.read_csv('dataset%s.csv'%str(fileindex), header=None, index_col=None)
    ams = pd.read_csv('ams%s.csv'%str(fileindex), header=None, index_col=None)
    
    values = dataset.values
    ams = ams.values
    
    scaler = MinMaxScaler(feature_range=(-1, 1))
    train_X = []
    train_Y = []
    for i in range(0,len(values)-10):
        j=0
        x = values[i+j:i+10+j,:]
        x = scaler.fit_transform(x)
        #x = x.flatten()
        train_X.append(x)
        train_Y.append(ams[i+10+j][0])

    train_X = np.array(train_X)
    train_Y = np.array(train_Y)

    return train_X, train_Y

def total_data_transform(number):
    train_X, train_Y = train_data_transform(0)
    train_X = list(train_X)
    train_Y = list(train_Y)
    for i in range(1, number):
        x, y = train_data_transform(i)
        x = list(x)
        y = list(y)
        train_X = train_X + x
        train_Y = train_Y + y

    train_X = np.array(train_X)
    train_Y = np.array(train_Y)
    
    state = np.random.get_state()
    np.random.shuffle(train_X)
    np.random.set_state(state)
    np.random.shuffle(train_Y)
    '''
    train_X = pd.DataFrame(train_X)
    train_Y = pd.DataFrame(train_Y)
    train_X.to_csv('train_X.csv',index = False, header = False)
    train_Y.to_csv('train_Y.csv',index = False, header = False)
    '''
    n = 300000
    test_X = train_X[n:]
    test_Y = train_Y[n:]
    train_X = train_X[:n]
    train_Y = train_Y[:n]
    np.save('train_X.npy', train_X)
    np.save('train_Y.npy', train_Y)
    np.save('test_X.npy', test_X)
    np.save('test_Y.npy', test_Y)
    return train_X, train_Y, test_X, test_Y

def fit_network(train_X, train_Y, test_X, test_Y, predict_X):
    
    model = Sequential()
    model.add(LSTM(128, activation='tanh', input_shape=(train_X.shape[1], train_X.shape[2]), return_sequences=True))
    model.add(LSTM(128, activation='tanh'))
    model.add(Dense(60, activation = 'linear', kernel_regularizer=l2(0.001)))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation = 'linear'))
    model.compile(loss=rmse, optimizer='adam')
    early_stopping = EarlyStopping(monitor='val_loss', patience=2, verbose=2, mode='min')
    history = model.fit(train_X, train_Y, epochs=3, batch_size=128, validation_data=(test_X, test_Y), verbose=1, shuffle=True, callbacks=[early_stopping])
    model.save('model.h5')
    '''
    model = Sequential()
    model.add(LSTM(300, activation="tanh", input_shape=(train_X.shape[1], train_X.shape[2])))
    model.add(Dropout(0.3))
    model.add(Dense(1, activation="linear"))
    
    model.compile(loss=rmse, optimizer='adam')
    early_stopping = EarlyStopping(monitor='val_loss', patience=2, verbose=2, mode='min')
    history = model.fit(train_X, train_Y, epochs=5, batch_size=128, validation_data=(test_X, test_Y), shuffle=True, verbose=1, callbacks=[early_stopping])
    model.save('model.h5')
    '''
    dataset = pd.read_csv('test_data.csv',  parse_dates = [['Date', 'Time']], index_col=0)
    dataset.drop('Unnamed: 0', axis=1, inplace=True)
    datavalue = dataset.values
    predict_Y = model.predict(predict_X)
    print(predict_Y)
    data = {'caseid':[i+1 for i in range(1000)], 'midprice':[datavalue[9+10*i, 0]+predict_Y[i][0] for i in range(1000)]}
    df = pd.DataFrame(data)
    df = df[142:]
    fileindex = 0
    df.to_csv("rest%s.csv"%str(fileindex), index = False)

n = read_raw()
read_test()
test_data_transform()
total_data_transform(n)
train_X = np.load('train_X.npy')
train_Y = np.load('train_Y.npy')
test_X = np.load('test_X.npy')
test_Y = np.load('test_Y.npy')
predict_X = np.load('predict_X.npy')
fit_network(train_X, train_Y, test_X, test_Y, predict_X)