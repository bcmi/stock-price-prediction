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
        x = x.flatten()
        predict_X.append(x)

    predict_X = np.array(predict_X)
    predict_X = pd.DataFrame(predict_X)
    predict_X.to_csv('predict_X.csv',index = False, header = False)

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
        x = x.flatten()
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

    train_X = pd.DataFrame(train_X)
    train_Y = pd.DataFrame(train_Y)
    train_X.to_csv('train_X.csv',index = False, header = False)
    train_Y.to_csv('train_Y.csv',index = False, header = False)

n = read_raw()
read_test()
test_data_transform()
total_data_transform(n)