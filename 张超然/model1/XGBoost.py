import pandas as pd
import numpy as np
import xgboost as xgb
import os
#os.chdir("E:\Artificial Intelligence")
f = open("train_data.csv")
df = pd.read_csv(f)
train_in = df.iloc[:, 1:].values
f.close()

def is_valid(data, i):
    for j in range(9):
        if data[i - 1, 0] != data[i, 0]:
            return False
        time1 = data[i - 1, 1].split(':')
        time2 = data[i, 1].split(':')
        if (int(time1[2]) + 3) % 60 != int(time2[2]):
            return False
    return True

def train_split(train_input):
    x_train = []
    y_train = []
    for i in range(1, len(train_input) - 31):
        if(is_valid(train_input, i)):
            tmpx = []
            tmpvolume = train_input[i - 1: i + 9, 4]
            m = np.mean(tmpvolume)
            scale = np.std(tmpvolume, ddof = 1)
            for j in range(9):
                tmpx.append(float(train_input[i + j - 1, 2]) - float(train_input[i + 8, 2]))
                tmpx.append(float(train_input[i + j - 1, 3]) - float(train_input[i + 8, 3]))
                tmpx.append((float(train_input[i + j - 1, 4]) - m) / scale)
                tmpx.append(float(train_input[i + j - 1, 5]))
                tmpx.append(float(train_input[i + j - 1, 7]))
                tmpx.append(float(train_input[i + j - 1, 9]))
            x_train.append(tmpx)
            tmpy = np.mean(train_input[i + 9: i + 29, 2])
            y_train.append(tmpy - float(train_input[i + 8, 2]))
        #if (i % 1000 == 0):
            #print(i)
    x_train = np.array(x_train, ndmin = 2)
    y_train = np.array(y_train)
    return x_train, y_train

x_train, y_train = train_split(train_in)
#print(len(x_train), len(y_train))
train_data = xgb.DMatrix(x_train, y_train)

param = {'max_depth': 4, 'eta': 1, 'objective': 'reg:linear'}
n_round = 3
watchlist = [(train_data, 'train')]
booster = xgb.train(param, train_data, num_boost_round=n_round, evals=watchlist)

f2 = open("test_data.csv")
df2 = pd.read_csv(f2)
test_in = df2.iloc[:,3:].values
f2.close()

def test_split(test_input):
    x_test = []
    for i in range(1000):
        tmpx = []
        tmpvolume = test_input[i * 10: i * 10 + 10, 2]
        m = np.mean(tmpvolume)
        scale = np.std(tmpvolume, ddof = 1)
        for j in range(9):
            tmpx.append(float(test_input[i * 10 + j, 0]) - float(test_input[i * 10 + 9, 0]))
            tmpx.append(float(test_input[i * 10 + j, 1]) - float(test_input[i * 10 + 9, 1]))
            tmpx.append((float(test_input[i * 10 + j, 2]) - m) / scale)
            tmpx.append(float(test_input[i * 10 + j, 3]))
            tmpx.append(float(test_input[i * 10 + j, 5]))
            tmpx.append(float(test_input[i * 10 + j, 7]))
        x_test.append(tmpx)
    x_test = np.array(x_test, ndmin = 2)
    return x_test

x_test = test_split(test_in)

data_test = xgb.DMatrix(x_test)
res = booster.predict(data_test)
for i in range(1000):
    res[i] = res[i] + float(test_in[i * 10 + 9, 0])
res = [list(res)]
res = np.array(res, ndmin = 2)
res = np.transpose(res)

pre= pd.DataFrame(columns = ['midprice'], data = res)

pre.to_csv("out.csv")