import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb

f = open("washed_train_data.csv")
df = pd.read_csv(f)
rawInput = df.iloc[:, 1:].values
f.close()

def find_validate(inputdata, i):
    for j in range(9):
        if inputdata[i - 1, 0] != inputdata[i, 0]:
            return False
        time1 = inputdata[i - 1, 1].split(':')
        time2 = inputdata[i, 1].split(':')
        if (int(time1[2]) + 3) % 60 != int(time2[2]):
            return False
    return True

def train_data_split(inputdata):
    x_train = []
    y_train = []
    for i in range(1, len(inputdata) - 31):
        if(find_validate(inputdata, i) == True):
            tmpx = []
            tmpvolumn = inputdata[i - 1: i + 9, 4]
            m = np.mean(tmpvolumn)
            scale = np.std(tmpvolumn, ddof = 1)
            for j in range(9):
                tmpx.append(float(inputdata[i + j - 1, 2]) - float(inputdata[i + 8, 2]))
                tmpx.append(float(inputdata[i + j - 1, 3]) - float(inputdata[i + 8, 3]))
                tmpx.append((float(inputdata[i + j - 1, 4]) - m) / scale)
                tmpx.append(float(inputdata[i + j - 1, 6]))
                tmpx.append(float(inputdata[i + j - 1, 8]))
            tmpx.append((float(inputdata[i + 8, 4]) - m) / scale)
            tmpx.append(float(inputdata[i + 8, 6]))
            tmpx.append(float(inputdata[i + 8, 8]))
            x_train.append(tmpx)
            tmpy = np.mean(inputdata[i + 9: i + 29, 2])
            y_train.append(tmpy - float(inputdata[i + 8, 2]))
        if i % 1000 == 0:
            print(i)
    x_train = np.array(x_train, ndmin = 2)
    y_train = np.array(y_train)
    return x_train, y_train

x_train, y_train = train_data_split(rawInput)
print(len(x_train), len(y_train))
data_train = xgb.DMatrix(x_train, y_train)

param = {'max_depth': 4, 'eta': 1, 'objective': 'reg:linear'}
n_round = 3
watchlist = [(data_train, 'train')]
booster = xgb.train(param, data_train, num_boost_round=n_round, evals=watchlist)

f2 = open("test_data.csv")
df2 = pd.read_csv(f2)
testData = df2.iloc[:,3:].values
f2.close()

def test_data_split(inputdata):
    x_test = []
    for i in range(1000):
        tmpx = []
        tmpvolumn = inputdata[i * 10: i * 10 + 10, 2]
        m = np.mean(tmpvolumn)
        scale = np.std(tmpvolumn, ddof = 1)
        for j in range(9):
            tmpx.append(float(inputdata[i * 10 + j, 0]) - float(inputdata[i * 10 + 9, 0]))
            tmpx.append(float(inputdata[i * 10 + j, 1]) - float(inputdata[i * 10 + 9, 1]))
            tmpx.append((float(inputdata[i * 10 + j, 2]) - m) / scale)
            tmpx.append(float(inputdata[i * 10 + j, 4]))
            tmpx.append(float(inputdata[i * 10 + j, 6]))
        tmpx.append((float(inputdata[i * 10 + 9, 2]) - m) / scale)
        tmpx.append(float(inputdata[i * 10 + 9, 4]))
        tmpx.append(float(inputdata[i * 10 + 9, 6]))
        x_test.append(tmpx)
    x_test = np.array(x_test, ndmin = 2)
    return x_test

x_test = test_data_split(testData)

data_test = xgb.DMatrix(x_test)
y_pre = booster.predict(data_test)
for i in range(1000):
    y_pre[i] = y_pre[i] + float(testData[i * 10 + 9, 0])
y_pre = [list(y_pre)]
y_pre = np.array(y_pre, ndmin = 2)
y_pre = np.transpose(y_pre)

pre_save = pd.DataFrame(columns = ['midprice'], data = y_pre)
p = 0
while(True):
    try:
        pre_save.to_csv("myRes" + str(p) + ".csv")
        break
    except:
        p = p + 1
