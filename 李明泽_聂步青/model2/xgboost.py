# -*- coding:utf-8 -*-
# 大作业 XGBOOST
import pandas as pd
import numpy as np
import xgboost as xgb
import csv

def is_valid(input_data, i):
    for j in range(9):
        # 不能隔天
        if input_data[i - 1, 0] != input_data[i, 0]:
            return False
        time1 = input_data[i - 1, 1].split(':')
        time2 = input_data[i, 1].split(':')
        # 间隔为3
        if (int(time1[2]) + 3) % 60 != int(time2[2]):
            return False
    return True

def train_data_split(input_data):
    x_train = []
    y_train = []
    for i in range(1, len(input_data) - 31):
        if(is_valid(input_data, i) == True):
            xlist = []
            tmpvolumn = input_data[i - 1: i + 9, 4]
            m = np.mean(tmpvolumn)
            scale = np.std(tmpvolumn, ddof = 1)
            for j in range(9):
                # 数据预处理
                xlist.append(float(input_data[i + j - 1, 2]) - float(input_data[i + 8, 2]))  # 差值
                xlist.append(float(input_data[i + j - 1, 3]) - float(input_data[i + 8, 3]))  # 差值
                xlist.append((float(input_data[i + j - 1, 4]) - m) / scale)   # 标准化
                xlist.append(float(input_data[i + j - 1, 6]))   # no
                xlist.append(float(input_data[i + j - 1, 8]))   # no
            xlist.append((float(input_data[i + 8, 4]) - m) / scale)   
            xlist.append(float(input_data[i + 8, 6]))
            xlist.append(float(input_data[i + 8, 8]))
            x_train.append(xlist)
            mean_y = np.mean(input_data[i + 9: i + 29, 2])
            y_train.append(mean_y - float(input_data[i + 8, 2]))

    x_train = np.array(x_train, ndmin = 2)
    y_train = np.array(y_train)
    return x_train, y_train

def test_data_split(input_data):
    x_test = []
    for i in range(1000):
        xlist = []
        tmpvolumn = input_data[i * 10: i * 10 + 10, 2]
        m = np.mean(tmpvolumn)
        scale = np.std(tmpvolumn, ddof = 1)
        for j in range(9):
            xlist.append(float(input_data[i * 10 + j, 0]) - float(input_data[i * 10 + 9, 0]))
            xlist.append(float(input_data[i * 10 + j, 1]) - float(input_data[i * 10 + 9, 1]))
            xlist.append((float(input_data[i * 10 + j, 2]) - m) / scale)
            xlist.append(float(input_data[i * 10 + j, 4]))
            xlist.append(float(input_data[i * 10 + j, 6]))
        xlist.append((float(input_data[i * 10 + 9, 2]) - m) / scale)
        xlist.append(float(input_data[i * 10 + 9, 4]))
        xlist.append(float(input_data[i * 10 + 9, 6]))
        x_test.append(xlist)
    x_test = np.array(x_test, ndmin = 2)
    return x_test

file = open("train_data.csv")
df = pd.read_csv(file)
rawInput = df.iloc[:, 1:].values
file.close()

x_train, y_train = train_data_split(rawInput)
data_train = xgb.DMatrix(x_train, y_train)

# 开始训练
param = {'max_depth': 4, 'eta': 1, 'objective': 'reg:linear'}
round = 3
watchlist = [(data_train, 'train')]
booster = xgb.train(param, data_train, num_boost_round=round, evals=watchlist)

test_file = open("test_data.csv")
df2 = pd.read_csv(test_file)
testData = df2.iloc[:, 3:].values
test_file.close()

x_test = test_data_split(testData)

# 预测
data_test = xgb.DMatrix(x_test)
y_pre = booster.predict(data_test)

for i in range(1000):
    y_pre[i] = y_pre[i] + float(testData[i * 10 + 9, 0])

y_pre = [list(y_pre)]
y_pre = np.array(y_pre, ndmin = 2)
y_pre = np.transpose(y_pre)

# 写入文件
with open('out.csv','w', newline='') as fout:
    fieldnames = ['caseid','midprice']
    writer = csv.DictWriter(fout,fieldnames = fieldnames)
    writer.writeheader()
    for i in range(142,len(y_pre)):
        writer.writerow({'caseid':str(i+1),'midprice':float(y_pre[i])})
