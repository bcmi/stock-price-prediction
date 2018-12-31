from sklearn.datasets import load_iris
import xgboost as xgb
from xgboost import plot_importance
#from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
import math
import pandas as pd
'''
基于Scikit-learn接口的分类
'''
# 读取文件原始数据
data = []
labels = []
with open("new_data.csv", encoding='UTF-8') as fileObject:
    for line in fileObject:
        line_split = line.split(',')
        data.append(line_split[:])
        labels.append(line_split[0])

length = len(data)
MAX = length-100
X = []
y = []

for cnt in range(MAX): 
    one_instance = []
    for row in range(10):
        for x in data[cnt+row]:
            one_instance.append(float(x))
    X.append(one_instance)
    
    sum = 0
    for i in range(20):
        sum += float(labels[cnt+i+10])
    mean = sum/20
    y.append(mean)
    
X_train = np.array(X)
y_train = np.array(y)
# 训练模型
model = xgb.XGBClassifier(max_depth=20, learning_rate=0.1, n_estimators=160, silent=True, objective='multi:softmax')
model.fit(X_train, y_train)
midprice_dict = {'caseid':[],'MidPrice':[]}
# 读取文件原始数据
for i in range(1000):
    test_data = []
    with open("test_data/minidata"+ str(i)+ ".csv", encoding='UTF-8') as fileObject:
        for line in fileObject:
            line_split = line.split(',')
            test_data.append(line_split[:])

    length = len(test_data)
    X = []
    one_instance = []
    for row in range(10):
        for x in test_data[row + 1]:
            one_instance.append(float(x))
    X.append(one_instance)

    X_test = np.array(X)
    # 对测试集进行预测
    ans = model.predict(X_test)
    if i >= 142:
        midprice_dict['MidPrice'].append(float(ans))
        midprice_dict['caseid'].append(i+1)
        
data_frame = pd.DataFrame(midprice_dict)
data_frame.to_csv('result1.csv',index=False,sep=',')