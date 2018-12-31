#!/usr/bin/env python
# -*- coding: utf-8 -*-


import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
import numpy as np
from xgboost import plot_importance
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.feature_selection import SelectFromModel
from sklearn.svm import LinearSVC
from sklearn.linear_model import Lasso


def loadDataset(filePath):
    df = pd.read_csv(filepath_or_buffer=filePath)
    return df

def trainandTest(X_train, y_train, X_test, filePath):
    preserve = pd.read_csv("preserve.csv")

    other_params = {'learning_rate': 0.05, 'n_estimators': 200, 'max_depth': 5, 'min_child_weight': 1, 'seed': 0,
      'subsample': 0.8, 'colsample_bytree': 0.8, 'gamma': 0, 'reg_alpha': 0.03, 'reg_lambda': 0.02, "objective" :"reg:linear"}

    model = xgb.XGBRegressor(**other_params)
    model.fit(X_train, y_train)
    
    ans = model.predict(X_test)

    # plot_importance(model)
    # plt.show()

    ans_len = len(ans)
    casenum = 143
    data_arr = []
    file_out = open('xgbresult.csv','w')
    file_out.write('caseid,midprice\n')
    for row in range(ans_len):
        midprice = preserve.iloc[9 + row * 10]["MidPrice"]
        data_arr.append([int(casenum), ans[row] + midprice])
        file_out.write(str(casenum)+','+str(ans[row] + midprice)+'\n')
        casenum = casenum + 1
  
    file_out.close()

if __name__ == '__main__':
    trainFilePath = 'xgbtrain.csv'
    testFilePath = 'xgbtest.csv'
    
    data = pd.read_csv(filepath_or_buffer=trainFilePath)
    data_num = len(data)

    datavalue = data.iloc[:, 1:-1].values

    X_train = []
    for row in range(data_num):
        tmp_list = datavalue[row]
        X_train.append(datavalue[row])

    y_train = data[:data_num]["AverageDiff"].values
    
    # 处理测试集
    tdata = pd.read_csv(filepath_or_buffer=testFilePath)
    data_num = len(tdata)
    X_test = []
    testvalue = tdata.iloc[:, 1:].values
    for row in range(0, data_num):
        tmp_list = testvalue[row]
        X_test.append(tmp_list)


    sk = SelectKBest(f_regression, k= 35)
    X_train = sk.fit_transform(X_train, y_train)
    X_test = sk.transform(X_test)

    trainandTest(X_train, y_train, X_test, testFilePath)
