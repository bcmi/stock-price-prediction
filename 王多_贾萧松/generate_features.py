import pandas as pd
from sklearn import preprocessing
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn import metrics
import time
import os,sys
import csv
from sklearn.preprocessing import PolynomialFeatures
import operator
from xgboost import plot_importance

train_file = "train_data_2.csv"
train_df = pd.read_csv(train_file)
column_name = list(train_df.columns)
tmp = column_name.copy() 
for i in range(len(column_name)):
    tmp[i] += "(t-0)"
train_df.columns = tmp


#create train_data
lags = range(1,30)

train_df = train_df.assign(**{
    '{}(t-{})'.format(col[:-5], t): train_df[col].shift(t)
    for t in lags
    for col in train_df
})

#drop None
train_df.dropna(axis=0, how='any', inplace=True)
#delete 0-19 feature
deleted_name = []
average_name = []
column_name.remove("MidPrice")
for i in range(0, 20):
    for name in column_name:
        deleted_name.append('{}(t-{})'.format(name, i))
    average_name.append('MidPrice(t-{})'.format(i))
column_name.append("MidPrice")
#calculate target
train_df["target"] = train_df[average_name].mean(axis=1)
deleted_name = deleted_name + average_name
train_df.drop(deleted_name, axis=1, inplace=True)


####time difference data
## Mid' = mid(t)/mid(t-1)-1
dif_name = ["Volume", "BidPrice1", "AskPrice1", "LastPrice", "BidVolume1", "AskVolume1"]
for i in range(21,30):
    train_df['RelMid(t-{})'.format(i)] = (train_df['MidPrice(t-{})'.format(i)]/train_df['MidPrice(t-{})'.format(i-1)])
    for name in dif_name:
        train_df['{}Dif(t-{})'.format(name, i)] = train_df['{}(t-{})'.format(name, i)] - train_df['{}(t-{})'.format(name, i-1)]
    


subtract = ["MidPrice", "LastPrice", "BidPrice1", "AskPrice1"]
# # #get meansubtract
mean_dict = {}
for name in subtract:
    mean_dict[name] = []
    for i in range(20, 30):
        mean_dict[name].append('{}(t-{})'.format(name, i))
for key, value in mean_dict.items():
    train_df[key] = train_df[value].mean(axis=1)

subtract = ["MidPrice", "LastPrice", "BidPrice1", "AskPrice1"]
for name in subtract:
    for i in range(20,30):
        train_df['{}(t-{})'.format(name, i)] = train_df['{}(t-{})'.format(name, i)]-train_df[name]#train_df[name]#

train_df["target"] = train_df["target"] - train_df["MidPrice"]
train_df.drop(subtract, axis=1, inplace=True)


tmp = ["day_hour", "weekday"]
#one-hot
one_hot = []
for name in tmp:
    for i in range(21,30):
        one_hot.append('{}(t-{})'.format(name, i))
train_df.drop(one_hot, axis=1, inplace=True)


dummies = pd.get_dummies(train_df['day_hour(t-20)'], prefix="day_hour1")
train_df = train_df.loc[:,train_df.columns!="day_hour(t-20)"].join(dummies)
dummies = pd.get_dummies(train_df['weekday(t-20)'], prefix="weekday1")
train_df = train_df.loc[:,train_df.columns!="weekday(t-20)"].join(dummies)


train_df.to_csv("train_data_3.csv")




# ########################################submit#############################
val_file = "test_data_2.csv"
val_df = pd.read_csv(val_file)
column_name = list(val_df.columns)
tmp = column_name.copy() 
for i in range(len(column_name)):
    tmp[i] += "(t-20)"
val_df.columns = tmp




#create test_data
lags = range(1, 10)  # Just two lags for demonstration.
        
val_df = val_df.assign(**{
'{}(t-{})'.format(col[:-6], t+20): val_df[col].shift(t)
for t in lags
for col in val_df
})


val_df = val_df.ix[list(range(9, 10000, 10))]


####time difference data
## Mid' = mid(t)/mid(t-1)-1
dif_name = ["Volume", "BidPrice1", "AskPrice1", "LastPrice", "BidVolume1", "AskVolume1"]
for i in range(21,30):
    val_df['RelMid(t-{})'.format(i)] = (val_df['MidPrice(t-{})'.format(i)]/val_df['MidPrice(t-{})'.format(i-1)])
    for name in dif_name:
        val_df['{}Dif(t-{})'.format(name, i)] = val_df['{}(t-{})'.format(name, i)] - val_df['{}(t-{})'.format(name, i-1)]



subtract = ["MidPrice", "LastPrice", "BidPrice1", "AskPrice1"]
# # #get meansubtract
mean_dict = {}
for name in subtract:
    mean_dict[name] = []
    for i in range(20, 30):
        mean_dict[name].append('{}(t-{})'.format(name, i))
for key, value in mean_dict.items():
    print(key)
    val_df[key] = val_df[value].mean(axis=1)

subtract = ["MidPrice", "LastPrice", "BidPrice1", "AskPrice1"]
for name in subtract:
    for i in range(20,30):
        val_df['{}(t-{})'.format(name, i)] = val_df['{}(t-{})'.format(name, i)]-val_df[name]#val_df[name]#
val_df.drop(subtract, axis=1, inplace=True)


tmp = ["day_hour", "weekday"]
#one-hot
one_hot = []
for name in tmp:
    for i in range(21,30):
        one_hot.append('{}(t-{})'.format(name, i))
val_df.drop(one_hot, axis=1, inplace=True)

dummies = pd.get_dummies(val_df['day_hour(t-20)'], prefix="day_hour1")
val_df = val_df.loc[:,val_df.columns!="day_hour(t-20)"].join(dummies)
dummies = pd.get_dummies(val_df['weekday(t-20)'], prefix="weekday1")
val_df = val_df.loc[:,val_df.columns!="weekday(t-20)"].join(dummies)


val_df.to_csv("test_data_3.csv")

