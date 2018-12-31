import pandas as pd
import numpy as np
# import torch
import csv
import pickle

train_data = []
train_label = []
input_interval = []
reader = csv.reader(open('washed_train_data_feature_plus.csv','r'))
next(reader)
for row in reader:
    if row != []:
        input_interval.append(row[3:])
    else:
        input_interval_np = np.array(input_interval,float)
        L = len(input_interval_np)
        if  L>= 30:
            for i in range(L-29):
                item = input_interval_np[i:i+10].copy()
                item_mean = item.mean(axis=0)
                item_std = item.std(axis=0)
                item = (item-item_mean)/(item_std+1)
                label = (input_interval_np[i+10:i+30,0].mean(axis=0) - item_mean[0]) / (item_std[0] + 1)
                item[0,7] = 0.0 # only for feature plus
                train_data.append(item)
                train_label.append(label)
        input_interval = []

train_data_np = np.array(train_data).reshape(-1,100)
train_label_np = np.array(train_label).reshape(-1)
print(train_data_np,train_data_np.shape)
print(train_label_np,train_label_np.shape)

pickle.dump([train_data_np,train_label_np],open('generated_train_data_feature_plus.pkl','wb'))