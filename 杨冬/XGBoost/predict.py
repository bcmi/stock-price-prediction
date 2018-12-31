
# coding: utf-8

# In[45]:


import pandas as pd
import numpy as np
import csv
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, LSTM, TimeDistributed, RepeatVector
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import load_model
import random
import time

import matplotlib.pyplot as plt

random.seed(time.time())


class TestCaseData(object):
    def __init__(self,
                 input_size=6,
                 time_steps=10,
                ):
        self.time_steps = time_steps;
        self.input_size = input_size;
        
        df = pd.read_csv("test_data.csv")
        self.raw_data = df[["LastPrice", "BidPrice1", "Volume",
                   "BidVolume1", "AskPrice1", "AskVolume1"]].values
        self.midprice = df[["MidPrice"]].values
        self.midprice_diff = self.get_price_diff(self.midprice)
        self.data = self.manip_vol(self.raw_data)
        
        # self.data[:, [0, 1, 4]] -= self.midprice
        
    def manip_vol(self, data):
        for i in range(1000):
            
            # Volume diff
            data[i * 10: (i+1) * 10, 2] -= np.append(np.array([[0 for i in range(6)]]), 
                                                     data[i * 10: (i+1) * 10 -1], axis=0)[:, 2]
            data[i * 10, 2] = data[i * 10 + 1, 2]
        # deal with < 0
        for i in range(len(data)):
            if (data[i][2] <= 0):
                data[i][2] = 0
        # do log
        # data[:, [3, 4, 6]] = np.log10(data[:, [3, 4, 6]])
            
        data = np.nan_to_num(data)
            
        vol_maxs = np.array([8148400., 30925500.,  9056100.])
        vol_mins = np.array([0, 0, 0])
        vol_means = np.array([36736.14723548, 249269.25007034, 236977.29252231])
        vol_stds = np.array([118625.54030232, 748339.48698892, 331295.10866236])
        data[:, [2, 3, 5]] = (data[:, [2, 3, 5]] - vol_means) / vol_stds
        return data
    
    def get_price_diff(self, data):
        new_data = np.copy(data)
        new_data -= np.append(np.array([[0]]), new_data[:-1], axis=0)
        for i in range(1000):
            new_data[i * 10, 0] = 0
        return new_data
        
    def get_test_cases(self, time_steps=10):
        
        test_cases = []
        for i in range(1000): # totally 1000 test cases
            x = np.copy(self.data[i * time_steps: (i+1) * time_steps])
            x[:, [0, 1, 4]] /= 4
            # x = np.append(x, self.midprice_diff[i*time_steps: (i+1) * time_steps], axis=1)
            test_cases.append(x.tolist())
        return np.array(test_cases)
    
    def get_mid_price_data(self, time_steps=10):
        return self.midprice[[i*10 + 9 for i in range(0, 1000)], [0]].reshape(1000,1)

    def get_up_down(self):
        
        up_down = np.zeros((1000,2), dtype=np.int32)
        for i in range(1000):
            for j in range(5, 10):
                if (self.raw_data[i*10 + j, 3] > self.raw_data[i*10 + j, 5]):
                    up_down[i, 0] += 1
                elif (self.raw_data[i*10 + j, 3] < self.raw_data[i*10 + j, 5]):
                    up_down[i, 1] += 1
        return up_down


# In[46]:


test_case_data = TestCaseData()
mid_price_data = test_case_data.get_mid_price_data()
up_down = test_case_data.get_up_down()


# In[47]:


print(up_down)


# In[25]:




X_test_cases = test_case_data.get_test_cases()
print(X_test_cases)


# In[51]:


model = load_model('6_feature_64+128_epoch_10.h5') # should be checked before running
res = model.predict(X_test_cases)


# In[52]:


for i in range(1000):
    if (up_down[i, 0] > up_down[i, 1]):
        res[i, 0] = np.abs(res[i, 0])
    else:
        res[i, 0] = - np.abs(res[i, 0])


# In[54]:


# res /= 100
# res += mid_price_data
with open('cell_128_sample_.csv','w',encoding='utf8',newline='') as fout:
    fieldnames = ['caseid','midprice']
    writer = csv.DictWriter(fout, fieldnames = fieldnames)
    writer.writeheader()
    for i in range(142, len(res)):
        writer.writerow({'caseid':str(i+1),'midprice':float(res[i][0])})


# In[ ]:




