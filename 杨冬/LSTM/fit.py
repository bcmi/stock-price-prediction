
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import keras
from keras import regularizers
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, LSTM, TimeDistributed, RepeatVector
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
import random
import time
import matplotlib.pyplot as plt

random.seed(time.time())

def get_int_time(str_t):
    h_m_s = str_t.split(":")
    return int(h_m_s[0])*360 + int(h_m_s[1]) * 60 + int(h_m_s[2])

class StockDataSet(object):
    def __init__(self,
                 input_size=6,
                 time_steps=10,
                 test_ratio=0.2,
                 ):
        self.input_size = input_size
        self.time_steps = time_steps
        self.test_ratio = test_ratio
        
        df = pd.read_csv("train_data.csv")
        data = df[["LastPrice", "BidPrice1", "Volume",
                   "BidVolume1", "AskPrice1", "AskVolume1"]].values
        self.midprice = df[["MidPrice"]].values
        self.midprice_diff = self.get_price_diff(self.midprice)
        self.date_time = df[["Date", "Time"]].values
        # data_list = self.get_available_data(data)
        # print(len(data_list))
        self.data = self.manip_vol(data)
        # by experience
        # self.normalized_data = (self.data - self.mins) / (self.maxs - self.mins)
        self.valid_break_point = int(len(self.data) * test_ratio)
        # print(data.shape)
        
    def get_price_diff(self, data):
        new_data = np.copy(data)
        new_data -= np.append(np.array([[0]]), new_data[:-1], axis=0)
        new_data[0, 0] = new_data[1, 0]
        return new_data
        
    def manip_vol(self, data):
        # Volume diff
        data[:, 2] -= np.append(np.array([[0 for i in range(6)]]), data[:-1], axis=0)[:, 2]
        data[0, 2] = data[1, 2]
        
        # deal with < 0
        for i in range(len(data)):
            if (data[i][2] <= 0):
                data[i][2] = 0
                
        # do log
        # data[:, [3, 4, 6]] = np.log10(data[:, [3, 4, 6]])

        # data = np.nan_to_num(data)
        
        # vol_maxs = data[:, [3, 4, 6]].max(axis=0)
        vol_maxs = np.array([8148400., 30925500., 9056100.])
        vol_means = np.array([36736.14723548, 249269.25007034, 236977.29252231])
        vol_stds = np.array([118625.54030232, 748339.48698892, 331295.10866236])
        # print(vol_maxs)
        vol_mins = np.array([0, 0, 0])
        data[:, [2, 3, 5]] = (data[:, [2, 3, 5]] - vol_means) / vol_stds
        return data
    
        
        
    def shuffle(self, X, Y):
        np.random.seed(10)
        randomList = np.arange(X.shape[0])
        np.random.shuffle(randomList)
        return X[randomList], Y[randomList]
        
    def get_train_data(self, time_steps=10):
        x, y = [], []
        i = self.valid_break_point
        while (i < len(self.data) - 3 * time_steps - 1):
            if ((self.date_time[i + 29][0] != self.date_time[i][0]) or
                get_int_time(self.date_time[i+29][1]) != (get_int_time(self.date_time[i][1]) + 29 * 3)):
                i = i + 30 # next day
                continue
            tmp_x = np.copy(self.data[i: i + time_steps])
            # tmp_x = np.append(tmp_x, self.midprice_diff[i: i + time_steps], axis = 1)
            tmp_y = np.mean(self.midprice[i + time_steps: i + 3 * time_steps]) - self.midprice[i + 9, 0]
            '''if (tmp_y > 0):
                tmp_y = [1, 0, 0]
            elif (tmp_y == 0):
                tmp_y = [0, 1, 0]
            else:
                tmp_y = [0, 0, 1]'''
                
            # scale
            tmp_x[:, [0, 1, 4]] /= 4
            tmp_y *= 100
            
            x.append(tmp_x.tolist())
            y.append(tmp_y)
            i += 1
        return np.array(x), np.array(y)
        
    def get_valid_data(self, time_steps=10):
        x, y = [], []
        valid_data = self.data[:self.valid_break_point]
        i = 0
        while (i < self.valid_break_point - 3 * time_steps - 1):
            if (self.date_time[i + 29][0] != self.date_time[i+30][0] or
                get_int_time(self.date_time[i+29][1]) != (get_int_time(self.date_time[i][1]) + 29 * 3)):
                i = i + 30 # next day
                continue
            tmp_x = np.copy(self.data[i: i + time_steps])
            # tmp_x = np.append(tmp_x, self.midprice_diff[i: i + time_steps], axis = 1)
            tmp_y = np.mean(self.midprice[i + time_steps: i + 3 * time_steps]) - self.midprice[i + 9, 0]
            '''if (tmp_y > 0):
                tmp_y = [1, 0, 0]
            elif (tmp_y == 0):
                tmp_y = [0, 1, 0]
            else:
                tmp_y = [0, 0, 1]'''
            # scale
            tmp_x[:, [0, 1, 4]] /= 4
            tmp_y *= 100
            
            x.append(tmp_x.tolist())
            y.append(tmp_y)
            i += 1
        return np.array(x), np.array(y)


# In[2]:


def buildManyToOneModel(shape):
    model = Sequential()
    # model.add(LSTM(128, input_shape=(shape[1], shape[2]), recurrent_initializer='orthogonal'))
    model.add(LSTM(64, input_shape=(shape[1], shape[2]), return_sequences=True,
                   recurrent_initializer='orthogonal', unit_forget_bias=True))
    model.add(Dropout(0.4))
    model.add(LSTM(128, recurrent_initializer="orthogonal", 
                   unit_forget_bias=True, return_sequences=False))
    # output shape: (1, 1)
    model.add(Dense(1))
    # model.add(Dense(3, activation='softmax'))
    # opt = keras.optimizers.RMSprop(lr=0.0005, rho=0.9, epsilon=None, decay=0.99)
    opt = keras.optimizers.Adam(lr=0.003, beta_1=0.9, beta_2=0.999, decay=0.9998)
    #model.compile(loss="categorical_crossentropy", optimizer="SGD")
    model.compile(loss="mse", optimizer="adam")
    model.summary()
    return model


# In[3]:


stock_data_set = StockDataSet()

X_train, Y_train = stock_data_set.get_train_data()
X_valid, Y_valid = stock_data_set.get_valid_data()

print(X_train)
print(Y_train)
# print(X_valid)
# print(Y_valid)
# return


# In[7]:


# len(X_train)


# In[4]:


model = buildManyToOneModel(X_train.shape)


# In[6]:


# from keras.utils import plot_model
# plot_model(model, to_file="model.png", show_shapes=True)


# In[14]:


callback = EarlyStopping(monitor="loss", patience=2 , verbose=1, mode="auto")
model.fit(X_train, Y_train, epochs=5, batch_size=32, validation_data=(X_valid, Y_valid), callbacks=[callback])
model.save('6_feature_64+128_epoch_5.h5')


# In[ ]:




