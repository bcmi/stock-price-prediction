import os
import json
import time
import math
import pandas as pd
import matplotlib.pyplot as plt
from keras import backend as K
import time
import warnings
import numpy as np
from numpy import newaxis
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
import matplotlib.pyplot as plt
import keras
import tensorflow as tf
from sklearn import preprocessing
import csv
from keras.regularizers import l2
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D

warnings.filterwarnings("ignore")
X_seq1 = [[],[],[],[],[],[],[]]
train_data_norm=[[],[],[],[],[],[],[]]
y=[]
X=[]
s=10
empty=20
split=0.9
gap_pos=[0, 4773, 9519, 14246, 18864, 23582, 28357, 33068, 37844, 42555, 47289, 52056, 56856, 61636, 66400, 71152, 75916, 80687, 85573, 90352, 95146, 99942, 104738, 109489, 114277, 119071, 123857, 128626, 133495, 138368, 143228, 147997, 152856, 157730, 162593, 167473, 172308, 177096, 181915, 186748, 191555, 196385, 201203, 206085, 210940, 215777, 220623, 225479, 230350, 235196, 240035, 244875, 249715, 254571, 259360, 264240, 273819, 283375, 288143, 297724, 302485, 312072, 316837, 321660, 326502, 331348, 336191, 341026, 345867, 350700, 355539, 360369, 365158, 369994, 374845, 379678, 385492, 390335, 395234, 401288, 406096, 412221, 418919, 425174]
result = [0 for i in range(1000)]
def rmse(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))

def min_max(train_data, low, high):
    global train_data_norm
    column_number=5
    for k in range(column_number):
        train_data_norm[k] += train_data[:, k][low:high].tolist()

    train_data_norm=np.array(train_data_norm)
    #print(train_data_norm)
    train_data_norm=np.diff(train_data_norm,axis=1)
    #print(train_data_norm)
    #print(train_data[:,0])
    for k in range(column_number):
        c_min = 2147483647
        c_max = -2147483647

        min_index=0
        # print(train_data_norm[k])
        length=len(train_data_norm[k])
        for i in range(length):
            if(train_data_norm[k][i]>c_max):
                c_max=train_data_norm[k][i]
            if(train_data_norm[k][i]<c_min):
                c_min=train_data_norm[k][i]
                min_index=i

        for i in range(length):
            train_data_norm[k][i]=(train_data_norm[k][i]-c_min)/(c_max-c_min)
        #print("---------------------")
        #print(c_min,c_max,min_index)
        #print(train_data_norm[k])
    train_data_norm=np.array(train_data_norm).T

    # for i in range(50):
    #     print(train_data_norm[i])
    return c_min,c_max

def get_train_data(train_data,low,high):#直接从csv得训练数据
    global train_data_norm
    column_number=7#lll
    #print(train_data[0])
    for k in range(column_number):
        train_data_norm[k] += train_data[:, k][low:high].tolist()
    train_data_norm=np.array(train_data_norm)
    train_data_norm=train_data_norm.T
    #print(train_data_norm)
    l=len(train_data_norm)
    # print(l)
    for i in range(l-s-empty+1):

        tmp=train_data_norm[i:i+s,:]

        tmp=np.array(tmp)
        #print(tmp)
        # for j in range(column_number):
        #     if ((j == 2) or (j == 4) or (j == 6)):
        #         tmp[j] = np.diff(tmp[j],axis=0)#axis=1 按行，axis=0 按列
        # print(tmp)
        #tmp=np.diff(tmp,axis=0)
        scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
        tmp = scaler.fit_transform(tmp)
        # if(i<3):
        #     print(i)
        #
        #     print(tmp)
        X.append(tmp)

def get_perdi(train_data, low, high):
    global X_seq1
    column_number=7
    #print(X_seq1)
    for k in range(column_number):
        X_seq1[k] += train_data[:, k][low:high].tolist()
    shift = X_seq1[0][-1]
    X_seq1=np.array(X_seq1)
    #print(X_seq1)
    # for i in range(column_number):
    #     if((i==2) or (i==4) or (i==6)):
    #         X_seq1[i]=np.diff(X_seq1[i],axis=1)
    #print(X_seq1)

    #X_seq1=np.diff(X_seq1,axis=1)

    #print(X_seq1)
    #print(train_data[:,0])
    # for k in range(column_number):
    #     c_min = 2147483647
    #     c_max = -2147483647
    #
    #     min_index=0
    #     # print(X_seq1[k])
    #     length=len(X_seq1[k])
    #     for i in range(length):
    #         if(X_seq1[k][i]>c_max):
    #             c_max=X_seq1[k][i]
    #         if(X_seq1[k][i]<c_min):
    #             c_min=X_seq1[k][i]
    #             min_index=i
    #     #print(c_min, c_max, min_index)
    #     if(c_min==c_max):
    #         #print(c_max,"________________")
    #         for i in range(length):###注意???
    #
    #             X_seq1[k][i] = 0.5
    #     else:
    #         for i in range(length):
    #             X_seq1[k][i]=(X_seq1[k][i]-c_min)/(c_max-c_min)
    #     #print("---------------------")
    #
    #     #print(X_seq1[k])
    X_seq1=np.array(X_seq1).T
    scaler=preprocessing.MinMaxScaler(feature_range=(-1,1))
    X_seq1=scaler.fit_transform(X_seq1)
    #print(X_seq1)
    return shift
def get_y(train_data,low,high):
    #print("get,y_------------------------------------")
    mid_price=train_data[:,0][low:high]#注意用作label的是从原数据得到的，而不是从归一化数据再得
    l = len(mid_price)
    for i in range(s,l-empty+1):

        sum=0
        for j in range(empty):

            sum+=mid_price[i+j]
            #print(i+j)

        sum/=empty
        sum-=mid_price[i-1]
        y.append(sum)

        #print(sum)

    #print("###########################")
    #print(mid_price.tolist())
    #print("y",y)

def get_x(train_data_norm,low,high):#十个节点归一个的输入用norm的data得
    # print(train_data_norm)
    X_nor=train_data_norm[low:high-1]#norm作差后少了一个？？
    #print(X_nor)
    l=len(X_nor)
    #print(l)
    for i in range(l-empty-s+1):

        tmp=X_nor[i:i+s]
        # if(i<3):
        #     print(tmp)
        X.append(tmp)

def fenduan(data1,train_data,low,high):

    global train_data_norm
    global y
    train_data = np.array(train_data)

    # low=0
    # high=2000#430039
    #min,max=min_max(train_data,0,high)#train_data
    #min, max = min_max(train_data, 0, 406096)  # train_data,因为这里有个volume下降
    #min, max = min_max(train_data, 0, 395234)  # train_data,因为这里有个volume下降

    #train_data_norm=np.array(train_data_norm)
    #print(min,max)
    #print(train_data_norm)#到这里就得到了训练的sample，作差且归一化


    get_y(train_data,low,high)

    #y=y[1:] lll
    #print(y)
    #get_x(train_data_norm,0,high)
    get_train_data(train_data, low, high)
    # print(X)
    length=len(y)
    split_pos=int(split*length)
    X_train=np.array(X[:split_pos])
    y_train=np.array(y[:split_pos])

    X_test=np.array(X[split_pos:])
    y_test=np.array(y[split_pos:])
    # print("%%%%%%%%%%%")
    #print(X_train)
    # print("!!!!!!!!!!!!!")
    # print(y_train)
    # for i in range(3):
    #     print(X_train[i])
    #     print(y_train[i])
    #     print("-------------------------")

    # model = Sequential()
    # model.add(LSTM(300, input_shape=(s, 7), return_sequences=False))
    # model.add(Dropout(0.3))
    # model.add(Dense(1, activation="linear"))

    model = Sequential()
    # model.add(LSTM(100,input_shape=(s, 7),return_sequences=True))
    # model.add(Dropout(0.2))

    model.add(Conv1D(filters=32, input_shape=(s,7),kernel_size=3, padding='same', activation='tanh'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(LSTM(100,return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(output_dim=1))  # Linear dense layer to aggregate into 1 val
    model.add(Activation('linear'))

    model.compile(loss=rmse, optimizer='adam')

    # model = Sequential()
    # model.add(LSTM(100, input_shape=(s-1, 6), return_sequences=True))
    # model.add(Dropout(0.5))
    # model.add(LSTM(100,return_sequences=False))
    # model.add(Dense(60, kernel_regularizer=l2(0.001)))
    # model.add(Dense(1, activation="linear"))

    #model.compile(loss=rmse, optimizer="adam")
    model.fit(X_train, y_train, nb_epoch=14, batch_size=16,verbose=2)


    print(model.evaluate(X_test, y_test, batch_size=32, verbose=2, sample_weight=None))

    #result = model.predict(X_test, batch_size=32, verbose=0)

















    mid_result=[0 for i in range(1000)]
    neg_count=0
    for i in range(1000):
        global  X_seq1
        X_seq1 = [[],[],[],[],[],[],[]]

        #print(data1)

        shift=get_perdi(data1, i * 11, 11 * i + 10)

        # print(X_seq1)
        h=[]
        h.append(X_seq1.tolist())
        h=np.array(h)
        X_seq1 = np.array(X_seq1)
        #print("!!!",shift)


        # print(h)
        r = model.predict(h, batch_size=32, verbose=0)
        if(r[0][0]<0):
            neg_count+=1
        mid_result[i]+=(r[0][0]+shift)
        #print(r)
        if(i<100):
            print("r",r)

        # if(r[0][0]!=0):
        #     print("jjjjjjj")


        #print(X_seq1)
    if((neg_count<200) or(neg_count>800)):
        print("wrong")
        return 0
    else:
        for i in range(1000):
            result[i]+=mid_result[i]
        return 1




def main():

    train_data = pd.read_csv('train_data.csv', usecols=['MidPrice', "LastPrice", "AskPrice", "BidPrice","Volume","BidVolume1","AskVolume1"])
    data1 = pd.read_csv('test_data.csv',usecols=['MidPrice',"LastPrice","AskPrice","BidPrice","Volume","BidVolume1","AskVolume1"])
    data1 = np.array(data1)
    #fenduan(train_data,0,2000)

    number_duan=len(gap_pos)
    valid_pred=0
    for i in range(60,82):
        print(i)
        tmp=fenduan(data1,train_data,gap_pos[i],gap_pos[i+1])
        valid_pred+=tmp
        global X_seq1,X,y,train_data_norm
        X_seq1 = [[],[],[],[],[],[],[]]
        train_data_norm = [[],[],[],[],[],[],[]]
        y = []
        X = []
        #print(result)

    for i in range(1000):
        result[i]/=(valid_pred)
    b = [i for i in range(1000)]
    with open('sample.csv', 'w') as fout:
        fieldnames = ['caseid', 'midprice']
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()
        for i in range(len(result)):
            if (i <= 141):
                continue
            writer.writerow({'caseid': str(b[i] + 1), 'midprice': float(result[i])})

main()