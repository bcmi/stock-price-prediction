#begin XGB
import numpy
import pandas as pd
import csv
import keras
import sklearn
from datetime import datetime, timedelta
from keras.models import Sequential
from keras.layers import LSTM, Dense, Activation, Dropout
from sklearn.preprocessing import MinMaxScaler
from keras.utils import np_utils
from keras import regularizers
from keras.callbacks import EarlyStopping
import xgboost as xgb
import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def get_data(path):
    #获取训练数据
    data = numpy.loadtxt(path,delimiter= ',' ,skiprows=1,usecols=(3,4,5,6,7,8,9))
    return data

def create_model_data(train_data):
    #训练数据准备
    dataX = numpy.copy(train_data[:,:])
    dataY = numpy.copy(train_data[:,0:1])
    date = numpy.copy(train_data[:,2])
    datasetX = []
    datasetY = []
    scaler = MinMaxScaler(feature_range=(-1,1))
    count = 0
    begin = 10

    for i in range(20):
        count += dataY[i+begin]
    for i in range(len(train_data)-30):
        if(i == 0):
            datasetY.append(count/20 - dataY[begin-1])
        else:
            count = count - dataY[i-1+begin] + dataY[i+begin+19]
            datasetY.append(count/20 - dataY[i-1+begin,0])
        x = dataX[i:i+10,:]
        x = scaler.fit_transform(x)
        x = x.flatten()
        datasetX.append(x)
    
    coverd = [0]
    for i in range(len(train_data)-1):
        if(date[i] > date[i+1]):
            coverd.append(i+1)
    coverd.append(len(train_data))
    print(len(coverd))
    #清洗
    trainX = []
    trainY = []
    for i in range(len(coverd)-1):
        for j in range(coverd[i],coverd[i+1]-30):
            trainX.append(datasetX[j])
            trainY.append(datasetY[j]) 
    
    trainX = numpy.array(trainX)
    trainY = numpy.array(trainY)
    numpy.save("train_X.npy",trainX)
    numpy.save("train_Y.npy",trainY)
    x = pd.DataFrame(trainX)
    y = pd.DataFrame(trainY)
    x.to_csv("trainX.csv",index = False, header = False)
    y.to_csv("trainY.csv",index = False, header = False)
    return trainX,trainY

def create_pred_data(test_data):
    dataX = numpy.copy(test_data[:,:])
    dataY = numpy.copy(test_data[:,0:1])
    datasetY = []
    for i in range(1000):
        datasetY.append(dataY[10*(i+1)-1])
    datasetX = []
    scaler = MinMaxScaler(feature_range=(-1,1))
    for i in range(1000):
        x = dataX[i*10:(i+1)*10,:]
        x = scaler.fit_transform(x)
        x = x.flatten()
        datasetX.append(x)
    datasetX = numpy.array(datasetX)
    datasetY = numpy.array(datasetY)
    numpy.save("test_X.npy",datasetX)
    numpy.save("test_Y.npy",datasetY)
    x = pd.DataFrame(datasetX)
    y = pd.DataFrame(datasetY)
    x.to_csv("predictX.csv",index = False, header = False)
    y.to_csv("tenY.csv",index = False,header = False)
    return datasetX,datasetY

def predict(model,testX,testY):
    result = model.predict(testX)
    result = result + testY
    return result


def output_result(result,road):
    #result为预测得到的结果
    stu = ['caseid','midprice']
    out = open(road,'w', newline='')
    csv_write = csv.writer(out,dialect='excel')
    csv_write.writerow(stu)
    for i in range(142,1000):
        temp = [i+1,result[i][0]]
        csv_write.writerow(temp)

def get_total_data():
    train_data = get_data("train_data.csv")
    create_model_data(train_data)
    test_data = get_data("test_data.csv")
    create_pred_data(test_data)

def final_xgb(times):
    #创建神经网络
    trainX = numpy.load("train_X.npy")
    trainY = numpy.load("train_Y.npy")
    testX = numpy.load("test_X.npy")
    testY = numpy.load("test_Y.npy")

    #model = xgb.XGBRegressor()
    model = xgb.XGBRegressor(silent=False, silent=False, reg_lambda= 1, eta =0.1, n_estimators=150,subsample =0.6 , max_depth= 6)
    model.fit(trainX,trainY)
    
    pred = model.predict(testX)
    result = {'caseid':[i+1 for i in range(142,1000)], 'midprice':[testY[i,0]+pred[i] for i in range(142,1000)]}
    fl = pd.DataFrame(result)
    fl.to_csv("XGB{}_cont.csv".format(times), index = False)
    return model

def main():
    get_total_data()
    final_xgb(11)
    
main()
