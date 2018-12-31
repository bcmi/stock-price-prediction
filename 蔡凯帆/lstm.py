#begin LSTM
import pandas as pd
import numpy
import csv
import keras
from keras.models import Sequential
from keras.layers import LSTM, Dense, Activation, Dropout, Flatten
from keras.layers import Conv1D, GlobalMaxPool1D
from keras.optimizers import SGD
from keras.utils import np_utils
from keras import regularizers
from keras.callbacks import EarlyStopping
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def normalization1(train_data):
    tmin,tmax = train_data.min(),train_data.max()
    return(train_data - tmin)/(tmax - tmin)

def pre_deal(train_data):
    for i in range(len(train_data)-1,0,-1):
       train_data[i] = train_data[i] - train_data[i-1]
    train_data[0] = train_data[0] - train_data[0]
    normalization1(train_data)
    return train_data

def get_data(path,start=0):
    df = pd.read_csv(path)
    df = df.loc[start:,["MidPrice","LastPrice","BidPrice1","BidVolume1","AskPrice1","AskVolume1"]]
    train_data = numpy.array(df).astype(float)
    return train_data

def create_model_data(train_data):
    dataX = train_data[:,1:]
    dataY = train_data[:,0:1]
    dataX = pre_deal(dataX)
    datasetX = dataX[:int(len(dataX)/10)*10-20,:]
    datasetY = []
    for i in range(int(len(dataX)/10)-2):
        temp = 0
        for j in range(20):
            temp = temp + dataY[10*(i+1)+j] - dataY[10*(i+1)-1]
        datasetY.append(temp/20)
    datasetX = datasetX.reshape(int(len(datasetX)/10),50,1)
    datasetY = numpy.array(datasetY)
    train_size = int(len(datasetX)*0.75)
    trainX,testX = datasetX[:train_size,:,:],datasetX[train_size:,:,:]
    trainY,testY = datasetY[:train_size,:],datasetY[train_size:,:]
    return trainX,trainY,testX,testY

def create_pred_data(test_data):
    dataX = test_data[:,1:]
    dataY = test_data[:,0:1]
    dataX = pre_deal(dataX)
    datasetX = dataX[:,:]
    datasetX = datasetX.reshape(int(len(datasetX)/10),50,1)
    datasetY = []
    for i in range(int(len(datasetX))):
        datasetY.append(dataY[10*(i+1)-1])
    datasetY = numpy.array(datasetY)
    return datasetX,datasetY

def get_model(train_data,lstmFirstLayer,lstmSecondLayer,lstmThirdLayer):
    trainX,trainY,testX,testY = create_model_data(train_data)
    print("have dealed")
    #lstm
    model = Sequential()
    model.add(LSTM(lstmFirstLayer, input_shape=(trainX.shape[1], trainX.shape[2]),return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(lstmSecondLayer,return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(lstmThirdLayer,return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=1, kernel_regularizer=regularizers.l2(0.001),activation='tanh'))
    model.compile(loss="mae", optimizer="adam")
    early_stopping = EarlyStopping(monitor='val_loss', patience=2)
    history = model.fit(trainX,trainY,batch_size=125,epochs=30,validation_data=(testX,testY),verbose=1,shuffle=True,callbacks=[early_stopping])

    hyperparams_name=str(lstmFirstLayer)+"-"+str(lstmSecondLayer)+"-"+str(lstmThirdLayer)
    model.save(os.path.join('MODEL{}_cont.h5'.format(hyperparams_name)))
    return model

def predict(model):
    test_data = get_data("test_data.csv")
    testX,testY = create_pred_data(test_data)
    result = model.predict(testX)
    result = result + testY
    return result


def output_result(result):
    stu = ['caseid','midprice']
    out = open('result7.csv','w', newline='')
    csv_write = csv.writer(out,dialect='excel')
    csv_write.writerow(stu)
    for i in range(142,1000):
        temp = [i+1,result[i][0]]
        csv_write.writerow(temp)


def main():
    lstmFirstLayer,lstmSecondLayer,lstmThirdLayer = 20,30,30
    loss,opt = "mse","adam"
    hyperparams_name=str(lstmFirstLayer)+"-"+str(lstmSecondLayer)+"-"+str(lstmThirdLayer)
    road = os.path.join('MODEL{}_cont.h5'.format(hyperparams_name))
    if (os.path.exists(road)):
        model = keras.models.load_model(road)
    else:
        train_data = get_data("train_data.csv",start=10)
        model = get_model(train_data,lstmFirstLayer,lstmSecondLayer,lstmThirdLayer)
    result = predict(model)
    output_result(result)

main()
