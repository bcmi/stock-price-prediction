#begin LSTM
import numpy
import csv
import keras
from keras.models import Sequential
from keras.layers import LSTM, Dense, Activation, Dropout
from keras.utils import np_utils
from keras import regularizers
from keras.callbacks import EarlyStopping
import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def normalization0(train_data):
    #对数据进行正则化处理(-1,1)
    m = numpy.mean(train_data)
    tmin,tmax = train_data.min(),train_data.max()
    return (train_data - m)/(tmax-tmin)

def normalization1(train_data):
        #对数据进行正则化处理(0,1)
        tmin,tmax = train_data.min(),train_data.max()
        return(train_data - tmin)/(tmax - tmin)

def pre_deal(train_data):
    """"预处理中还包括数据清洗"""
    #对训练数据进行预处理
    #将数据转化为增量
    for i in range(len(train_data)-1,0,-1):
       train_data[i] = train_data[i] - train_data[i-1]
    train_data[0] = train_data[0] - train_data[0]
    normalization0(train_data)
    return train_data

def get_data(path):
    #获取训练数据
    train_data = numpy.loadtxt(path,delimiter= ',' ,skiprows=1,usecols=(3,4,6,7,8,9))
    return train_data

def create_model_data(train_data):
    #训练数据准备
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

def get_model(train_data,lstmFirstLayer,lstmSecondLayer,lo,opt):
    trainX,trainY,testX,testY = create_model_data(train_data)
    print("have dealed")
    #创建神经网络
    model = Sequential()
    model.add(LSTM(lstmFirstLayer, input_shape=(trainX.shape[1], trainX.shape[2]),return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(40,return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(lstmSecondLayer,return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=1, kernel_regularizer=regularizers.l2(0.001),activation='tanh'))
    #rmsprop = keras.optimizers.RMSprop(lr=0.0001)
    #model.compile(loss="mse", optimizer=rmsprop)
    #model.compile(loss="mse", optimizer="adagrad")
    model.compile(loss=lo, optimizer=opt)
    early_stopping = EarlyStopping(monitor='val_loss', patience=5)
    model.fit(trainX,trainY,batch_size=100,epochs=100,validation_data=(testX,testY),verbose=1,shuffle=False,callbacks=[early_stopping])
    #进行测试
    #testScore = model.evaluate(testX, testY,batch_size=365, verbose=1)
    #print("Model Accuracy: %.2f%%" % (testScore*100))
    #保存model
    hyperparams_name=str(lstmFirstLayer)+"-"+str(lstmSecondLayer)
    model.save(os.path.join('MODEL{}_cont.h5'.format(hyperparams_name)))
    return model

def predict(model):
    test_data = get_data("test_data.csv")
    testX,testY = create_pred_data(test_data)
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

def main():

    lstmFirstLayer,lstmSecondLayer = 40,40
    loss,opt = "mae","rmsprop"
    hyperparams_name=str(lstmFirstLayer)+"-"+str(lstmSecondLayer)+"-"+loss + "-" + opt
    road = os.path.join('MODEL{}_cont.h5'.format(hyperparams_name))
    print(road)
    if (os.path.exists(road)):
        model = keras.models.load_model(road)
    else:
        train_data = get_data("train_data.csv")
        model = get_model(train_data,lstmFirstLayer,lstmSecondLayer,loss,opt)
        model.save(os.path.join(road))
    result = predict(model)
    r = os.path.join('RESULT{}_cont.csv'.format(hyperparams_name))
    output_result(result,r)

main()