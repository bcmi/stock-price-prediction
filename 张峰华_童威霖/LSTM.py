#begin LSTM
import numpy
import csv
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras import regularizers
from keras.callbacks import EarlyStopping


#get train data
def get_data(path):
    train_data = numpy.loadtxt(path,delimiter= ',' ,skiprows=1,usecols=(3,4,6,7,8,9))
    return train_data

    
#normalization of data from -1 to 1
def data_normalize(train_data):   
    mean_data = numpy.mean(train_data)
    train_max = train_data.max()    
    train_min = train_data.min()
    result = (train_data - mean_data)/(train_max-train_min)
    return result

    
#data predeal and transform data to increment 
def pre_deal(train_data):
    for i in range(len(train_data)-1,0,-1):
       train_data[i] = train_data[i] - train_data[i-1]
    train_data[0] = train_data[0] - train_data[0]
    data_normalize(train_data)
    return train_data


#data divition
def create_model_data(train_data):
    train = train_data[:,1:]
    label = train_data[:,0:1]
    train = pre_deal(train)
    train_set = train[:int(len(train)/10)*10-20,:]
    label_set = []
    for i in range(int(len(train)/10)-2):
        temp = 0
        for j in range(20):
            temp = temp + label[10*(i+1)+j] - label[10*(i+1)-1]
        label_set.append(temp/20)
    train_set = train_set.reshape(int(len(train_set)/10),50,1)
    label_set = numpy.array(label_set)
    return train_set, label_set

    
#prepare of predicton data
def create_pred_data(predict_data):
    predict = predict_data[:,1:]
    label = predict_data[:,0:1]
    predict = pre_deal(predict)
    predict_set = predict[:,:]
    predict_set = predict_set.reshape(int(len(predict_set)/10),50,1)
    label_set = []
    for i in range(int(len(predict_set))):
        label_set.append(label[10*(i+1)-1])
    label_set = numpy.array(label_set)
    return predict_set,label_set

    
def get_model_1(train_data):
    #get dataset
    train_set,label_set = create_model_data(train_data)
    #establish the LSTM network
    model = Sequential()
    model.add(LSTM(40, input_shape=(train_set.shape[1], train_set.shape[2]),return_sequences=True))
    model.add(Dropout(0.2))

    model.add(LSTM(40,return_sequences=True))
    model.add(Dropout(0.2))    
    
    model.add(LSTM(40,return_sequences=False))
    model.add(Dropout(0.2))
    
    model.add(Dense(units=1, kernel_regularizer=regularizers.l2(0.001),activation='tanh'))

    model.compile(loss="mae", optimizer="rmsprop")
    early_stopping = EarlyStopping(monitor='val_loss', patience=5)
    
    model.fit(train_set,label_set,batch_size=100,epochs=100,validation_split = 0.25,verbose=1,shuffle=False,callbacks=[early_stopping])

    return model

 
def get_model_2(train_data):
    #get dataset
    train_set,label_set = create_model_data(train_data)
    #establish the LSTM network
    model = Sequential()
    model.add(LSTM(40, input_shape=(train_set.shape[1], train_set.shape[2]),return_sequences=True))
    model.add(Dropout(0.1))

    model.add(LSTM(40,return_sequences=True))
    model.add(Dropout(0.1))    
    
    model.add(LSTM(40,return_sequences=False))
    model.add(Dropout(0.1))
    
    model.add(Dense(units=1, kernel_regularizer=regularizers.l2(0.001),activation='tanh'))

    model.compile(loss="mae", optimizer="rmsprop")
    early_stopping = EarlyStopping(monitor='val_loss', patience=10)
    
    model.fit(train_set,label_set,batch_size=100,epochs=100,validation_split = 0.25,verbose=1,shuffle=False,callbacks=[early_stopping])

    return model
    
    
def get_model_3(train_data):
    #get dataset
    train_set,label_set = create_model_data(train_data)
    #establish the LSTM network
    model = Sequential()
    model.add(LSTM(30, input_shape=(train_set.shape[1], train_set.shape[2]),return_sequences=True))
    model.add(Dropout(0.2))   
    
    model.add(LSTM(30,return_sequences=False))
    model.add(Dropout(0.2))
    
    model.add(Dense(units=1, kernel_regularizer=regularizers.l2(0.001),activation='tanh'))

    model.compile(loss="mae", optimizer="rmsprop")
    early_stopping = EarlyStopping(monitor='val_loss', patience=10)
    
    model.fit(train_set,label_set,batch_size=100,epochs=100,validation_split = 0.2,verbose=1,shuffle=False,callbacks=[early_stopping])

    return model
    
    
def get_model_4(train_data):
    #get dataset
    train_set,label_set = create_model_data(train_data)
    #establish the LSTM network
    model = Sequential()
    model.add(LSTM(40, input_shape=(train_set.shape[1], train_set.shape[2]),return_sequences=True))
    model.add(Dropout(0.15))
    
    model.add(LSTM(40,return_sequences=False))
    model.add(Dropout(0.15))
    
    model.add(Dense(units=1, kernel_regularizer=regularizers.l2(0.001),activation='tanh'))

    model.compile(loss="mse", optimizer="adagrad")
    early_stopping = EarlyStopping(monitor='val_loss', patience=1)
    
    model.fit(train_set,label_set,batch_size=100,epochs=100,validation_split = 0.2,verbose=1,shuffle=False,callbacks=[early_stopping])

    return model
    
    
#mid price prdition    
def predict(model):  
    predict_data = get_data("test_data.csv")
    predict_set ,label_set = create_pred_data(predict_data)
    result = model.predict(predict_set)
    result = result + label_set
    return result

    
#output of result
def output_result(result):
    stu = ['caseid','midprice']
    out = open('result.csv','w', newline='')
    csv_write = csv.writer(out,dialect='excel')
    csv_write.writerow(stu)
    for i in range(142,1000):
        temp = [i+1,result[i][0]]
        csv_write.writerow(temp)


def main(): 
    train_data = get_data("train_data.csv")
    #model = get_model_1(train_data)
    #model = get_model_2(train_data)
    #model = get_model_3(train_data)
    model = get_model_4(train_data)
    result = predict(model)
    output_result(result)

if __name__ == "__main__":  
    main()