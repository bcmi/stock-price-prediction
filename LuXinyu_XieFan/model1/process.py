import pandas as pd
import numpy as np
import tensorflow as tf
import csv

     
train_f=open('train_data.csv') 
test_f=open("test_data.csv")
df=pd.read_csv(train_f)  
data=df.iloc[:,3:10].values 
std_data=data

df_test=pd.read_csv(test_f)
temppanda=df_test.iloc[1420:,:]
test_data=temppanda.iloc[:,3:10].values
time=10



def larger_test(data):
    ma=[]
    print("len:",len(data))
    size=len(data)//time

    for i in range(size):
        if(i<size-2):
            temp=data[i*time:i*time+1,:]
            
            for j in range(1,time):
                temp=np.concatenate((temp,data[i*time+j:i*time+j+1,:]),axis=1)
            ma.append(temp)
        else:
            temp=data[i*time:i*time+1,:]
            for j in range(1,time):
                temp=np.concatenate((temp,data[i*time+j:i*time+j+1,:]),axis=1)
            ma.append(temp)
    return ma

def larger_77(data):
    ma=[]
    ma_y=[]
    print("len:",len(data))
    size=len(data)//time

    for i in range(size-2):
        temp=data[i*time:i*time+1,:]
        y=np.mean((std_data[(i+1)*time:(i+3)*time,0]),axis=0)-std_data[(i+1)*time-1,0]
        # y=np.reshape(y,(y.shape[0],1))
        for j in range(1,time):
            temp=np.concatenate((temp,data[i*time+j:i*time+j+1,:]),axis=1)
        ma.append(temp)
        ma_y.append(y)
    temp=data[(i+1)*time:(i+1)*time+1,:]
    y=np.mean((std_data[(i+2)*time:(i+4)*time,0]),axis=0)-std_data[(i+2)*time-1,0]
    
    for j in range(1,time):
        temp=np.concatenate((temp,data[i*time+j:i*time+j+1,:]),axis=1)
    ma.append(temp)
    ma_y.append(y)
   
    return ma,ma_y

def get_train_data():

    data_train=data 
    normalized_train_data=(data_train-np.mean(data_train,axis=0))/np.std(data_train,axis=0)   
    
    ma,ma_y=larger_77(normalized_train_data)
    ma=np.array(ma)
    ma=np.reshape(ma,(ma.shape[0],ma.shape[2]))


    with open('train_y.csv','w',newline='') as csv_file:
      csv_writer = csv.writer(csv_file)
      for row in range(len(ma)):
        li=[]
        li.append(ma_y[row])
        csv_writer.writerow(li)

    with open('train_ma.csv','w',newline='') as csv_file:
      csv_writer = csv.writer(csv_file)
      for row in range(len(normalized_train_data)):
        li=[]
        li.extend(normalized_train_data[row].tolist())
        csv_writer.writerow(li)
 



def get_test_data():


    data_test=test_data
    mean=np.mean(data_test,axis=0)
    std=np.std(data_test,axis=0)
    normalized_test_data=(data_test-mean)/std


    ma = larger_test(normalized_test_data)
    ma=np.array(ma)
    ma=np.reshape(ma,(ma.shape[0],ma.shape[2]))



    with open('test_ma.csv','w',newline='') as csv_file:
      csv_writer = csv.writer(csv_file)
      csv_writer.writerow(["1",'2','3','4','5','6','7'])
      for row in range(len(normalized_test_data)):
        li=[]
        li.extend(normalized_test_data[row].tolist())
        csv_writer.writerow(li)



def process_train_data():
    get_train_data()



def process_test_data():
    get_test_data()



process_train_data()
process_test_data()
