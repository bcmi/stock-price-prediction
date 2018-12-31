import tensorflow as tf
import pickle
import numpy as np
import csv
import random
from sklearn.preprocessing import MinMaxScaler

global maxs
global mins
global mean
global std
# def get_train_data():
#   # global maxs
#   # global mins
#   global mean
#   global std
#   time_step=10
#   file=open("train_data.pkl","rb")
#   train_data=pickle.load(file)
#   file.close()  
#   train_data=np.array(train_data)
#   train_data_x=train_data[:,1:]
#   train_data_y=train_data[:,0]
#   mean=np.mean(train_data_x,axis=0)
#   std=np.std(train_data_x,axis=0)
#   train_data_x=(train_data_x-mean)/std
#   # maxs=np.max(train_data_x,axis=0)
#   # mins=np.min(train_data_x,axis=0)
#   # train_data_x=(train_data_x-mins)/(maxs-mins)  
#   train_x,train_y=[],[]
#   for i in range(len(train_data)-3*time_step):
#     x=train_data_x[i:i+time_step]
#     y=np.mean(train_data_y[i+time_step:i+3*time_step])
#     y=[y-train_data_y[i+time_step-1]]
#     train_x.append(x.tolist())
#     train_y.append(y)
#   train_x=np.array(train_x)
#   train_y=np.array(train_y)
#   train_y=train_y*1000
#   return train_x,train_y
def get_train_data():
  global mean
  global std
  global maxs
  global mins
  time_step=10
  file=open("train_data.pkl","rb")
  train_data=pickle.load(file)
  day_index=pickle.load(file)
  file.close()
  train_data=np.array(train_data)
  train_data_x=train_data[:,1:]
  train_data_y=train_data[:,0]
  mean=np.mean(train_data_x,axis=0)
  std=np.std(train_data_x,axis=0)

  maxs=np.max(train_data_x,axis=0)
  mins=np.min(train_data_x,axis=0)
 # train_data_x=(train_data_x-mean)/std
  train_data_x[1]=(train_data_x[1]-mins[1])/(maxs[1]-mins[1])
  train_data_x[3]=(train_data_x[3]-mins[3])/(maxs[3]-mins[3])
  train_data_x[5]=(train_data_x[5]-mins[5])/(maxs[5]-mins[5])
  train_data_x[0]=(train_data_x[0]-mean[0])/std[0]
  train_data_x[2]=(train_data_x[2]-mean[2])/std[2]
  train_data_x[4]=(train_data_x[4]-mean[4])/std[4]
  

  train_x,train_y=[],[]
  i=0
  while i<len(day_index)-1:
    start=day_index[i]
    end=day_index[i+1]
    j=start
    while j<end-3*time_step:
      x=train_data_x[j:j+time_step]
      y=np.mean(train_data_y[j+time_step:j+3*time_step])
      y=[y-train_data_y[j+time_step-1]]
      train_x.append(x.tolist())
      train_y.append(y)
      j+=1
    i+=3
  train_x=np.array(train_x)
  train_y=np.array(train_y)
  train_y=train_y*100
  return train_x,train_y
  
        

def get_test_data():
  # global maxs
  # global mins
  time_step=10
  file=open("test_data.pkl","rb")
  test_data=pickle.load(file)
  file.close()
  test_data=np.array(test_data)
  test_data_x=test_data[:,1:]
  test_data_y=test_data[:,0]
  # maxs=np.max(test_data_x,axis=0)
  # mins=np.min(test_data_x,axis=0)
  #test_data_x=(test_data_x-mins)/(maxs-mins)
  test_data_x[1]=(test_data_x[1]-mins[1])/(maxs[1]-mins[1])
  test_data_x[3]=(test_data_x[3]-mins[3])/(maxs[3]-mins[3])
  test_data_x[5]=(test_data_x[5]-mins[5])/(maxs[5]-mins[5])
  test_data_x[0]=(test_data_x[0]-mean[0])/std[0]
  test_data_x[2]=(test_data_x[2]-mean[2])/std[2]
  test_data_x[4]=(test_data_x[4]-mean[4])/std[4]
  test=[]
  test_y=[]
  size=len(test_data)//10
  for i in range(size):
    x=test_data_x[i*time_step:i*time_step+time_step]
    y=test_data_y[i*time_step:i*time_step+time_step]
    test.append(x.tolist())
    test_y.append([y[-1]])
  test=np.array(test)
  test_y=np.array(test_y)
  return test,test_y





def train():
  train_x,train_y=get_train_data()
  test_x,y=get_test_data()
  model=tf.keras.Sequential()
  model.add(tf.keras.layers.LSTM(128,input_shape=(train_x.shape[1],train_x.shape[2]),use_bias=True,recurrent_initializer='orthogonal',bias_initializer='zeros',unit_forget_bias=True,return_sequences=True))
  model.add(tf.keras.layers.LSTM(128,return_sequences=False))
  # model.add(tf.keras.layers.LSTM(64,return_sequences=False))
  model.add(tf.keras.layers.Dense(128))
  model.add(tf.keras.layers.Dense(1))
  a=tf.keras.optimizers.Adam(lr=0.001)
  model.compile(optimizer='adam', loss='mae',metrics=['accuracy'])
  model.fit(train_x, train_y, epochs=5, batch_size=128, shuffle=True)
  predict=model.predict(test_x)
  predict=predict/100+y
  return predict






if __name__ == "__main__":
  result=train()
  file=open("result.pkl","wb")
  pickle.dump(result,file)
  file.close() 
  print(result.shape)

  # train_x,train_y=get_train_data()
  # test_x,y=get_test_data()
  # print(train_x.shape)
  # print(train_y.shape)
  # print(test_x.shape)
  # #print(train_y)
  