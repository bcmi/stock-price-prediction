import tensorflow as tf
import pickle
import numpy as np
import csv
import random
from tensorflow import keras
global mean
global std

def get_train_data(predict_rate):
  global mean
  global std
  time_step=10
  file=open("train_data.pkl","rb")
  train_data=pickle.load(file)
  day_index=pickle.load(file)
  file.close()
  train_data=np.array(train_data)
  mean=np.mean(train_data,axis=0)
  std=np.std(train_data,axis=0)
  train_data_y=train_data[:,0]
  train_data=(train_data-mean)/std
  train_x,train_y=[],[]  
  i=0 
  while i<len(day_index)-1:
    start=day_index[i]
    end=day_index[i+1]
    j=start
    while j<end-3*time_step:
      x=train_data[j:j+time_step]
      m_b=np.mean(train_data_y[j:j+time_step])
      m_a=np.mean(train_data_y[j+time_step:j+3*time_step])
      l=0
      if m_b > m_a*(1+predict_rate):
          l=1
      elif m_b < m_a*(1-predict_rate):
          l=-1
      y=[0,0,0]
      if l==-1:
        y[0]=1
      elif l==0:
        y[1]=1
      else:
        y[2]=1
      train_x.append(x)
      train_y.append(y)
      j+=1
    i+=1
  train_x=np.array(train_x)
  train_y=np.array(train_y)
  return train_x,train_y

def get_test_data():
  time_step=10
  file=open("test_data.pkl","rb")
  test_data=pickle.load(file)
  file.close()
  test_data=np.array(test_data)
  test=[]
  test_data_y=test_data[:,0]
  test_y=[]
  test_data=(test_data-mean)/std
  size=len(test_data)//10
  for i in range(size):
    x=test_data[i*time_step:i*time_step+time_step]
    y=test_data_y[i*time_step:i*time_step+time_step]
    test.append(x)
    test_y.append([y[-1]])
  test=np.array(test)

  test_y=np.array(test_y)
  return test,test_y


def data_compress():
  
  input_img=keras.layers.Input(shape=(None,7))
  encode=keras.layers.Dense(128,activation='relu')(input_img)
  encode=keras.layers.Dense(64,activation='relu')(encode)
  encode_out=keras.layers.Dense(2)(encode)
  decode=keras.layers.Dense(64,activation='relu')(encode_out)
  decode=keras.layers.Dense(128,activation='relu')(decode)
  decode=keras.layers.Dense(7,activation='tanh')(decode)
  autoencoder = keras.Model(inputs=input_img, outputs=decode)
  encoder = keras.Model(inputs=input_img, outputs=encode_out)
  autoencoder.compile(optimizer='adam', loss='mse')
  return autoencoder,encoder

def classify(predict_rate):
    train_x,train_y=get_train_data(predict_rate)
    test,y=get_test_data()
    auto,enco=data_compress()
    auto.fit(train_x, train_x, epochs=3, batch_size=64, shuffle=True)
    train_x=enco.predict(train_x)
    test=enco.predict(test)
    model=keras.Sequential()
    model.add(tf.keras.layers.LSTM(128,input_shape=(train_x.shape[1],train_x.shape[2]),use_bias=True,recurrent_initializer='orthogonal',bias_initializer='zeros',unit_forget_bias=True,return_sequences=True))
    model.add(tf.keras.layers.LSTM(128,return_sequences=False))
    # model.add(tf.keras.layers.LSTM(64,return_sequences=False))
    model.add(tf.keras.layers.Dense(128))
    model.add(tf.keras.layers.Dense(3))
    #a=tf.keras.optimizers.Adam(lr=0.001)
    model.compile(optimizer='adam', loss='categorical_crossentropy',metrics=['accuracy'])
    model.fit(train_x, train_y, epochs=2, batch_size=128, shuffle=True)
    predict=model.predict(test)
    return predict,y

if __name__ == "__main__":
      
    predict,y=classify(0.001)
    file=open('predict.pkl','wb')
    pickle.dump(predict,file)
    pickle.dump(y,file)
    file.close()
    

