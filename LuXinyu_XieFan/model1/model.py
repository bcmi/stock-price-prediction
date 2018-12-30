from keras.models import  Sequential
import keras.layers as kl
from keras.layers import Dense
from keras.models import Model
from keras import regularizers
import keras as keras
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from bokeh.plotting import output_file, figure, show
import csv

from keras.layers.recurrent import LSTM



class NeuralNetwork:
    def __init__(self, input_shape, stock_or_return):
        self.input_shape = input_shape
        self.stock_or_return = stock_or_return

    def make_train_model(self):
        pd_train = pd.read_csv("train_ma.csv")
        pd_train_y = pd.read_csv("train_y.csv")

        input_data = kl.Input(shape=(10, self.input_shape))
        lstm = kl.LSTM(10,input_shape=(10, self.input_shape),activity_regularizer=regularizers.l2(0.003),
                       recurrent_regularizer=regularizers.l2(0), dropout=0.2, recurrent_dropout=0.2)(input_data)
        perc = kl.Dense(1,input_shape=(10, self.input_shape), activity_regularizer=regularizers.l2(0.005))(lstm)
     

        model=Model(input_data,perc)



        model.compile(optimizer="adam", loss="mean_squared_error",metrics=['mse'])

        # load data
        


        #print(pd_train)
        train = np.reshape(np.array(pd_train.iloc[:pd_train_y.shape[0]//10*10*10,:].values),
                           (len(np.array(pd_train.iloc[:pd_train_y.shape[0]//10*10*10,:].values))//10, 10, self.input_shape))
        print(pd_train_y.shape[0]//10*10)
        train_y = np.array(pd_train_y.iloc[:pd_train_y.shape[0]//10*10,:].values)
        # train_stock = np.array(pd.read_csv("train_stock.csv"))
        train_y=train_y
        train=train
        # train model
        print(train_y.shape)
        #train_y=np.reshape(train_y,(1,train_y.shape[0],train_y.shape[1]))
        model.fit(train, train_y, epochs=50,batch_size=40,shuffle=True)

        model.save("model.h5", overwrite=True, include_optimizer=True)

        pd_test = pd.read_csv("test_ma.csv")
        test=np.array(pd_test.iloc[:,:].values)

        test_x = np.reshape(test,
                            (len(test)//10, 10, self.input_shape))
   
        prediction_data = []
        stock_data = []
        for i in range(len(test_x)):
            prediction = (model.predict(np.reshape(test_x[i], (1, 10, self.input_shape))))
         
            prediction_data.append(np.reshape(prediction, (1,)))
            prediction_corrected = (prediction_data - np.mean(prediction_data))/np.std(prediction_data)
        caseid=143
        test_f=open("test_data.csv")
        df_test=pd.read_csv(test_f)
        temppanda=df_test.iloc[1420:,:]
        std_data=temppanda.iloc[:,3:10].values
        ind=10
        with open('write.csv','w',newline='') as csv_file:
          csv_writer = csv.writer(csv_file)
          csv_writer.writerow(["caseid","midprice"])
          for price in prediction_data:
            li=[]
            li.append(caseid)
            li.extend(price+std_data[ind-1,0])
            csv_writer.writerow(li)
            caseid+=1
            ind+=10
        print(len(prediction_data))



if __name__ == "__main__":
    model = NeuralNetwork(7, True)
    model.make_train_model()
