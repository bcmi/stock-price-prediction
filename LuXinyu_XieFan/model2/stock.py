from keras.models import  Sequential
import keras.layers as kl
from keras.models import Model
from keras import regularizers
import keras as keras
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from bokeh.plotting import output_file, figure, show

import tensorflow as tf
import csv

from keras.layers.recurrent import LSTM



class NeuralNetwork:
    def __init__(self, input1_shape, input2_shape, stock_or_return):
        # self.input_shape = input_shape
        self.stock_or_return = stock_or_return
        self.input1_shape = input1_shape
        self.input2_shape = input2_shape
        self.epochs = 20

    def make_train_model(self):
        pd_train = pd.read_csv("train_standalized.csv")
        pd_train_y = pd.read_csv("train_y_minus1.csv")

        #train data without midprice lastprice
        train1 = np.reshape(np.array(pd_train.iloc[:pd_train_y.shape[0]//10*10*10,2:].values),
                            (len(np.array(pd_train.iloc[:pd_train_y.shape[0]//10*10*10,2:].values))//10, 10, self.input1_shape))

        train1_y = np.reshape(np.array(pd_train.iloc[:pd_train_y.shape[0]//10*10*10,1].values),
                              (len(np.array(pd_train.iloc[:pd_train_y.shape[0]//10*10*10,1].values))//10, 10, 1))

        input_data1 = kl.Input(shape=(10, self.input1_shape))
        lstm1 = kl.LSTM(10, input_shape=(10, self.input1_shape))(input_data1)
        Dense = kl.Dense(10)(lstm1)
        out1 = kl.core.Reshape((10,1))(Dense)

        #second lstm input: midprice

        train2 = np.reshape(np.array(pd_train.iloc[:pd_train_y.shape[0]//10*10*10,0].values),
                            (len(np.array(pd_train.iloc[:pd_train_y.shape[0]//10*10*10,0].values))//10, 10, self.input2_shape))

        train2_y = np.array(pd_train_y.iloc[:pd_train_y.shape[0]//10*10,0].values)

        input_data2 = kl.Input(shape=(10, self.input2_shape))

        input_data = kl.concatenate([out1, input_data2])


        lstm = kl.LSTM(1, input_shape=(10, 2))(input_data)
        out = kl.Dense(1)(lstm)
        print(out.shape)
        print(out1.shape)

        model = Model(inputs=[input_data1, input_data2], outputs=[out1, out])

        model.compile(optimizer="adam", loss='mean_squared_error', metrics=['mse'])

        print("train1_y:",train1_y.shape)
        print("train2_y:",train2_y.shape)

        model.fit([train1, train2], [train1_y, train2_y], epochs=self.epochs)

        model.save("model.h5", overwrite=True, include_optimizer=True)

        pd_test = pd.read_csv("test_standalized.csv")

        test1 = np.reshape(np.array(pd_test.iloc[:,2:].values),
                            (len(np.array(pd_test.iloc[:,:].values))//10, 10, self.input1_shape))

        test2 = np.reshape(np.array(pd_test.iloc[:,0].values),
                            (len(np.array(pd_test.iloc[:,:].values))//10, 10, self.input2_shape))

        prediction_data = []
        stock_data = []

        prediction = model.predict([test1, test2])
        prediction_data = prediction[1]
        caseid=143
        with open('write.csv','w',newline='') as csv_file:
          csv_writer = csv.writer(csv_file)
          csv_writer.writerow(["caseid","midprice"])

          for i in range(len(prediction_data)):
            li = [caseid]
            caseid += 1
            price = prediction_data[i]+3
            print(price)
            li.extend(price)
            csv_writer.writerow(li)

        print(len(prediction_data))

if __name__ == "__main__":
    model = NeuralNetwork(5, 1, True)
    model.make_train_model()
