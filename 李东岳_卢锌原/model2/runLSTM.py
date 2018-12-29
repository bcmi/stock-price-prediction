from NewDataLoader import DataLoader
from LSTM_Model import Model
import os
import json
import time
import math
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

os.environ["CUDA_VISIBLE_DEVICES"]="1,2"
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.6 
session = tf.Session(config=config)

"""
setting configs
"""
trainDataFileName = "train_data.csv"
testDataFileName = "test_data.csv"
modelSaveDir = "saved_models"
cols=["MidPrice", 
      "LastPrice",
      "AskVolume1",
      "BidVolume1",
      "AskPrice1",
      "BidPrice1",
      "Volume"]
sequenceLength = 30
batchSize = 32
epochNum = 2

def plot_results(predicted_data, true_data):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
    plt.plot(predicted_data, label='Prediction')
    plt.legend()
    plt.show()

def main():
    if not os.path.exists(modelSaveDir): 
        os.makedirs(modelSaveDir)

    data = DataLoader(
        trainDataFileName, 
        testDataFileName,
    )

    morning_model = Model("morning_model")
    afternoon_model = Model("afternoon_model")
    
    # morning_model.build_model()
    # afternoon_model.build_model()

    # for _ in range(2):
    #     # out-of memory generative training
    #     print("...Begin Training Morning LSTM...")
    #     for i in range(len(data.morning_data)):
    #         print("...Morning Data: Day "+ str(i)+ " ...")
    #         x, y = data.get_morning_train_data()
    #         morning_model.train(x, y, epochs = 1, batch_size=32, save_dir=modelSaveDir)
    #     morning_model.save_model()

    #     print("...Begin Training Afternoon LSTM...")
    #     for i in range(len(data.afternoon_data)):
    #         print("...Afternoon Data: Day "+ str(i)+ " ...")
    #         x, y = data.get_afternoon_train_data()
    #         afternoon_model.train(x, y, epochs = 1, batch_size=32, save_dir=modelSaveDir)
    #     afternoon_model.save_model()

    morning_model.load_model()
    afternoon_model.load_model()

    predictions = []
    for _ in range(int(len(data.test_date_time)/10)):
        tmp_test, if_noon = data.get_next_test_data()
        tmpPrediction = []
        if if_noon:
            tmpPrediction = morning_model.predict_data(tmp_test)
        else:
            tmpPrediction = afternoon_model.predict_data(tmp_test)
        predictions += np.ndarray.tolist(tmpPrediction[0])
        print(tmpPrediction)
    predictions = np.array(predictions)
    np.savetxt("predictions.txt", predictions)

if __name__ == "__main__":
    main()