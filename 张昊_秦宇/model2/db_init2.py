import csv
import os
import torch
import pandas as pd
import numpy as np
import copy

from sklearn.preprocessing import StandardScaler

FOLDER = 'data'
TRAIN_SET_DATA_FILE = 'train_data.csv'
TEST_SET_DATA_FILE = 'test_data.csv'

"""
databse:
0: midPrice
1: lastPrice
2: bidPrice1
3: bidVolume1
4: askPrice1
5: askVolume1
"""

def validate(data):
    for i in range(1, len(data)):
        if int(data[i, 0] - data[i - 1, 0]) != 3:
            return False
    return True

def initDatabase(batch_size, num_features, sequence_length, predict_len):
    '''
    dataset: total_days * num_single_order_book_a_day * num_features dimensions matrix
    '''
    csv_file = pd.read_csv(os.path.join(FOLDER, TRAIN_SET_DATA_FILE))
    
    OriginOrderBook = []
    OrderBook = []
    for name, group in csv_file.groupby('Date'):
        # newgroup = group.fillna(0)
        OriginOrderBook.append(group.loc[:,[ 'Time', 'MidPrice', 'LastPrice', 'Volume',
                     'BidPrice1', 'BidVolume1', 'AskPrice1', 'AskVolume1']].values)

    # Change the time and volume value !!
    for single_day_data in OriginOrderBook:
        for data in single_day_data:
            time = data[0]
            time = time.split(":")
            seconds = int(time[0]) * 3600 + int(time[1]) * 60 + int(time[2])
            data[0] = seconds
        OrderBook.append(single_day_data)

    inputs_dataset = []
    labels_dataset = []
    num = 0
    for single_day_data in OrderBook:
        num += 1
        print(num, len(OrderBook))
        single_inputs = []
        single_labels = []
        single_day_data = np.array(single_day_data.astype(np.float32))

        mean = single_day_data.mean(axis=0)
        std = single_day_data.std(axis=0)
        single_day_data[:, 1:] = (single_day_data[:, 1:] - mean[1:]) / std[1:]
        # Get the value for every group
        for st in range(len(single_day_data) - sequence_length - predict_len + 1):
            if not validate(single_day_data[st:st+sequence_length+predict_len]):
                continue
            single_inputs.append(single_day_data[st: st + sequence_length])
            single_labels.append(single_day_data[st + sequence_length:\
             st + sequence_length + predict_len, 1].mean())

        for i in range(len(single_inputs)):
            single_inputs[i] = single_inputs[i][:,1:]
        inputs_dataset.extend(single_inputs)
        labels_dataset.extend(single_labels)
    print(len(inputs_dataset))
    inputs_dataset = torch.FloatTensor(inputs_dataset)
    labels_dataset = torch.FloatTensor(labels_dataset)
    #dataset = {'inputs': inputs_dataset, 'labels': labels_dataset}
    #torch.save(dataset, 'data/dataset2_2')

    return inputs_dataset, labels_dataset

def init_test_data():
    csv_file = pd.read_csv(os.path.join(FOLDER, TEST_SET_DATA_FILE))
    TestBook = []
    for name, group in csv_file.groupby('Date'):
        newgroup = group.fillna(0)
        TestBook.append(group.loc[:,[ 'MidPrice', 'LastPrice', 'Volume',
                     'BidPrice1', 'BidVolume1', 'AskPrice1', 'AskVolume1']].values)
    test_dataset = []
    mean_dataset = []
    std_dataset = []
    for test_data in TestBook:
        test_data = np.array(test_data)
        test_data.astype(np.float32)
        mean = test_data.mean(axis = 0)
        std = test_data.std(axis = 0)

        test_data = (test_data - mean) / std

        for i in range(0, len(test_data), 10):
            test_dataset.append(test_data[i:i+10])
            mean_dataset.append(mean[0])
            std_dataset.append(std[0])
    test_dataset = torch.FloatTensor(test_dataset)
    return test_dataset, mean_dataset, std_dataset

if __name__ == "__main__":
    initDatabase(128, 7, 10, 20)
