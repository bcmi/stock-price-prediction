import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import time
import math

TIME_COL = "Time"

cols=["MidPrice", 
      "LastPrice",
      "AskVolume1",
      "BidVolume1",
      "AskPrice1",
      "BidPrice1",
      "Volume"]


class DataLoader():
    '''
    TODO List:
        1. Split the data in morning and afternoon
        2. Normalize the Data
        3. Get real Train Data
        4. Generate Batch
    '''
    def __init__(self, trainDatafile, testDataFile):
        trainDataFrame = pd.read_csv(trainDatafile)
        testDataFrame = pd.read_csv(testDataFile)

        self.data_train = trainDataFrame.get(cols).values[:]
        self.data_test = testDataFrame.get(cols).values[:]
        self.train_date_time = trainDataFrame.get(TIME_COL)
        self.test_date_time = testDataFrame.get(TIME_COL)

        self.test_index = 0

        """ orginally split train data and test data
        # i_split = int(len(dataframe) * split)
        # self.data_train = dataframe.get(cols).values[:i_split]
        # self.data_test  = dataframe.get(cols).values[i_split:]
        """

        self.len_train  = len(self.data_train)
        self.len_test   = len(self.data_test)
        self.len_train_windows = None

        self.morning_data = []
        self.morning_index = 0
        self.afternoon_data = []
        self.afternoon_index = 0
        self.split_data_by_time()

    '''
    Split the data into morning and afternoon type
        morning_data & afternoon_data: contrains the indexes
    '''
    def split_data_by_time(self):
        tmpIndex = 0
        for i, tmpTime in enumerate(self.train_date_time):
            hour, minute, second = [int(k) for k in tmpTime.split(":")]
            if i == self.len_train-1:
                if hour > 11:
                    self.afternoon_data.append((tmpIndex,i+1))
                else:
                    self.morning_data.append((tmpIndex,i+1))
                break
            next_hour, next_minute, next_second = [int(k) for k in self.train_date_time[i+1].split(":")]
            if hour < 12 and next_hour > 11:
                self.morning_data.append((tmpIndex,i+1))
                tmpIndex = i+1
            if hour > 11 and next_hour < 12:
                self.afternoon_data.append((tmpIndex,i+1))
                tmpIndex = i+1

    
    '''
    Generate next batch datas for training: Morning
    Note: one day is a batch
    '''
    def next_morning_batch(self, seq_len = 10, batch_size = 32):
        start, end = self.morning_data[self.morning_index]
        self.morning_index = (self.morning_index+1)%len(self.morning_data)

        i = start
        while i < (end - seq_len):
            x_batch = []
            y_batch = []
            for b in range(batch_size):
                if i >= (end - seq_len):
                    # stop-condition for a smaller final batch if data doesn't divide evenly
                    yield np.array(x_batch), np.array(y_batch)
                    i = 0
                x, y = self._next_window(i, seq_len, normalise = True)
                x_batch.append(x)
                y_batch.append(y) 
                i += 1
            yield np.array(x_batch), np.array(y_batch)
        
    '''
    Get Morning data
    '''
    def get_morning_train_data(self, seq_len=10, normalise=True):
        '''
        Create x, y train data windows
        Warning: batch method, not generative, make sure you have enough memory to
        load data, otherwise use generate_training_window() method. 
        '''
        start, end = self.morning_data[self.morning_index]
        self.morning_index = (self.morning_index+1)%len(self.morning_data)

        data_x = []
        data_y = []
        for i in range(start, end - seq_len):
            x, y = self._next_window(i, seq_len, normalise)
            data_x.append(x)
            data_y.append(y)
        return np.array(data_x), np.array(data_y)

    '''
    Get Afternoon data
    '''
    def get_afternoon_train_data(self, seq_len=10, normalise=True):
        '''
        Create x, y train data windows
        Warning: batch method, not generative, make sure you have enough memory to
        load data, otherwise use generate_training_window() method. 
        '''
        start, end = self.afternoon_data[self.morning_index]
        self.afternoon_index = (self.afternoon_index+1)%len(self.afternoon_data)

        data_x = []
        data_y = []
        for i in range(start, end - seq_len):
            x, y = self._next_window(i, seq_len, normalise)
            data_x.append(x)
            data_y.append(y)
        return np.array(data_x), np.array(data_y)

    '''
    Generate next batch datas for training: Afternoon
    '''
    def next_afternoon_batch(self, seq_len = 10, batch_size = 200):
        start, end = self.afternoon_data[self.morning_index]
        self.afternoon_index = (self.afternoon_index+1)%len(self.afternoon_data)

        i = start
        while i < (end - seq_len):
            x_batch = []
            y_batch = []
            for b in range(batch_size):
                if i >= (end - seq_len):
                    # stop-condition for a smaller final batch if data doesn't divide evenly
                    yield np.array(x_batch), np.array(y_batch)
                    i = 0
                x, y = self._next_window(i, seq_len, normalise = True)
                x_batch.append(x)
                y_batch.append(y) 
                i += 1
            yield np.array(x_batch), np.array(y_batch)

    '''
    Generates the next data window from the given index location i
    '''
    def _next_window(self, i, seq_len, normalise = True):
        window = self.data_train[i:i+seq_len]
        window = self.normalise_windows(window, single_window=True)[0] if normalise else window
        x = window[:]
        y = self.data_train[i+10:i+20, 0] - self.data_train[i+9, 0]
        y = [sum(y)/20]
        return x, y

    '''
    Normalise window with a base value of zero
    '''
    def normalise_windows(self, window_data, single_window=False):
        normalised_data = []
        window_data = [window_data] if single_window else window_data
        for window in window_data:
            normalised_window = []
            means, stds = self.measure_mean_and_std(window)
            for col_i in range(window.shape[1]):
                normalised_col = [(float(p) / float(window[0, col_i])) - 1 for p in window[:, col_i]]
                normalised_window.append(normalised_col)
            # reshape and transpose array back into original multidimensional format
            normalised_window = np.array(normalised_window).T 
            normalised_data.append(normalised_window)
        return np.array(normalised_data)

    def measure_mean_and_std(self, windows_data):
        means = []
        stds = []
        windows_data = np.array(windows_data)
        for ch in range(windows_data.shape[-1]):
            means.append(np.mean(windows_data[:, ch]))
            stds.append(np.std(windows_data[:, ch]))
        return means, stds

    def get_next_test_data(self):
        data_windows = self.data_test[self.test_index: self.test_index+10]

        data_windows = self.normalise_windows(data_windows, single_window=True)

        hour, minute, second = [int(k) for k in self.test_date_time[self.test_index].split(":")]
        self.test_index = (self.test_index+10)%len(self.test_date_time)
        if hour < 12:
            # morning data: 0
            return data_windows, 0
        else:
            # afternoon data: 1
            return data_windows, 1