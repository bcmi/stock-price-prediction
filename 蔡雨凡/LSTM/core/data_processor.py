import math
import numpy as np
import pandas as pd

class DataLoader():
    def __init__(self, filename, split, cols):
        dataframe = pd.read_csv(filename)
        i_split = int(len(dataframe) * split)
        self.data_train = dataframe.get(cols).values[:i_split]
        self.data_test  = dataframe.get(cols).values[i_split:]
        self.len_train  = len(self.data_train)
        self.len_test   = len(self.data_test)
        self.len_train_windows = None
        self.mean = []
        self.std = []

    def get_test_data(self, seq_len, normalise):

        data_windows = []
        for i in range(self.len_test - seq_len):
            data_windows.append(self.data_test[i:i+seq_len])
        #print(data_windows)
        data_windows = np.array(data_windows).astype(float)
        data_windows = self.normalise_windows(data_windows, single_window=False) if normalise else data_windows
        x = data_windows[:, :]
        y = data_windows[:, :, [0]]
        return x,y

    def get_train_data(self, seq_len, normalise):
        '''
        Create x, y train data windows
        Warning: batch method, not generative, make sure you have enough memory to
        load data, otherwise use generate_training_window() method.
        '''
        data_x = []
        data_y = []
        for i in range(self.len_train - seq_len):
            x, y = self._next_window(i, seq_len, normalise)
            data_x.append(x)
            data_y.append(y)
        return np.array(data_x), np.array(data_y)

    def generate_train_batch(self, seq_len, batch_size, normalise):
        '''Yield a generator of training data from filename on given list of cols split for train/test'''
        i = 0
        while i < (self.len_train - seq_len):
            x_batch = []
            y_batch = []
            for b in range(batch_size):
                if i >= (self.len_train - seq_len):
                    # stop-condition for a smaller final batch if data doesn't divide evenly
                    yield np.array(x_batch), np.array(y_batch)
                    i = 0
                x, y = self._next_window(i, seq_len, normalise)
                x_batch.append(x)
                y_batch.append(y)
                i += 1
            yield np.array(x_batch), np.array(y_batch)

    def _next_window(self, i, seq_len, normalise):
        '''Generates the next data window from the given index location i'''
        window = self.data_train[i:i+10]             
        test_window = self.data_train[i+10:i+20]

        y = np.mean(test_window[:],axis=0)[0] - np.mean(window[:],axis=0)[0]
        window = self.normalise_windows(window, single_window=True)[0] if normalise else window  
        x = window[:10]
        

        #y = window[20][0]
        #print("y",y)
        return x, y

    def normalise_windows(self, window_data, single_window=False):
        '''Normalise window with a base value of zero'''
        normalised_data = []
        window_data = [window_data] if single_window else window_data
        #print(window_data)
        for window in window_data:
            #print("window:", window)
            #print("mean:", np.mean(window,axis=0))
            #print("std:", np.std(window, axis=0))
            mean = np.mean(window[:],axis=0)
            std = np.std(window[:],axis=0)
            #print(mean)
            self.mean.append(mean)
            self.std.append(std)
            normalised_window = []
            for col_i in range(window.shape[1]):
                std_tmp = float(std[col_i])
                #if col_i <= 1:
                    #normalised_col = [float(p) - float(mean[col_i]) for p in window[:, col_i]]
                    #normalised_window.append(normalised_col)
                    #print(normalised_window)
                    #continue
                if std_tmp < 0.0000001:
                    #print(mean[col_i])
                    std_tmp+=0.0000001
                normalised_col = [(float(p) - float(mean[col_i]))/std_tmp for p in window[:, col_i]]
                normalised_window.append(normalised_col)
                #print(normalised_window)
            normalised_window = np.array(normalised_window).T # reshape and transpose array back into original multidimensional format
            #print(normalised_window)
            normalised_data.append(normalised_window)
        return np.array(normalised_data)