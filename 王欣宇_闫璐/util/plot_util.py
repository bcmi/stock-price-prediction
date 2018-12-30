# encoding: utf-8
# file: plot_util.py
# author: shawn233

from __future__ import print_function
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
from data_util import OrderBook

BASE_DIR = os.path.dirname(os.path.abspath (sys.argv[0]))
DATA_DIR = os.path.join (os.path.dirname (BASE_DIR), 'data')

order_book = OrderBook (256, DATA_DIR)

font1 = {
        'family': 'Times New Roman',
        'weight': 'normal',
        'size': 23
}


def plot_by_data_matrix (data_matrix, axis_span=1):
    day_matrix_list = order_book.divide_by_day (data_matrix)
    
    index_base = 0
    for day_matrix in day_matrix_list:

        mid_price = day_matrix[:, order_book.index['MidPrice']].astype(np.float32)
        print ('mid_price shape', mid_price.shape)
        mean = np.mean (mid_price)
        stddev = np.std (mid_price)
        print ('mean', mean)
        print ('stddev', stddev)

        '''
        draw_mid_price = []
        for i in range (mid_price.shape[0]):
            if i % step == 0:
                draw_mid_price.append (mid_price[i])
        '''
        x_axis = np.arange (mid_price.shape[0])
        x_axis = (x_axis + index_base) * axis_span
        plt.plot (x_axis, mid_price, lw=3)

        index_base += mid_price.shape[0]




def plot_mid_prices ():
    train_data_matrix, test_data_matrix = order_book.get_data_matrix ()
    print ("Plotting train data...")
    
    plt.subplot (121)
    plt.xticks (np.arange(0, 400001, 100000), fontsize=20)
    plt.yticks (fontsize=20)
    plt.ylim (3.0, 3.9)
    plt.title ('train data', font1)
    plt.xlabel ('# record', font1)
    plt.ylabel ('mid price', font1)
    plot_by_data_matrix (train_data_matrix)
    print ("Plotting test data...")
    plt.subplot (122)
    plt.xticks (fontsize=20)
    plt.yticks (fontsize=20)
    plt.title ('prediction data', font1)
    plt.ylim (3.0, 3.9)
    plt.xlabel ('# record', font1)
    plt.ylabel ('mid price', font1)
    plot_by_data_matrix (test_data_matrix)
    
    plt.show()




def plot_mid_prices_by_time_stamp (data_matrix):
    time_stamp = data_matrix[:, order_book.index['TimeStamp']].astype(np.int32)
    min_time_stmap = np.min (time_stamp)

    time_stamp = time_stamp - min_time_stmap

    day_matrix_list = order_book.divide_by_day (data_matrix)
    
    index_base = 0
    for day_matrix in day_matrix_list:

        mid_price = day_matrix[:, order_book.index['MidPrice']].astype(np.float32)
        print ('mid_price shape', mid_price.shape)
        mean = np.mean (mid_price)
        stddev = np.std (mid_price)
        print ('mean', mean)
        print ('stddev', stddev)

        '''
        draw_mid_price = []
        for i in range (mid_price.shape[0]):
            if i % step == 0:
                draw_mid_price.append (mid_price[i])
        '''

        x_axis = time_stamp[index_base:(index_base+mid_price.shape[0])]
        plt.plot (x_axis, mid_price)

        index_base += mid_price.shape[0]
    


if __name__ == "__main__":
    plot_mid_prices()