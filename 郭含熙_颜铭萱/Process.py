import torch
import tensorflow
from tensorflow import Variable
from torch import nn
import torch
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torch.utils.data as Data
import pandas as pd
import csv
import numpy as np
from torch import autograd

file = pd.read_csv('train_data.csv')
index = []
data = file.loc[:, ['Time', 'Date', 'LastPrice', 'Volume', 'BidPrice1', 'BidVolume1', 'AskPrice1', 'AskVolume1', 'MidPrice']].values
for i in range(len(data)):
    tmp = data[i][0].split(':')
    if tmp[0] == '12':
        index.append(i)
for item in index:
    #print('remove: ', data[item][0])
    data = np.delete(data, item, 0)

tmp_date = data[0][1]
tmp_index = 0
with open('train_data_processed.csv', 'w', newline='') as fout:
    fieldnames = ['caseid', 'Date', 'Time', 'MidPrice', 'LastPrice', 'Volume', 'BidPrice1', 'BidVolume1', 'AskPrice1', 'AskVolume1']
    writer = csv.DictWriter(fout, fieldnames=fieldnames)
    writer.writeheader()
    for i in range(0, len(data)):
        if tmp_date != data[i][1]:
            tmp_index += 1
            tmp_date = data[i][1]
        writer.writerow({'caseid': str(tmp_index),
                         'Date': data[i][1],
                         'Time': data[i][0],
                         'MidPrice': data[i][8],
                         'LastPrice': data[i][2],
                         'Volume': data[i][3],
                         'BidPrice1': data[i][4],
                         'BidVolume1': data[i][5],
                         'AskPrice1': data[i][6],
                         'AskVolume1': data[i][7]
                         })

