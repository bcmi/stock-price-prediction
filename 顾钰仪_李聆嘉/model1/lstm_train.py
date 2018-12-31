import torch
import numpy as np
import math
import LSTM
import datetime
import sys
from torch.autograd import Variable

input_size = 6
hidden_size = 16

train_num = 13000

net = LSTM.LSTM(input_size, hidden_size)
optimizer = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.8)
# optimizer = torch.optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.99))
loss_func = torch.nn.MSELoss()
# loss_func = torch.nn.L1Loss(reduce=False, size_average=False)

file_train = open('../dataset/train_data.csv','r')
line = file_train.readline()

for t in range(train_num):
    """
        line[3] -- midprice, line[6] -- bidprice, line[7] -- bidvolume
        line[8] -- askprice, line[9] -- askvolume
        series[0] -- midprice, series[1] -- bidprice, series[2] -- bidvolume
        series[3] -- askprice, series[4] -- askvolume
    """
    series = []
    y_series = []
    labels = []
    times = []
    series_sum = 0
    bid_vol_sum = 0
    bid_ser_sum = 0
    ask_vol_sum = 0
    ask_ser_sum = 0
    ask_vol_max = -sys.maxsize
    ask_vol_min = sys.maxsize
    bid_vol_max = -sys.maxsize
    bid_vol_min = sys.maxsize
    time_max = -sys.maxsize
    time_min = sys.maxsize

    # x = torch.FloatTensor(10).zero_()
    # y = torch.FloatTensor(10).zero_()

    for i in range(10):
        data = []
        line = file_train.readline().split(',')
        labels.append(float(line[3]))
        data.append(float(line[3]))
        data.append(float(line[6]))
        data.append(float(line[7]))
        data.append(float(line[8]))
        data.append(float(line[9]))
        data.append(float(0.0))
        series.append(data)
        times.append(datetime.datetime.strptime(line[1]+' '+line[2],"%Y-%m-%d %H:%M:%S"))
        series_sum += float(line[3])
        bid_vol_sum += float(line[7])
        ask_vol_sum += float(line[9])
        bid_ser_sum += float(line[6])
        ask_ser_sum += float(line[8])
    ave = series_sum/10
    ave_bid_pri = bid_ser_sum/10
    ave_ask_pri = ask_ser_sum/10

    for i in range(10):
        if i == 0:
            series[i][2] = 0
            series[i][4] = 0
        else:
            series[i][2] -= series[i-1][2]
            series[i][4] -= series[i-1][4]
            series[i][5] = float((times[i] - times[i-1]).seconds)
        series[i][0] -= ave
        series[i][1] -= ave_bid_pri
        series[i][3] -= ave_ask_pri
        if series[i][2] <= bid_vol_min:
            bid_vol_min = series[i][2]
        if series[i][2] >= bid_vol_max:
            bid_vol_max = series[i][2]
        if series[i][4] <= ask_vol_min:
            ask_vol_min = series[i][4]
        if series[i][4] >= ask_vol_max:
            ask_vol_max = series[i][4]
        if series[i][5] <= time_min:
            time_min = series[i][5]
        if series[i][5] >= time_max:
            time_max = series[i][5]

    for i in range(10):
        series[i][2] = (series[i][2] - bid_vol_min)/(bid_vol_max - bid_vol_min)
        series[i][4] = (series[i][4] - ask_vol_min)/(ask_vol_max - ask_vol_min)
        if time_max != time_min:
            series[i][5] = (series[i][5] - time_min)/(time_max - time_min)
        else:
            series[i][5] /= 3

    # print(series)
    x = torch.FloatTensor(series)
    print("first:",x.size())
    for  i in range(9):
        data = []
        data.append(x[i+1][0])
        data.append(x[i+1][1])
        data.append(x[i+1][2])
        data.append(x[i+1][3])
        data.append(x[i+1][4])
        data.append(x[i+1][5])
        y_series.append(data)

    data =[]
    line = file_train.readline().split(',')
    data.append(float(line[3]) - ave)
    data.append(float(line[6]) - ave_bid_pri)
    data.append((float(line[7]) - y_series[8][2] - bid_vol_min)/(bid_vol_max - bid_vol_min))
    data.append(float(line[8]) - ave_ask_pri)
    data.append((float(line[9]) - y_series[8][4] - ask_vol_min)/(ask_vol_max - ask_vol_min))
    t = datetime.datetime.strptime(line[1]+' '+line[2],"%Y-%m-%d %H:%M:%S")
    if time_min != time_max:
        data.append(((t-times[9]).seconds - time_min)/(time_max - time_min))
    else:
        data.append(float((t-times[9]).seconds)/3)
    y_series.append(data)
    y = torch.FloatTensor(y_series)

    x = x[np.newaxis,:]
    print("after:",x.size())
    # y = y[:,np.newaxis]
    print(y.size())

    prediction = net(Variable(x))
    print(prediction.size(),y.size())
    loss = loss_func(prediction, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    #print('y: ',y.data.numpy(),' pre: ',prediction.data.numpy())

file_train.close()
torch.save(net.state_dict(),'data_lstm.pkl')
print('train complete: ',train_num)