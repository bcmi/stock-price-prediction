import torch
import numpy as np
import math
import LSTM
import random
import sys
import datetime

input_size = 6
hidden_size = 16

net = LSTM.LSTM(input_size, hidden_size)
net.load_state_dict(torch.load('data_lstm.pkl'))

file_test = open('../dataset/test_data.csv', 'r')
line = file_test.readline()

file_out = open('result_lstm.csv', 'w')
file_out.write('caseid,midprice\n')

case = 1
while case < 143:
    line = file_test.readline().split(',')
    if len(line) < 9:
        case += 1

while case <= 1000:
    series = []
    times = []
    series_sum = 0
    bid_vol_sum = 0
    bid_ser_sum = 0
    ask_vol_sum = 0
    ask_ser_sum = 0
    series_sum = 0
    ask_vol_max = -sys.maxsize
    ask_vol_min = sys.maxsize
    bid_vol_max = -sys.maxsize
    bid_vol_min = sys.maxsize
    time_max = -sys.maxsize
    time_min = sys.maxsize
    # x = torch.FloatTensor(10).zero_()

    for i in range(10):
        data = []
        line = file_test.readline().split(',')
        data.append(float(line[3]))
        data.append(float(line[6]))
        data.append(float(line[7]))
        data.append(float(line[8]))
        data.append(float(line[9]))
        data.append(float(0.0))
        series.append(data)
        times.append(datetime.datetime.strptime(line[1] + ' ' + line[2], "%Y-%m-%d %H:%M:%S"))
        series_sum += float(line[3])
        bid_ser_sum += float(line[6])
        ask_ser_sum += float(line[8])

    ave = series_sum / 10
    ave_bid_pri = bid_ser_sum / 10
    ave_ask_pri = ask_ser_sum / 10
    end1 = series[8][0]
    end2 = series[9][0]
    # print("end2 before:",end2)

    for i in range(10):
        if i == 0:
            series[i][2] = 0
            series[i][4] = 0
        else:
            series[i][2] -= series[i - 1][2]
            series[i][4] -= series[i - 1][4]
            series[i][5] = float((times[i] - times[i - 1]).seconds)
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
        series[i][2] = (series[i][2] - bid_vol_min) / (bid_vol_max - bid_vol_min)
        series[i][4] = (series[i][4] - ask_vol_min) / (ask_vol_max - ask_vol_min)
        if time_max != time_min:
            series[i][5] = (series[i][5] - time_min) / (time_max - time_min)
        else:
            series[i][5] /= 3

    x = torch.FloatTensor(series)
    x = x[np.newaxis, :]

    # print("pred:",net(x).data.numpy()[-1][0])
    prediction = net(x).data.numpy()[-1][0] + end2 + random.uniform(-0.0001, 0.0001)
    # prediction = net(x).data.numpy()[-1][0] + end2
    # print(prediction)

    file_out.write(str(case) + ',' + str(prediction) + '\n')

    line = file_test.readline()
    case += 1

file_test.close()
file_out.close()
