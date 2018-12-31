import torch
import numpy as np
import LSTM
import matplotlib.pyplot as plt
import preprocess

input_size = 1
hidden_size = 32

net = LSTM.LSTM(input_size, hidden_size)
net.load_state_dict(torch.load('data_lstm.pkl'))

file_test = open('dataset/test_data.csv','r')
line = file_test.readline()

file_out = open('result_lstm.csv','w')
file_out.write('caseid,midprice\n')

show_x = []
show_y = []
show_pre = []

case = 1
while case < 143:
    line = file_test.readline().split(',')
    if len(line) < 9:
        case += 1
        
while case <= 1000:
    # normalization
    series = []
    series_sum = 0
    x = torch.FloatTensor(10).zero_()

    for i in range(10):
        line = file_test.readline().split(',')
        series.append(float(line[3]))
        series_sum += float(line[3])
        end = float(line[3])
    ave = series_sum/10
    '''
    for i in range(10):
        series_sum += (series[i] - ave)*(series[i] - ave)
    mse = series_sum/10

    for i in range(10):
        series[i] = (series[i] - ave)/mse
        x[i] = series[i]
    x = x[np.newaxis, :, np.newaxis]
    '''
    for i in range(10):
        series[i] = series[i] - ave
        x[i] = series[i]
    x = x[np.newaxis, :, np.newaxis]

    prediction = net(x)[-1].data.numpy()[0]
    #print(prediction)
    prediction = prediction + end

    file_out.write(str(case)+','+str(prediction)+'\n')

    line = file_test.readline()
    case += 1
    show_x.append(case)
    show_y.append(end)
    show_pre.append(prediction)

show_x = np.array(show_x)
show_y = np.array(show_y)
show_pre = np.array(show_pre)
plt.figure()
plt.plot(show_x, show_y)
plt.plot(show_x, show_pre, color='red',linestyle='--')
plt.show()

file_test.close()
file_out.close()