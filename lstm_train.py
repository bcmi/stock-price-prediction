import torch
import numpy as np
import LSTM
import preprocess

input_size = 1
hidden_size = 32

train_num = 6000

net = LSTM.LSTM(input_size, hidden_size)
optimizer = torch.optim.SGD(net.parameters(), lr=0.0095, momentum=0.9)
loss_func = torch.nn.MSELoss()

for epoch in range(3):
    file_train = open('dataset/train_data.csv', 'r')
    line = file_train.readline()
    for i in range(201204):
        line = file_train.readline()

    for t in range(train_num):
        # normalization
        series = []
        series_sum = 0
        x = torch.FloatTensor(10).zero_()

        for i in range(10):
            line = file_train.readline().split(',')
            series.append(float(line[3]))
            series_sum += float(line[3])
        ave = series_sum / 10

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

        real_sum = 0
        for j in range(20):
            line = file_train.readline().split(',')
            real_sum += float(line[3])
        real_sum = real_sum / 20

        y = torch.FloatTensor(1).zero_()
        y[0] = real_sum - ave

        prediction = net(x)[-1]
        loss = loss_func(prediction, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # print('y: ',y.data.numpy()[0],' predict: ',prediction.data.numpy()[0], ' loss: ',loss.data.numpy())
    file_train.close()

torch.save(net.state_dict(),'data_lstm.pkl')
print('train complete: ',train_num)