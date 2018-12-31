import csv
import torch.nn as nn
import torch
import torch.optim as opt


class Stock(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer):
        super(Stock, self).__init__()

        self.hidden_dim = hidden_dim

        self.lstm = nn.LSTM(input_dim, hidden_dim, layer,batch_first=True)

        self.mid2out = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        lstm_out,_ = self.lstm(x)
        out = lstm_out[:,-1,:]
        out = self.mid2out(out)
        return out

def train_and_test():
    crudetrain = open("train_data.csv", "r")
    crudetrain = csv.DictReader(crudetrain)
    train = []
    date = '2018-06-01'
    time = [9, 29, 58]
    group = []
    for item in crudetrain:
        flag = False
        if item['Time'].split(':')[0] == '12':
            continue
        if item['Date'] == date:
            t = item['Time'].split(':')
            for i in range(3):
                t[i] = int(t[i])
            if time[2] < 57:
                if t[2] == time[2] + 3:
                    flag = True
            elif t[2] == time[2] - 57:
                if time[1] < 59 or (t[1] == 0 and t[0] == time[0] + 1):
                    flag = True
        if flag:
            group.append([float(item['MidPrice']), float(item['LastPrice']),
                      float(item['BidPrice1']), float(item['AskPrice1']),
                      float(item['BidVolume1']), float(item['AskVolume1'])])
            for i in range(3):
                time[i] = t[i]
        else:
            train.append(group)
            date = item['Date']
            t = item['Time'].split(':')
            for i in range(3):
                time[i] = int(t[i])
            group = [[float(item['MidPrice']), float(item['LastPrice']),
                      float(item['BidPrice1']), float(item['AskPrice1']),
                      float(item['BidVolume1']), float(item['AskVolume1'])]]

    crudetrain = train
    train = []
    label = []
    for i in crudetrain:
        for j in range(0,len(i)-29,5):
            seq = i[j:j+10]
            s = 0
            for k in range(20):
                s += i[j+k+10][0]
            s /= 20
            s -= seq[-1][0]
            train.append(seq)
            label.append([s])

    num = 10
    size = 6

    learning_rate = 0.01
    model = Stock(size, 64, 8)
    loss_function = nn.MSELoss()
    optimizer = opt.SGD(model.parameters(), lr=learning_rate)

    epoch = 2
    batch_size = 1024
    l = open("loss.csv", "w")
    l = csv.writer(l)

    for i in range(epoch):
        print("epoch",i)
        for j in range(0, len(train),batch_size):
            print(j)
            x = torch.Tensor(train[j:j+batch_size]).view(-1,num,size)
            y = torch.Tensor(label[j:j+batch_size])

            optimizer.zero_grad()

            pred_delta = model(x)

            loss = loss_function(pred_delta, y)
            loss.backward()
            optimizer.step()

            l.writerow([i, j,float(loss)])

    test = open("test_data.csv", "r")
    test = csv.DictReader(test)
    result = open("submission.csv", "w")
    result = csv.writer(result)
    result.writerow(["caseid", "midprice"])

    count = 0
    seq = []
    for item in test:
        count += 1
        if count <= 1420:
            continue
        seq.append([float(item['MidPrice']), float(item['LastPrice']),
                    float(item['BidPrice1']), float(item['AskPrice1']),
                    float(item['BidVolume1']), float(item['AskVolume1'])])
        if count % 10 ==0:
            lastprice = seq[-1][0]
            seq = torch.Tensor([seq]).view(-1, num, size)
            deltap = float(model(seq))
            print(deltap)
            result.writerow([count // 10, lastprice+deltap])
            seq = []

train_and_test()
