import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

train = pd.read_csv('train_data.csv')
header = list(train)[1:]
train = np.array(train.iloc[:, 1:])

# delete same row
i = 0
while i < train.shape[0]:
        if i % 1000 == 0:
                print('stage1: '+'%.3f' % (i*100/train.shape[0])+'%')
        if i == 0:
                i += 1
                continue
        if int(train[i][1][6:]) == int(train[i-1][1][6:]) and int(train[i][1][3:5]) == int(train[i-1][1][3:5]) and int(train[i][1][0:2]) == int(train[i-1][1][0:2]):
                train = np.delete(train, i, axis=0)
                i-=1
        i+=1
np.savetxt('train_mod.csv', train, delimiter=',', fmt="%s", header=','.join(header))


train = pd.read_csv('train_mod.csv').drop(['# Date', 'Time', 'MidPrice'], axis=1)
print(train.shape)
train = np.array(train)

# add features
train = np.concatenate((train, train[:, [3]] - train[:, [5]]), axis=1)  # BidVolume - AskVolume
train = np.concatenate((train, train[:, [2]] - train[:, [4]]), axis=1)  # BidPrice - AskPrice
train = np.concatenate((train, train[:, [2]]*train[:, [3]] - train[:, [4]]*train[:, [5]]), axis=1)  # BidPrice * BidVolume - AskPrice * AskVolume
train_ = train

train = np.concatenate((np.zeros((1, train.shape[1])), np.diff(train, axis=0)), axis=0)
train[:, 6:] = train_[:, 6:]
scaler = MinMaxScaler()
train = scaler.fit_transform(train)
X_train = []
Y_train = []
train_time = pd.read_csv('train_mod.csv')
train_time = np.array(train_time)
print(train.shape)
i = 0
while i < train.shape[0]-30:
        if i % 100 == 0:
                print('stage2: '+'%.3f' % (i*100/430039.0)+'%')
        X = train[i+1:i+10, :].flatten()

        # make sure data is continuous
        if abs(int(train_time[i+1][1][0:2])*3600+int(train_time[i+1][1][3:5])*60+int(train_time[i+1][1][6:])-
               (int(train_time[i+29][1][0:2])*3600+int(train_time[i+29][1][3:5])*60+int(train_time[i+29][1][6:]))) > 100:
                i += 1
                continue
        Y = np.mean(train_time[i+10:i+30, 2]) - train_time[i+9, 2]
        X_train.append(X)
        Y_train.append(Y)
        i += 1

# shuffle
X_train = np.array(X_train)
Y_train = np.array(Y_train)
s = np.arange(X_train.shape[0])
np.random.shuffle(s)
X_train = X_train[s]
Y_train = Y_train[s]
np.save('X_train.npy', X_train)
np.save('Y_train.npy', Y_train)


test = pd.read_csv('test_data.csv').drop(['Date', 'Time', 'MidPrice'], axis=1)
test = np.array(test.iloc[:, 1:])

# add features
test = np.concatenate((test, test[:, [3]] - test[:, [5]]), axis=1)  # BidVolume - AskVolume
test = np.concatenate((test, test[:, [2]] - test[:, [4]]), axis=1)  # BidPrice - AskPrice
test = np.concatenate((test, test[:, [2]]*test[:, [3]] - test[:, [4]]*test[:, [5]]), axis=1)  # BidPrice * BidVolume - AskPrice * AskVolume
test_ = test.copy()

test = np.concatenate((np.zeros((1, test.shape[1])), np.diff(test, axis=0)), axis=0)  # take diff between row
test[:, 6:] = test_[:, 6:]
test = scaler.transform(test)
X_test = []
for i in range(1000):
        X_i = np.array(test[i*10+1:i*10+10, :]).flatten()
        X_test.append(X_i)
X_test = np.array(X_test)
np.save('X_test.npy', X_test)
