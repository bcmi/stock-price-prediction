import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import time

import warnings
def warn(*args, **kwargs):
    pass
warnings.warn = warn
warnings.filterwarnings("ignore", category=RuntimeWarning)

# delete same row
'''
i=0
while i < train.shape[0]:
        if i % 1000 == 0:
                print('%f' % (i / 430039.0), '1')
        if i == 0:
                i += 1
                continue
        if int(train[i][1][6:]) == int(train[i-1][1][6:]) and int(train[i][1][3:5]) == int(train[i-1][1][3:5]) and int(train[i][1][0:2]) == int(train[i-1][1][0:2]):
                train = np.delete(train, i, axis=0)
                i-=1
        i+=1
np.savetxt('train_mod.csv', train, delimiter=',', fmt="%s", header=','.join(header))
'''

train = pd.read_csv('train_mod.csv')
train_value = np.array(train)[:, 2:]  # mp, lp, v, bp, bv, ap, av
train_time = pd.read_csv('train_mod.csv')
train_time = np.array(train_time)
# train_diff = np.concatenate((np.zeros((1, train.shape[1])), np.diff(train, axis=0)), axis=0)
idx = 0
X = []
X_nonscale = []
Y = []
Y_pre = []
while idx < train_value.shape[0] - 30:
        t1 = time.time()
        # 不要跨越上下午, ugly though
        if abs(int(train_time[idx][1][0:2])*3600+int(train_time[idx][1][3:5])*60+int(train_time[idx][1][6:])-(int(train_time[idx+29][1][0:2])*3600+int(train_time[idx+29][1][3:5])*60+int(train_time[idx+29][1][6:]))) > 100:
                idx += 1
                continue
        X_batch = train_value[idx:idx+10, :]  # shape: (10, 7)
        EMA = [[X_batch[0, 0]]]
        SMA = [[X_batch[0, 0]]]  # also bollinger mid
        BBandUpper = [[X_batch[0, 0]]]
        BBandLower = [[X_batch[0, 0]]]
        K = 2  # Bollinger Bands parameter
        # RSI
        U = []
        D = []
        t2 = time.time()
        for i in range(1, 10):
                if X_batch[i, 0] > X_batch[i-1, 0]:
                        U.append(X_batch[i, 0] - X_batch[i-1, 0])
                        D.append(0)
                else:
                        U.append(0)
                        D.append(X_batch[i-1, 0] - X_batch[i, 0])
                alpha = 2/(i+1.0)
                EMA.append([alpha*X_batch[i, 0]+(1-alpha)*EMA[i-1][0]])
                SMA.append([(SMA[i-1][0]*i+X_batch[i, 0])/(i+1.0)])
                std = np.std(X_batch[:i+1, 0])
                BBandUpper.append([SMA[i][0]+K*std])
                BBandLower.append([SMA[i][0]-K*std])
        t3 = time.time()
        EMA = np.array(EMA)
        RSI = np.mean(U)/(np.mean(D)+np.mean(U))
        X_batch = np.concatenate((X_batch, EMA), axis=1)
        X_batch = np.concatenate((X_batch, SMA), axis=1)
        X_batch = np.concatenate((X_batch, BBandUpper), axis=1)
        X_batch = np.concatenate((X_batch, BBandLower), axis=1)
        X_batch = np.concatenate((X_batch, X_batch[:, [0]]-SMA), axis=1)  # Bias
        # X_batch = np.diff(X_batch, axis=0)
        X_nonscale.append(X_batch)
        X_batch = MinMaxScaler().fit_transform(X_batch)
        # X_batch = X_batch.flatten()
        # X_batch = np.concatenate((X_batch, [RSI]), axis=0)
        t4 = time.time()
        X.append(X_batch)
        Y.append(np.mean(train_value[idx+10:idx+30, 0]) - np.mean(train_value[idx:idx+10, 0]))
        Y_pre.append(np.diff(train_value[idx:idx+10, 0]))
        '''
        if idx == 0:
                X = np.array([X_batch])
                Y = np.array([np.mean(train_value[idx+10:idx+30, 0]) - train_value[idx+9, 0]])
        else:
                X = np.concatenate((X, [X_batch]), axis=0)
                Y = np.concatenate((Y, [np.mean(train_value[idx+10:idx+30, 0]) - train_value[idx+9, 0]]), axis=0)
        '''
        t5 = time.time()
        if idx % 1000 == 0:
                print(idx)
        idx += 1
X = np.array(X)
X_nonscale = np.array(X_nonscale)
Y = np.array(Y)
Y_pre = np.array(Y_pre)
s = np.arange(X.shape[0])
np.random.shuffle(s)
X = X[s]
X_nonscale = X_nonscale[s]
Y = Y[s]
Y_pre = Y_pre[s]

test = pd.read_csv('test_data.csv')
test = np.array(test)[:, 3:]
X_test = []
Y_test_pre = []
idx = 0
while idx < test.shape[0]:
        X_batch = test[idx:idx+10, :]  # shape: (10, 7)
        EMA = [[X_batch[0, 0]]]
        SMA = [[X_batch[0, 0]]]  # also bollinger mid
        BBandUpper = [[X_batch[0, 0]]]
        BBandLower = [[X_batch[0, 0]]]
        K = 2  # Bollinger Bands parameter
        # RSI
        U = []
        D = []
        for i in range(1, 10):
                if X_batch[i, 0] > X_batch[i-1, 0]:
                        U.append(X_batch[i, 0] - X_batch[i-1, 0])
                        D.append(0)
                else:
                        U.append(0)
                        D.append(X_batch[i-1, 0] - X_batch[i, 0])
                alpha = 2/(i+1.0)
                EMA.append([alpha*X_batch[i, 0]+(1-alpha)*EMA[i-1][0]])
                SMA.append([(SMA[i-1][0]*i+X_batch[i, 0])/(i+1.0)])
                std = np.std(X_batch[:i+1, 0])
                BBandUpper.append([SMA[i][0]+K*std])
                BBandLower.append([SMA[i][0]-K*std])
        EMA = np.array(EMA)
        RSI = np.mean(U)/(np.mean(D)+np.mean(U))
        X_batch = np.concatenate((X_batch, EMA), axis=1)
        X_batch = np.concatenate((X_batch, SMA), axis=1)
        X_batch = np.concatenate((X_batch, BBandUpper), axis=1)
        X_batch = np.concatenate((X_batch, BBandLower), axis=1)
        X_batch = np.concatenate((X_batch, X_batch[:, [0]]-SMA), axis=1)  # Bias
        X_batch = MinMaxScaler().fit_transform(X_batch)
        X_test.append(X_batch)
        Y_test_pre.append(np.diff(test[idx:idx+10, 0]))
        idx += 10

X_test = np.array(X_test)
Y_test_pre = np.array(Y_test_pre)
np.save('Fin_X_GAN.npy', X)
np.save('Fin_X_GAN_nonscale.npy', X_nonscale)
np.save('Fin_Y_GAN.npy', Y)
np.save('Fin_Y_pre.npy', Y_pre)
np.save('Fin_test_GAN.npy', X_test)
np.save('Fin_test_pre.npy', Y_test_pre)
