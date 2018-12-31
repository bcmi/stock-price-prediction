import pickle
import numpy as np

def get_dataset(cut_data):
    X = []
    Y = []
    actual = []
    for piece in cut_data:
        p = np.array(piece)
        p = np.hstack((p[:,0:2], p[:, 3:]))
        for i in range(0, len(p) - 29):
            x = p[i:i+10, :].reshape(-1)  
            X.append(x)
            actual.append(np.mean(p[i+10:i+30, 0]))
            if np.mean(p[i+9:i+10, 0]) > np.mean(p[i+10:i+30, 0]):
                y = 0
            else:
                y = 1
            Y.append(y)
    return np.array(X), np.array(Y), np.array(actual) 

def cal_mse(n, last_price, actual_price, fluc):
    a = last_price
    rate = np.random.rand()
    if rate > 0.3:
        if n==0:
            a -= fluc *1e-5
        else:
            a += fluc *1e-5
    else:
        if n==0:
            a += fluc *1e-5
        else:
            a -= fluc *1e-5
    mse = (a - actual_price)**2
    return mse

f = open('data/cut_data', 'rb')
pieces = pickle.load(f)
X, Y, actual = get_dataset(pieces)
randlist = np.arange(len(X))
np.random.shuffle(randlist)
# print(randlist)
X = X[randlist]
Y = Y[randlist]
actual = actual[randlist]

# 35 - 40 
num = 100000
for j in range(20, 60):
    se = 0
    for i in range(num):
        se += cal_mse(Y[i], X[i][63], actual[i], j)
    print(j, se / num)
# f = open('data/y_val', 'rb')
# f2 = open('data/actual_val', 'rb')



# predict = pickle.load(f)
# actual = pickle.load(f2)

# print(predict.shape, actual.shape)

# error = 0
# for i in range(len(predict)):
#     c = np.argmax(predict[i])
#     error += cal_mse(c, actual[i])

# print(error / len(predict))

