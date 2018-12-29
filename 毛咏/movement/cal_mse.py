import pickle
import numpy as np

def cal_mse(n, price):
    a = price[0]
    if n==0:
        a += 2.9e-4
    elif n==2:
        a -= 2.9e-4
    mse = (a - price[1])**2
    return mse


f = open('data/y_val', 'rb')
f2 = open('data/actual_val', 'rb')



predict = pickle.load(f)
actual = pickle.load(f2)

print(predict.shape, actual.shape)

error = 0
for i in range(len(predict)):
    c = np.argmax(predict[i])
    error += cal_mse(c, actual[i])

print(error / len(predict))