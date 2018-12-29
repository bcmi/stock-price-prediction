import pandas as pd
import numpy as np
import csv
import random
import time
import os

random.seed(time.time())
time_steps = 10
train_ratio = 0.99

# MidPrice=(BidPrice1+AskPrice1)/2,与LastPrice强相关
# BidVolume>AskVolume价格走高

# -------------数据读取----------------#
# 读取测试集
df_t = pd.read_csv("test_data.csv")
# test_data_x = df_t[["LastPrice", "Volume", "BidPrice1",
#                "BidVolume1", "AskPrice1", "AskVolume1"]].values
test_data_x = df_t[["LastPrice", "Volume", "BidPrice1",
                "AskPrice1"]].values
test_bid = df_t[["BidVolume1"]].values
test_ask = df_t[["AskVolume1"]].values
test_data_y = df_t[["MidPrice"]].values

test_cases = []
for i in range(1000):
    m = test_data_y[i * time_steps : (i+1) * time_steps, :]
    test_cases.append(m)
test_cases = np.array(test_cases)

# avg delta
delta = []
for i in range(1000):
    tmpsum = 0.0
    for j in range(len(test_cases[i]) - 1):
        tmpsum += abs(test_cases[i][j + 1][0] - test_cases[i][j][0])
    avg_delta = tmpsum / 9.0
    delta.append(avg_delta)
print(delta[0])
print(delta[1])
print(delta[2])
res = []
for i in range(1000 - 1):
    q = 0
    a = test_data_y[(i+1) * time_steps - 1][0] * 0.972
    b = test_data_y[(i+1) * time_steps][0] * 0.028
    if test_data_y[(i+1) * time_steps][0] > test_data_y[(i+1) * time_steps - 1][0]:
        q = 1
    else:
        q = -1
    y = a + b
    # y1 = test_data_y[(i+1) * time_steps - 1][0] + delta[i] * q * 2
    # y = (y + y1) / 2.0
    res.append(y)
res.append(test_data_y[1000 * time_steps - 1][0])

with open('res_t.csv', 'w', encoding='utf8', newline='') as fout:
    fieldnames = ['caseid','midprice']
    writer = csv.DictWriter(fout, fieldnames = fieldnames)
    writer.writeheader()
    for i in range(142, len(res)):
        writer.writerow({'caseid':str(i+1),'midprice':float(res[i])})
        # writer.writerow({'caseid':str(i+1),'midprice':float(res[i][0] / 100.0 + base_mid_price[i])})
print("Done.")