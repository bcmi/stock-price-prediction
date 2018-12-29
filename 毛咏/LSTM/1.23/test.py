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


res = []
for i in range(1000 - 1):
    a = test_data_y[(i+1) * time_steps - 1][0] * 0.7
    b = test_data_y[(i+1) * time_steps][0] * 0.3
    y = a + b
    res.append(y)
res.append(test_data_y[1000 * time_steps - 1][0])

with open('res_t.csv', 'w', encoding='utf8', newline='') as fout:
    fieldnames = ['caseid','midprice']
    writer = csv.DictWriter(fout, fieldnames = fieldnames)
    writer.writeheader()
    for i in range(142, len(res)):
        writer.writerow({'caseid':str(i+1),'midprice':float(res[i])})
#         writer.writerow({'caseid':str(i+1),'midprice':float(res[i][0] / 100.0 + base_mid_price[i])})
print("Done.")