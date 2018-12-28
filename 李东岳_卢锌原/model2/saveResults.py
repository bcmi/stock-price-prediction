import numpy as np
import pandas as pd

cols=["MidPrice", 
      "LastPrice",
      "AskVolume1",
      "BidVolume1",
      "AskPrice1",
      "BidPrice1",
      "Volume"]

predictions = np.loadtxt("predictions.txt")


testDataFrame = pd.read_csv("test_data.csv")
data_test = testDataFrame.get(cols).values[:]

count = 0
for i in range(int(len(data_test)/10)):
    tmpData = []
    for j in range(10):
        tmpData.append(data_test[count])
        count += 1
    predictions[i] = (predictions[i]/100 + 1) * tmpData[0][0]

np.savetxt("results.txt", predictions[142:])

import csv    
with open('result.csv','w') as fout:
        fieldnames = ['caseid','midprice']
        writer = csv.DictWriter(fout, fieldnames = fieldnames)
        writer.writeheader()
        for i in range(143,1001):
            writer.writerow({'caseid':str(i),'midprice':float(predictions[i-1])})