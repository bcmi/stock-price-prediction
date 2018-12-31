import pandas as pd
import numpy as np
import csv
 

 
 
f=open('train_data.csv') 
df=pd.read_csv(f)     #读入股票数据
midprice=df.iloc[:,3].values#取第3列

length=len(midprice)
 

avar_last10=[]
avar_mid20=[]
for i in range(length-29):
    sum_last10=0
    sum_mid20=0
    for j in range(8,10):
        if j ==8:
            sum_last10 += 0.2*midprice[i+j]
        if j ==9:
            sum_last10 += 0.8*midprice[i+j]
    for k in range(20):
        sum_mid20 += midprice[i+10+k]
    avar_last10.append(sum_last10)
    avar_mid20.append(sum_mid20/20)
 
z1 = np.polyfit(avar_last10, avar_mid20, 1)
 

   
t=open('test_data.csv') 
dt=pd.read_csv(t)     #读入股票数据
lastprice_test=dt.iloc[:,3].values  #取第4列
avar_last_test=[]
for i in range (1000):
    lastsum=0
    for j in range(8,10):
        if j==8:
            lastsum+=0.2*lastprice_test[i*11+j]
        else:
            lastsum+=0.8*lastprice_test[i*11+j]
    avar_last_test.append(lastsum )

 

csvFile = open("five.csv", "a", newline='')            #创建csv文件
writer = csv.writer(csvFile)                  #创建写的对象                        
writer.writerow(["caseid","midprice"])     #写入列的名称4
for i in range (143,1001):
    last_test=avar_last_test[i-1]
    mid_test=(last_test)*z1[0]+z1[1] 
    print(mid_test)
    writer.writerow([i,mid_test])
        
 
csvFile.close()