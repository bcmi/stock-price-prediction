import time
import datetime
import csv
import numpy as np
from sklearn import preprocessing
data = []
title = ""

# with open('train_data.csv') as f:
#     reader = csv.reader(f)
#     title = next(reader)
#     title = title[3:]
#     title.append("day_time")
#     title.append("weekday")
#     prev_date = ""
#     block = []
#     prev = []
#     for row in reader:
#         if prev[1:] == row[1:] or row[2].startswith("11:30:") or row[2].startswith("12:"):
#             continue
#         vec = [float(i) for i in row[3:]]
#         vec.append(float(3600*float(row[2][0:2])+60*float(row[2][3:5])+float(row[2][6:8])))
#         vec.append(float(datetime.datetime(int(row[1][0:4]),int(row[1][5:7]), int(row[1][8:10])).weekday()+1))
#         data.append(vec)

def get_train_unix(row):
    return float(time.mktime(time.strptime(row[1]+' '+row[2],"%Y-%m-%d %H:%M:%S")))#time.mktime(time.strptime(row[1]+' '+row[2],"%Y/%m/%d %H:%M:%S")))

def get_test_unix(row):
    return float(time.mktime(time.strptime(row[1]+' '+row[2],"%Y/%m/%d %H:%M:%S")))


with open('train_data.csv') as f:
    reader = csv.reader(f)
    title = next(reader)
    title = title[3:]
    #title.append("day_time")
    title.append("day_hour")
    title.append("weekday")
    #title = ["id"] + title
    prev_date = ""
    block = []
    prev = []
    for row in reader:
        # line = row[2:]
        if prev and (prev[1:] == row[1:] or row[2].startswith("11:30:") or row[2].startswith("12:")):
            continue

        if row[1] != prev_date or (len(prev)!=0 and (prev[2].startswith("11") and row[2].startswith("13"))):
            prev_date = row[1]
            data.append(block)
            block = []
        prev = row
        # vec = [float(i) for i in row[3:5]] + [float(i) for i in row[6:]]
        vec = [float(i) for i in row[3:]]
        #vec.append(float(3600*float(row[2][0:2])+60*float(row[2][3:5])+float(row[2][6:8])))
        if(row[2][0:2] == "15"):
            vec.append(14.0)
        else:
            vec.append(float(row[2][0:2]))
        vec.append(float(datetime.datetime(int(row[1][0:4]),int(row[1][5:7]), int(row[1][8:10])).weekday()+1))
        #engineering!!! 0 MidPrice,1 LastPrice,2 Volume,3 BidPrice1,4 BidVolume1, 5 AskPrice1, 6 AskVolume1, 7 day_hour, 8 weekday
        vec.append(vec[3]*vec[4])
        vec.append(vec[5]*vec[6])
        vec.append(vec[3]*vec[4]-vec[5]*vec[6])
        vec.append(2*(vec[3]-vec[5])/(vec[3]+vec[5]))
        vec.append((vec[4]+vec[6])/2)
        vec.append((vec[3]-vec[5])/vec[-1])
        vec.append(vec[4]-vec[6])
        vec.append(vec[3]-vec[5])
        vec.append(vec[3]/vec[0] - 1)
        vec.append(vec[5]/vec[0]-1)
        #vec.append(get_train_unix(row))
        block.append(vec)
    title.extend(["BidTotal","AskTotal","TotalDif", "RelaPriceDif", "Depth", "K", "VolDif", "BidAskDif", "RelBid", "RelAsk"])
    data = data[1:]
    data.append(block)


# title.extend(["Mid_Price_Ratio", "Last_Price_Ratio", "Bid_Price_Ratio", "Ask_Price_Ratio"])
# prev = []
# for term in data:
#     if(not prev):
#         print("no")
#         term.append(0)
#     else:
#         term.append(term[0]/prev[0]-1)
#     for i in range(len(term)):
#         if i in [1,3,5]:
#             if(not prev):
#                 term.append(0)
#             else:
#                 term.append(term[i]/prev[i]-1)
#     prev = term[:-4]


# with open('bitraining_set_no_norm.csv','w') as of:
#     for t in title[:-1]:
#         of.write(t+',')
#     of.write(title[-1]+'\n')
#     for d in data:
#         #of.write(str(d[0])+",")
#         for i in range(1,len(d)-1):
#             of.write(str(d[i])+',')
#         of.write(str(d[-1])+'\n')




data2 = []
with open("test_data.csv") as f:
    reader = csv.reader(f)
    t = next(reader)
    for row in reader:
        if not row[0]:
            continue
        vec = [float(i) for i in row[3:]]
        day_time = row[2].split(":")
        day_time = [float(i) for i in day_time]
        #vec.append(float(3600*day_time[0]+60*day_time[1]+day_time[2]))
        if(day_time[0]==12):
            day_time[0]=11
        vec.append(day_time[0])
        weekday = row[1].split("/")
        weekday = [int(i) for i in weekday]
        vec.append(float(datetime.datetime(weekday[0],weekday[1],weekday[2]).weekday()+1))
        #feature engineering
        vec.append(vec[3]*vec[4])
        vec.append(vec[5]*vec[6])
        vec.append(vec[3]*vec[4]-vec[5]*vec[6])
        vec.append(2*(vec[3]-vec[5])/(vec[3]+vec[5]))
        vec.append((vec[4]+vec[6])/2)
        vec.append((vec[3]-vec[5])/vec[-1])
        vec.append(vec[4]-vec[6])
        vec.append(vec[3]-vec[5])
        vec.append(vec[3]/vec[0] - 1)
        vec.append(vec[5]/vec[0]-1)
        #vec.append(get_test_unix(row))
        data2.append(vec)


# for term in data2:
#     if(not prev):
#         print("no")
#         term.append(0)
#     else:
#         term.append(term[0]/prev[0]-1)
#     for i in range(len(term)):
#         if i in [1,3,5]:
#             if(not prev)
#                 term.append(0)
#             else:
#                 term.append(term[i]/prev[i]-1)
#     prev = term[:-4]



# with open('training_set_no_norm.csv','w') as of:
#     for t in title[:-1]:
#         of.write(t+',')
#     of.write(title[-1]+'\n')
#     for d in data:
#         #of.write(str(d[0])+",")
#         for i in range(len(d)-1):
#             of.write(str(d[i])+',')
#         of.write(str(d[-1])+'\n')



with open('train_data_2.csv','w') as of:
    for t in title[:-1]:
        of.write(t+',')
    of.write(title[-1]+'\n')
    for block in data:
        for d in block:
            for i in range(len(d)-1):
                of.write(str(d[i])+',')
            of.write(str(d[-1])+'\n')
        of.write(','*(len(d)-1) + '\n')

with open('test_data_2.csv','w') as of:
    for t in title[:-1]:
        of.write(t+',')
    of.write(title[-1]+'\n')
    for d in data2:
        for i in range(len(d)-1):
            of.write(str(d[i])+',')
        of.write(str(d[-1])+'\n')

# X = np.array(data+data2)
# mean = np.mean(X,axis=0)
# std = np.std(X,axis=0)
# print(mean)
# print(std)

# data = np.array(data)
# data2 = np.array(data2)

# data[:] = (data[:]- mean) / std
# data2[:] = (data2[:]-mean)/std

# p_mean =  3.48102319
# p_std = 1.55847408e-01



# with open('training_set.csv','w') as of:
#     for t in title[:-1]:
#         of.write(t+',')
#     of.write(title[-1]+'\n')
#     for d in data:
#         #of.write(str(d[0])+",")
#         for i in range(len(d)-1):
#             of.write(str(d[i])+',')
#         of.write(str(d[-1])+'\n')

# with open('test_set.csv','w') as of:
#     for t in title[:-1]:
#         of.write(t+',')
#     of.write(title[-1]+'\n')
#     for d in data2:
#         for i in range(len(d)-1):
#             of.write(str(d[i])+',')
#         of.write(str(d[-1])+'\n')