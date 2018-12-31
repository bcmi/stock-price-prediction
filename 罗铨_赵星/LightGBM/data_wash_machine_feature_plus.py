import pandas as pd
import numpy as np
import time
import csv

def get_time_dif(data1,time1,data2,time2):
    timearray1 = time.strptime(data1+time1,"%Y-%m-%d%H:%M:%S")
    timearray2 = time.strptime(data2+time2,"%Y-%m-%d%H:%M:%S")
    timestamp1 = int(time.mktime(timearray1))
    timestamp2 = int(time.mktime(timearray2))
    # print(timestamp1,timestamp2,timestamp2-timestamp1)
    return timestamp2 - timestamp1

org_data = pd.read_csv("train_data.csv").values
print('org_shape:',org_data.shape)
print(org_data,type(org_data))

with open('washed_train_data_feature_plus.csv','w',newline='') as csv_file:
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['ID','Date','Time','MidPrice','LastPrice','Volume','BidPrice1','BidVolume1','AskPrice1','AskVolume1','difVolume','deltaPrice1','deltaVolume1'])
    out_data = []
    read_pointer = 0
    last_row = None
    totalout_row = 0
    totalout_interval = 0
    while read_pointer < org_data.shape[0]:
        row = org_data[read_pointer]
        if type(last_row) != type(None):
            if row[1] != last_row[1] or get_time_dif(last_row[1],last_row[2],row[1],row[2]) != 3:
                if len(out_data) >= 30:
                    totalout_row += len(out_data)
                    totalout_interval += 1
                    for out_item in out_data:
                        csv_writer.writerow(out_item)
                    csv_writer.writerow([])
                out_data = []
        if out_data == []:
            out_data.append(np.append(row,[0,row[6]-row[8],row[7]-row[9]]))
        else:
            out_data.append(np.append(row,[row[5]-last_row[5],row[6]-row[8],row[7]-row[9]]))
        last_row = row
        read_pointer += 1
    if len(out_data) >= 30:
        totalout_row += len(out_data)
        totalout_interval += 1
        for out_item in out_data:
            csv_writer.writerow(out_item)
    print('num_after_wash:',totalout_row)
    print('total_interval:',totalout_interval)