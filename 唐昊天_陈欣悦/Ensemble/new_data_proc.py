import torch
import torch.utils.data as data
import csv
import pickle
import numpy as np
import datetime
import copy
def get_next_time(t):
    x = datetime.datetime.strptime(t, "%H:%M:%S") + datetime.timedelta(seconds=3)
    new_t = datetime.datetime.strftime(x, "%H:%M:%S")
    return new_t

def get_time_diff(t1, t2):
    x = datetime.datetime.strptime(t1, "%H:%M:%S") 
    y = datetime.datetime.strptime(t2, "%H:%M:%S")
    return (x-y).seconds

def special_range(date):
    date = date.split('-')
    y = int(date[0])
    m = int(date[1])
    d = int(date[2])
    if m < 8:
        return 0
    elif m > 8 or d >= 20:
        return 1
    
def read_data_test(phase="test"):
    print("========Reading Test Features========")
    f = open("all/%s_data.csv"%phase, "r")
    reader = csv.DictReader(f)
    fieldnames = ['Date', 'Time', 'MidPrice', 'LastPrice', 'Volume', 'BidPrice1', 'BidVolume1', 'AskPrice1', 'AskVolume1']

    midPrices = []
    dates = []
    times = []
    lastPrices = []
    volumes = []
    bidPrices = []
    bidVolumes = []
    askPrices = []
    askVolumes = []
    dic = {'Date': dates, 'Time': times, 'MidPrice': midPrices, 'LastPrice': lastPrices,\
        'Volume': volumes, 'BidPrice1': bidPrices, 'BidVolume1': bidVolumes, 'AskPrice1': askPrices,\
        'AskVolume1': askVolumes}
    
    processed_date = {}
    day_start_time = None
    afternoon_start_time = None
    cur_processing_date = None
    reader = list(reader)
    i = 0
    
    all_data = []
    cur_day_data = []
    
    day_flag = 0
    while i < len(reader):
        row = reader[i]
        cur_hr = int(row['Time'][:2])
        if row['Date'] not in processed_date.keys():
            day_flag = 0
            if len(cur_day_data) > 0:
                all_data.append(np.array(cur_day_data).reshape(-1,7))
                cur_day_data = []
            cur_processing_date = row['Date']
            print('Processing date %s ... '%cur_processing_date)
            processed_date[row['Date']] = 1
        #elif cur_hr >= 13 and day_flag == 0:
        #    all_data.append(np.array(cur_day_data).reshape(-1,7))
        #    cur_day_data = []
        #    day_flag = 1
        for field in fieldnames:
            if not 'Date' in field and not 'Time' in field:
                cur_day_data.append(float(row[field]))
        
        i+=1
    all_data.append(np.array(cur_day_data).reshape(-1,7))
    
    print("========End Reading Test Features========")
    return all_data
        
def read_data(phase="train"):
    print("========Reading Training Features========")
    f = open("all/%s_data.csv"%phase, "r")
    reader = csv.DictReader(f)
    fieldnames = ['Date', 'Time', 'MidPrice', 'LastPrice', 'Volume', 'BidPrice1', 'BidVolume1', 'AskPrice1', 'AskVolume1']

    processed_date = {}
    day_start_time = None
    afternoon_start_time = None
    cur_processing_date = None
    reader = list(reader)
    #reader = list(filter(lambda x: x['Time'][:2]!="12", reader))
    i = 0
    
    all_data = []
    cur_day_data = []
    
    ff = open('dump.txt','w')
    while i < len(reader):
        row = reader[i]
        cur_hr = int(row['Time'][:2])
        
        if row['Date'] not in processed_date.keys():
            if len(cur_day_data) > 0:
                all_data.append(np.array(cur_day_data).reshape(-1,7))
                cur_day_data = []
            cur_processing_date = row['Date']
            print('Processing date %s ... '%cur_processing_date)
            processed_date[row['Date']] = 1
            afternoon_start_time = None
            while 1:
                row = reader[i]
                cur_hr = int(row['Time'][:2])
                if cur_hr > 11:
                    i -= 1
                    break
                    
                if special_range(row['Date']) or get_time_diff(reader[i+1]['Time'], row['Time']) == 3 and get_time_diff(reader[i+2]['Time'], reader[i+1]['Time']) == 3 :
                    day_start_time = row['Time']
                    for field in fieldnames:
                        if not 'Date' in field and not 'Time' in field:
                            cur_day_data.append(float(row[field]))
                        
                    break
                else:
                    i += 1
            continue
            
        elif cur_hr >= 13 and afternoon_start_time is None:
            #all_data.append(np.array(cur_day_data).reshape(-1,7))
            #cur_day_data = []
            while 1:
                if i >= len(reader):
                    break
                row = reader[i]
                if row['Date'] != cur_processing_date:
                    i -= 1
                    break
                if special_range(row['Date']) or get_time_diff(reader[i+1]['Time'], row['Time']) == 3 and get_time_diff(reader[i+2]['Time'], reader[i+1]['Time']) == 3:
                    afternoon_start_time = row['Time']
                    for field in fieldnames:
                        if not 'Date' in field and not 'Time' in field:
                            cur_day_data.append(float(row[field]))
                        
                    break
                else:
                    i += 1
            continue
        
        if i >= len(reader):
            break
        row = reader[i]
        cur_hr = int(row['Time'][:2])
            
        if cur_hr <= 11:
            while 1:
                row = reader[i]
                cur_hr = int(row['Time'][:2])
                if cur_hr > 11:
                    i -= 1
                    break
                if get_time_diff(row['Time'], day_start_time) % 3 != 0 or get_time_diff(row['Time'], day_start_time) < 3:
                    i += 1
                else:
                    break
            
            day_start_time = row['Time']
            if cur_hr <= 11:
                for field in fieldnames:
                    if not 'Date' in field and not 'Time' in field:
                        cur_day_data.append(float(row[field]))

        
        elif cur_hr >= 13:
            while 1:
                if i >= len(reader):
                    break
                row = reader[i]
                if row['Date'] != cur_processing_date:
                    i -= 1
                    break
                if get_time_diff(row['Time'], afternoon_start_time) % 3 != 0 or get_time_diff(row['Time'], afternoon_start_time) < 3:
                    i += 1
                else:
                    
                    break
            afternoon_start_time = row['Time']
            for field in fieldnames:
                if not 'Date' in field and not 'Time' in field:
                    cur_day_data.append(float(row[field]))
        
        for field in row:
            ff.write(str(row[field])+" ")
        ff.write("\n")
        i += 1
        
    
    #ff.close()
    f.close()
    all_data.append(np.array(cur_day_data).reshape(-1,7))
    print("========End Reading Training Features========")
    return all_data


def batch_formulation(data, phase="train", seed=2018):
    np.random.seed(seed)
    batches = []
    labels = []
    margins = []
    
    if phase == "train":
        for i in range(len(data)):
            cur_data = data[i]
            #cur_data: MidP, LastP, V, BidP, BidV, AskP, AskV
            #to_append: LastP, V, BidP, BidV, AskP, AskV
            for t in range(0,len(cur_data)-30,10):
                to_append = copy.deepcopy(cur_data[t:t+10,1:])
                margins.append([cur_data[t:t+10,0].mean(), cur_data[t:t+10,3].mean(), cur_data[t:t+10,5].mean()])
                #margins.append([cur_data[t+9,0], cur_data[t+9,3], cur_data[t+9,5]])
                #print(margins[-1])
                to_append[:,0] = to_append[:,0]-to_append[:,0].mean()
                to_append[:,2] = to_append[:,2]-to_append[:,2].mean()
                to_append[:,4] = to_append[:,4]-to_append[:,4].mean()


                to_append[:,1] = np.ediff1d(cur_data[t:t+10,2], to_begin=0)
                to_append[:,1] = (to_append[:,1]-to_append[:,1].mean())/(to_append[:,1].std()+1e-5)
                to_append[:,3] = (to_append[:,3]-to_append[:,3].mean())/(to_append[:,3].std()+1e-5)
                to_append[:,5] = (to_append[:,5]-to_append[:,5].mean())/(to_append[:,5].std()+1e-5)
                
                
                
                labels.append(cur_data[t+10:t+30,0].mean() - cur_data[t+9, 0])
                #reshape tailored for XGB
                batches.append(np.hstack(to_append.reshape(-1)))
    else:
        for i in range(len(data)):
            cur_data = data[i]
            for t in range(0, len(cur_data)-9, 10):
                to_append = copy.deepcopy(cur_data[t:t+10,1:])
                margins.append([cur_data[t:t+10,0].mean(), cur_data[t:t+10,3].mean(), cur_data[t:t+10,5].mean()])
                #margins.append([cur_data[t+9,0], cur_data[t+9,3], cur_data[t+9,5]])
                #print(margins[-1])
                to_append[:,0] = to_append[:,0]-to_append[:,0].mean()
                to_append[:,2] = to_append[:,2]-to_append[:,2].mean()
                to_append[:,4] = to_append[:,4]-to_append[:,4].mean()


                to_append[:,1] = np.ediff1d(cur_data[t:t+10,2], to_begin=0)
                to_append[:,1] = (to_append[:,1]-to_append[:,1].mean())/(to_append[:,1].std()+1e-5)
                to_append[:,3] = (to_append[:,3]-to_append[:,3].mean())/(to_append[:,3].std()+1e-5)
                to_append[:,5] = (to_append[:,5]-to_append[:,5].mean())/(to_append[:,5].std()+1e-5)
                
                #batches.append(to_append.reshape(-1))
                batches.append(to_append.reshape(-1))
    
    all_batches, all_labels, all_margins = np.array(batches), np.array(labels), np.array(margins)
    if phase == "train":
        idx_perm = np.random.permutation(len(all_batches))
        all_train = []
        #added
        all_holdout_idx = []
        fifth_len = len(all_batches)//5
        for i in range(5):
            val_idx = idx_perm[fifth_len*i:fifth_len*(i+1)]
            train_idx = np.setdiff1d(idx_perm, val_idx)
            
            train_batches = all_batches[train_idx]
            train_labels = all_labels[train_idx]
            train_margins = all_margins[train_idx]

            val_batches = all_batches[val_idx]
            val_labels = all_labels[val_idx]
            val_margins = all_margins[val_idx]
            all_train.append((train_batches, train_labels, train_margins, val_batches, val_labels, val_margins))
            all_holdout_idx.append(val_idx)
        return all_train, all_holdout_idx, all_labels
   
    else:
        return all_batches, all_margins
    """
    return all_batches, all_labels, all_margins
batches, labels, train_margins= batch_formulation(data)
    """