import csv
from sklearn import preprocessing

# get data
with open('dataset/train_data.csv','r') as train_file:
    reader = csv.DictReader(train_file)
    MidPrice = [row['MidPrice'] for row in reader]
    Volume = [row['Volume'] for row in reader]
    BidPrice = [row['BidPrice1'] for row in reader]
    BidVolume = [row['BidVolume1'] for row in reader]
    AskPrice = [row['AskPrice1'] for row in reader]
    AskVolume1 = [row['AskVolume1'] for row in reader]

with open('dataset/test_data.csv','r') as test_file:
    reader = csv.DictReader(test_file)
    Test_MidPrice = [row['MidPrice'] for row in reader]
    Test_Volume = [row['Volume'] for row in reader]
    Test_BidPrice = [row['BidPrice1'] for row in reader]
    Test_BidVolume = [row['BidVolume1'] for row in reader]
    Test_AskPrice = [row['AskPrice1'] for row in reader]
    Test_AskVolume1 = [row['AskVolume1'] for row in reader]

def z_score(x, axis):
    x = np.array(x).astype(float)
    xr = np.rollaxis(x, axis=axis)
    mean = np.mean(x, axis=axis)
    std = np.std(x, axis=axis)
    xr -= np.mean(x, axis=axis)
    xr /= np.std(x, axis=axis)
    # print(x)
    return x,mean,std

def re_z_score(x, mean, std):
    for i in x.size():
        x[i] *= std
        x[i] += mean
    return x