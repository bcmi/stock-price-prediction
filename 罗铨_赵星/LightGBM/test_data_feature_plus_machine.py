import pandas as pd

data = pd.read_csv('test_data.csv')
print(data,data.shape)
data['difVolume'] = data['Volume'].diff()
for i in range(0,data.shape[0],10):
    data.loc[i,'difVolume'] = 0.0
data['deltaPrice1'] = data['BidPrice1'] - data['AskPrice1']
data['deltaVolume1'] = data['BidVolume1'] - data['AskVolume1']
data.columns = ['ID','Date','Time','MidPrice','LastPrice','Volume','BidPrice1','BidVolume1','AskPrice1','AskVolume1','difVolume','deltaPrice1','deltaVolume1']
print(data,data.shape)
data.to_csv('test_data_feature_plus.csv',index=False)