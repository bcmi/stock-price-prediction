from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

# 预处理数据，去掉日期列
dataset = read_csv('test_data.csv', header=0, index_col=0)
dataset.drop('Date', axis=1, inplace=True)
dataset.drop('Time', axis=1, inplace=True)
dataset.drop('Volume', axis=1, inplace=True)
dataset.drop('LastPrice', axis=1, inplace=True)
dataset.drop('MidPrice', axis=1, inplace=True)
dataset=dataset.loc[430039+50*142:]
dataset['Y'] = 0
print(dataset.head(5))
dataset.to_csv('test.csv')
ds = read_csv('train_data.csv', header=0, index_col=0)
ds['Y'] = 0
for i in range(1,21):
    ds['Y']=ds['Y']+ds['MidPrice'].shift(-i)/20.0
ds.drop('Date', axis=1, inplace=True)
ds.drop('Time', axis=1, inplace=True)
ds.drop('Volume', axis=1, inplace=True)
ds.drop('LastPrice', axis=1, inplace=True)
ds.drop('MidPrice', axis=1, inplace=True)
ds=ds.loc[:430038-20]
ds.to_csv('train.csv')
