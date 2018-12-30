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
dataset = read_csv('train_data.csv', header=0, index_col=0)
dataset.drop('Date', axis=1, inplace=True)
dataset.drop('Time', axis=1, inplace=True)
print(dataset.head(5))
dataset.to_csv('train.csv')
