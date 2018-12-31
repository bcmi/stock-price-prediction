
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


dataset_train = pd.read_csv('newdata.csv')
training_set = dataset_train.iloc[:, 1:9].values

from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))


training_set_scaled = sc.fit_transform(training_set)

X = []
y = []
for i in range(10, len(training_set_scaled)):
    X.append(training_set_scaled[i-10:i, 0:7])
    y.append(training_set_scaled[i, 7])

X, y = np.array(X), np.array(y)
y_train, y_test = y[:int(len(training_set_scaled) * 0.9)], y[int(len(training_set_scaled) * 0.9):]
X_train, X_test = X[:int(len(training_set_scaled) * 0.9)], X[int(len(training_set_scaled) * 0.9):]


X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 7))


from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import Flatten
from keras.models import load_model
from keras import optimizers


regressor = Sequential()
regressor.add(LSTM(units = 32, return_sequences = True, input_shape = (X_train.shape[1], 7)))
regressor.add(Flatten())
regressor.add(Dense(units = 1))

adamop = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

regressor.compile(optimizer = adamop, loss = 'mean_squared_error')

history = regressor.fit(X_train, y_train, epochs = 10, batch_size = 64, shuffle = False, validation_data=(X_test, y_test))

regressor.save('my_model.h5')

regressor = load_model('my_model.h5')


file_test = open('test_data.csv','r')
line = file_test.readline()


test = pd.read_csv('test_data.csv')
test_set = test.iloc[1420:, 3:11].values

test_set_scaled = sc.fit_transform(test_set)

mid_prices = test_set[:, 0]

mid_prices = mid_prices.reshape([-1, 1])
standmid = sc.fit_transform(mid_prices)


file_out = open('newresult.csv','w')
file_out.write('caseid,midprice\n')

xtest = []
for i in range(1, len(test_set_scaled) // 10 + 1):
   xtest.append(test_set_scaled[i * 10-10:i * 10, 0:8])

xtest = np.array(xtest)

predicted_file = regressor.predict(xtest)

predicted_file = sc.inverse_transform(predicted_file)


case = 143

for i in range(len(predicted_file)):
    file_out.write(str(case)+','+str(predicted_file[i][0])+'\n')
    case = case + 1

file_out.close()