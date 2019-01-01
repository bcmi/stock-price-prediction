import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import csv
from keras.layers import LSTM, Dense, Dropout
from keras.models import Sequential
from keras.callbacks import TensorBoard


def get_train_data(date_time, raw_data_scaled, raw_data):
    features, offset = [], []
    base = []  # predict offset, midprice = base + offset
    for i in range(0, raw_data.shape[0] - 20):
        day1, hour1 = date_time[i][0][-2:], int(date_time[i][1][:2])
        day2, hour2 = date_time[i+9][0][-2:], int(date_time[i+9][1][:2])
        
        # drop invalid(discontinuous) trainning data
        if day1 != day2 or (hour1 < 12 and hour2 > 11) or (hour1 > 11 and hour2 < 12):
            continue
        
        features.append(raw_data_scaled[i:i+10])
        last_midprice = raw_data[i+9][0]
        base.append(last_midprice)
        offset.append(raw_data[i+10:i+30, 0].mean() - last_midprice)

    return features, offset, base


def get_test_data(raw_data_scaled, raw_data):
    features = []
    base = []
    for i in range(1000):
        features.append(raw_data_scaled[i*10:(i+1)*10])
        base.append(raw_data[i*10 + 9][0])
    
    return features, base


def write_file(filename, predict):
    with open(filename, "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["caseid", "midprice"])
        for i in range(142, len(predict)):
            writer.writerow([(i + 1), predict[i]])


def lstm(x_train, y_train, x_test, y_test, features_test):
	regressor = Sequential()
	regressor.add(LSTM(units=50,return_sequences = True,input_shape = (10, 7)))
	regressor.add(Dropout(0.2))
	regressor.add(LSTM(units=50))
	regressor.add(Dropout(0.2))
	regressor.add(Dense(units=50,activation='relu'))
	regressor.add(Dense(units=1))
	regressor.compile(optimizer = "adam",loss = "mean_squared_error")

	tb = TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_images=True)
	regressor.fit(x_train, y_train, epochs=5,batch_size=32, verbose=2, callbacks=[tb])

	regressor.save("1.h5")

	y_predict = regressor.predict(x_test)
	mse = mean_squared_error(y_test, y_predict)
	print(mse)

	offset_predict = regressor.predict(np.array(features_test))
	return offset_predict


if __name__ == '__main__':
	train_file = pd.read_csv('train_data.csv')
	test_file = pd.read_csv('test_data.csv')

	raw_train_data = train_file[['MidPrice', 'LastPrice', 'Volume', 'BidPrice1', 'BidVolume1', 'AskPrice1', 'AskVolume1']].values
	raw_test_data = test_file[['MidPrice', 'LastPrice', 'Volume', 'BidPrice1', 'BidVolume1', 'AskPrice1', 'AskVolume1']].values

	scaler = StandardScaler()
	# scaler = MinMaxScaler()

	raw_train_data_scaled = scaler.fit_transform(raw_train_data)
	raw_test_data_scaled = scaler.transform(raw_test_data)

	raw_train_data_time_and_date = train_file[['Date', 'Time']].values

	features_train, offset_train, base_train = get_train_data(raw_train_data_time_and_date, raw_train_data_scaled, raw_train_data)
	features_test, base_test = get_test_data(raw_test_data_scaled, raw_test_data)

	x_train, x_test = np.array(features_train[4000:]), np.array(features_train[:4000])
	y_train, y_test = np.array(offset_train[4000:]), np.array(offset_train[:4000])

	offset_predict = lstm(x_train, y_train, x_test, y_test, features_test)
	final_predict = list(np.array(base_test) + offset_predict.flatten())
	write_file('lstm_result.csv', final_predict)
