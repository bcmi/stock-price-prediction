import pickle
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, LSTM, TimeDistributed, RepeatVector
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn import preprocessing
import pandas as pd


def get_class(p):
    if p[0]:
        return 0
    elif p[1]:
        return 1
    else:
        return 2

def loaddata(filename = 'train_data.csv'):
	df = pd.read_csv(filename)
	train_data = df.iloc[ : , 3 : 10]
	return train_data

def smoothed_midprice(p):
    res = []
    for i in range(5):
        res.append(np.mean(p[: i + 1, 0]))
    for i in range(5, len(p)):
        res.append(np.mean(p[i - 4: i + 1, 0]))
    return np.array(res)

def get_dataset(pieces):
    input_set = []
    output_set = []
    actual_price = []
    count1 = 0
    count2 = 0
    count3 = 0
    for piece in pieces:
        p = np.array(piece)
        p = np.hstack((p[ : , : 2], p[ : , 3 : ]))
        
        mb = smoothed_midprice(p)

        for i in range(len(p) - 29):
            x = []
            for j in range(10):
                x = np.append(x, p[i + j])
            actual_price.append((p[i + 9, 0], np.mean(p[i + 10 : i + 30, 0])))
            input_set.append(x)

            ma = np.mean(mb[i + 10 : i + 20])
            alpha = 0.00005
            if mb[i + 9] > ma * (1 + alpha):
                output_set.append(2)
                count3 += 1
            elif mb[i + 9] < ma * (1 - alpha):
                output_set.append(0)
                count1 += 1
            else:
                output_set.append(1)
                count2 += 1

    print(count1, count2, count3)
    output_set = keras.utils.to_categorical(output_set)
    return np.array(input_set), np.array(output_set), np.array(actual_price)

def shuffle(X, Y, actual):
	randomList = np.arange(X.shape[0])
	np.random.shuffle(randomList)
	return X[randomList], Y[randomList], actual[randomList]

def splitDataset(X, Y, actual, rate = 0.2):
    l = len(X)
    return X[int(rate * l) : ], X[ : int(rate * l)], Y[int(rate * l) : ], Y[ : int(rate * l)], actual[int(rate * l) : ], actual[ : int(rate * l)]

def buildModel(shape):
    model = Sequential()
    model.add(Dense(128, input_dim = shape[1],  activation = "relu"))
    model.add(Dropout(0.5))
    model.add(Dense(128, activation = 'relu'))
    model.add(Dropout(0.5))
    model.add(Dense(3, activation = 'softmax'))

    sgd = SGD(lr = 0.005, decay = 1e-6, momentum = 0.9, nesterov = True)
    model.compile(loss = 'categorical_crossentropy',
                optimizer = sgd,
                metrics = ['accuracy'])
    return model

# read midprice of test data
df = pd.read_csv("test_data.csv")
test_data_mid_price = df[["MidPrice"]].values
test_cases = []
for i in range(1000):
    x = test_data_mid_price[i * 10 : (i+1) * 10, : ]
    test_cases.append(x)
test_cases = np.array(test_cases)

print(test_cases.shape)


f = open("cut_data.data", 'rb')
pieces = pickle.load(f)
X, Y, actual = get_dataset(pieces)
X, Y, actual = shuffle(X, Y, actual)
print(X.shape, Y.shape, actual.shape)


X_train, X_val, Y_train, Y_val, actual_train, actual_val = splitDataset(X, Y, actual)

x_scaler = preprocessing.StandardScaler()
X_train = x_scaler.fit_transform(X_train)
X_val = x_scaler.transform(X_val)

model = buildModel(X_train.shape)
callback = EarlyStopping(monitor = 'loss', patience = 2, verbose = 1, mode = "auto")
model.fit(X_train, Y_train, epochs = 6, batch_size = 32, validation_data = (X_val, Y_val), callbacks = [callback], shuffle = True)
model.save("classifier_v1")
score = model.evaluate(X_val, Y_val, batch_size = 64)

res_val = model.predict(X_val)

f2 = open('y_val', 'wb')
f3 = open('actual_val', 'wb')
pickle.dump(res_val, f2)
pickle.dump(actual_val, f3)
print(score)

# test data

test_X_pd = loaddata('test_data.csv').iloc[1420 : , : ]
_test_X_pd = test_X_pd.values
test_diff_volume = []
for i in range(len(_test_X_pd) - 1):
    test_diff_volume.append(_test_X_pd[i + 1, 2] - _test_X_pd[i, 2])
test_diff_volume.append(test_diff_volume[-1])
for i in range(len(test_diff_volume)):
    if(test_diff_volume[i] < 0):
        test_diff_volume[i] = 0
test_X_pd['diff_volume'] = np.array(test_diff_volume)
test_X_raw = test_X_pd.values
test_X_raw = np.hstack((test_X_raw[ : , 0 : 2], test_X_raw[ : , 3 : ]))
test_X_raw = np.reshape(test_X_raw, (-1, 70))
test_x = x_scaler.transform(test_X_raw)

prediction = model.predict(test_x)
# prediction = Y_scaler.inverse_transform(prediction)

print(prediction)
print("length of p:" + str(len(prediction)))

# avg delta
delta = []
for i in range(1000):
	tmpsum = 0.0
	for j in range(len(test_cases[i]) - 1):
		tmpsum += abs(test_cases[i][j + 1][0] - test_cases[i][j][0])
	avg_delta = tmpsum / 9.0
	delta.append(avg_delta)


res = np.zeros((len(prediction)))
for i in range(len(prediction)):
    c = np.argmax(prediction[i])
    res[i] = test_X_raw[i, -7]
    if c==0:
        res[i] += delta[i + 142]
    elif c == 2:
        res[i] -= delta[i + 142]

caseid = list(range(143,1001))
df = pd.DataFrame({'caseid':caseid, 'midprice':res})
df.to_csv('res.csv',index = False, sep = ',')