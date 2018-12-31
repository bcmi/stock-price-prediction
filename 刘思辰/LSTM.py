import numpy as np
import pandas as pd
import tensorflow as tf

time_step = 10
def get_data_set_train():
    file = open('train_data.csv')
    filepd = pd.read_csv(file)
    datapd = filepd.iloc[:, 3:].values
    length = 430039
    datacal = np.zeros(length)
    valid = length - 20
    for i in range(0, valid):
        for j in range(i + 1, i + 21):
            datacal[i] += datapd[j][0]
        datacal[i] /= 20
    data_x, data_y = [], []
    batch = (valid-time_step) // time_step
    for i in range(0, valid - time_step):
        midprice_set0 = datapd[i:i+time_step, :]
        midprice_set = np.copy(midprice_set0)
        temp = np.copy(midprice_set[:, 2])
        for j in range(1, time_step):
            midprice_set[j, 2] = midprice_set[j, 2] - temp[j - 1]
        midprice_set[0, 2] = midprice_set[1, 2]
        data_y.append(datacal[i+time_step - 1] - midprice_set0[-1, 0])
        midprice_set[:, 0] = midprice_set[:, 0] - midprice_set0[-1, 0]
        midprice_set[:, 1] = midprice_set[:, 1] - midprice_set0[-1, 1]
        midprice_set[:, 2] = midprice_set[:, 2] - midprice_set0[-1, 2]
        midprice_set[:, 3] = midprice_set[:, 3] - midprice_set0[-1, 3]
        midprice_set[:, 4] = midprice_set[:, 4] - midprice_set0[-1, 4]
        midprice_set[:, 5] = midprice_set[:, 5] - midprice_set0[-1, 5]
        midprice_set[:, 6] = midprice_set[:, 6] - midprice_set0[-1, 6]
        if np.std(midprice_set[:, 0]) == 0:
            midprice_set[:, 0] = midprice_set[:, 0] - np.mean(midprice_set[:, 0])
        else:
            midprice_set[:, 0] = (midprice_set[:, 0] - np.mean(midprice_set[:, 0])) / np.std(midprice_set[:, 0])
        if np.std(midprice_set[:, 1]) == 0:
            midprice_set[:, 1] = midprice_set[:, 1] - np.mean(midprice_set[:, 1])
        else:
            midprice_set[:, 1] = (midprice_set[:, 1] - np.mean(midprice_set[:, 1])) / np.std(midprice_set[:, 1])
        if np.std(midprice_set[:, 2]) == 0:
            midprice_set[:, 2] = midprice_set[:, 2] - np.mean(midprice_set[:, 2])
        else:
            midprice_set[:, 2] = (midprice_set[:, 2] - np.mean(midprice_set[:, 2])) / np.std(midprice_set[:, 2])
        if np.std(midprice_set[:, 3]) == 0:
            midprice_set[:, 3] = midprice_set[:, 3] - np.mean(midprice_set[:, 3])
        else:
            midprice_set[:, 3] = (midprice_set[:, 3] - np.mean(midprice_set[:, 3])) / np.std(midprice_set[:, 3])
        if np.std(midprice_set[:, 4]) == 0:
            midprice_set[:, 4] = midprice_set[:, 4] - np.mean(midprice_set[:, 4])
        else:
            midprice_set[:, 4] = (midprice_set[:, 4] - np.mean(midprice_set[:, 4])) / np.std(midprice_set[:, 4])
        if np.std(midprice_set[:, 5]) == 0:
            midprice_set[:, 5] = midprice_set[:, 5] - np.mean(midprice_set[:, 5])
        else:
            midprice_set[:, 5] = (midprice_set[:, 5] - np.mean(midprice_set[:, 5])) / np.std(midprice_set[:, 5])
        if np.std(midprice_set[:, 6]) == 0:
            midprice_set[:, 6] = midprice_set[:, 6] - np.mean(midprice_set[:, 6])
        else:
            midprice_set[:, 6] = (midprice_set[:, 6] - np.mean(midprice_set[:, 6])) / np.std(midprice_set[:, 6])
        data_x.append(midprice_set)
    data_x = np.array(data_x)
    data_y = np.array(data_y)
    return data_x, data_y


def get_data_set_test():
    file = open('test_data.csv')
    filepd = pd.read_csv(file)
    datapd = filepd.iloc[:, 3:].values
    length = np.size(datapd, 0)
    size = length // time_step
    data_x, value_value = [], []
    for i in range(0, size):
        midprice_set0 = datapd[i*time_step:(i+1)*time_step, :]
        midprice_set = np.copy(midprice_set0)
        temp = np.copy(midprice_set[:, 2])
        for j in range(1, time_step):
            midprice_set[j, 2] = midprice_set[j, 2] - temp[j - 1]
        midprice_set[0, 2] = midprice_set[1, 2]
        value_value.append(midprice_set0[-1, 0])
        midprice_set[:, 0] = midprice_set[:, 0] - midprice_set0[-1, 0]
        midprice_set[:, 1] = midprice_set[:, 1] - midprice_set0[-1, 1]
        midprice_set[:, 2] = midprice_set[:, 2] - midprice_set0[-1, 2]
        midprice_set[:, 3] = midprice_set[:, 3] - midprice_set0[-1, 3]
        midprice_set[:, 4] = midprice_set[:, 4] - midprice_set0[-1, 4]
        midprice_set[:, 5] = midprice_set[:, 5] - midprice_set0[-1, 5]
        midprice_set[:, 6] = midprice_set[:, 6] - midprice_set0[-1, 6]
        if np.std(midprice_set[:, 0]) == 0:
            midprice_set[:, 0] = midprice_set[:, 0] - np.mean(midprice_set[:, 0])
        else:
            midprice_set[:, 0] = (midprice_set[:, 0] - np.mean(midprice_set[:, 0])) / np.std(midprice_set[:, 0])
        if np.std(midprice_set[:, 1]) == 0:
            midprice_set[:, 1] = midprice_set[:, 1] - np.mean(midprice_set[:, 1])
        else:
            midprice_set[:, 1] = (midprice_set[:, 1] - np.mean(midprice_set[:, 1])) / np.std(midprice_set[:, 1])
        if np.std(midprice_set[:, 2]) == 0:
            midprice_set[:, 2] = midprice_set[:, 2] - np.mean(midprice_set[:, 2])
        else:
            midprice_set[:, 2] = (midprice_set[:, 2] - np.mean(midprice_set[:, 2])) / np.std(midprice_set[:, 2])
        if np.std(midprice_set[:, 3]) == 0:
            midprice_set[:, 3] = midprice_set[:, 3] - np.mean(midprice_set[:, 3])
        else:
            midprice_set[:, 3] = (midprice_set[:, 3] - np.mean(midprice_set[:, 3])) / np.std(midprice_set[:, 3])
        if np.std(midprice_set[:, 4]) == 0:
            midprice_set[:, 4] = midprice_set[:, 4] - np.mean(midprice_set[:, 4])
        else:
            midprice_set[:, 4] = (midprice_set[:, 4] - np.mean(midprice_set[:, 4])) / np.std(midprice_set[:, 4])
        if np.std(midprice_set[:, 5]) == 0:
            midprice_set[:, 5] = midprice_set[:, 5] - np.mean(midprice_set[:, 5])
        else:
            midprice_set[:, 5] = (midprice_set[:, 5] - np.mean(midprice_set[:, 5])) / np.std(midprice_set[:, 5])
        if np.std(midprice_set[:, 6]) == 0:
            midprice_set[:, 6] = midprice_set[:, 6] - np.mean(midprice_set[:, 6])
        else:
            midprice_set[:, 6] = (midprice_set[:, 6] - np.mean(midprice_set[:, 6])) / np.std(midprice_set[:, 6])
        data_x.append(midprice_set)
        '''print(i)
        print(midprice_set)'''
    data_x = np.array(data_x)
    return data_x, value_value


if __name__ == "__main__":
    train_x, train_y = get_data_set_train()
    test_x, value_set = get_data_set_test()
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.LSTM(128, input_shape=(train_x.shape[1], train_x.shape[2]), use_bias=True,
                                   recurrent_initializer='orthogonal', bias_initializer='zeros', unit_forget_bias=True,
                                   return_sequences=True))
    model.add(tf.keras.layers.LSTM(128, return_sequences=False))
    model.add(tf.keras.layers.Dense(128))
    model.add(tf.keras.layers.Dense(1))
    model.compile(optimizer='adam', loss='mae', metrics=['accuracy'])
    model.fit(train_x, train_y, epochs=5, batch_size=128, shuffle=True)
    result = model.predict(test_x)
    size = len(result)
    for i in range(0, size):
        result[i] = result[i] + value_set[i]
        print(result[i])
    result = result.flatten()
    result_valid = result[142:]
    size_valid = np.size(result_valid, 0)
    result_number = []
    for i in range(0, size_valid):
        result_number.append(i + 143)
    result_number = np.array(result_number, dtype=int)
    print(result_valid)
    print(result_number)
    resultpd = pd.DataFrame(data={"caseid":result_number, "midprice":result_valid})
    print(resultpd)
    name = 'resultlstm' +  '.csv'
    resultpd.to_csv(name, index=False)
