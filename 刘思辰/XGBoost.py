import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
import numpy as np
from sklearn.preprocessing import Imputer
#from sklearn import XGBModel

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
    for i in range(0, batch):
        midprice_set0 = datapd[i*time_step:(i+1)*time_step, :]
        midprice_set = np.copy(midprice_set0)
        temp = np.copy(midprice_set[:, 2])
        for j in range(1, time_step):
            midprice_set[j, 2] = midprice_set[j, 2] - temp[j - 1]
        midprice_set[0, 2] = midprice_set[1, 2]
        data_y.append(datacal[(i+1)*time_step - 1] - midprice_set0[-1, 0])
        midprice_set[:, 0] = midprice_set[:, 0] - midprice_set0[-1, 0]
        midprice_set[:, 1] = midprice_set[:, 1] - midprice_set0[-1, 1]
        midprice_set[:, 3] = midprice_set[:, 3] - midprice_set0[-1, 3]
        midprice_set[:, 5] = midprice_set[:, 5] - midprice_set0[-1, 5]
        if np.std(midprice_set[:, 2]) == 0:
            midprice_set[:, 2] = midprice_set[:, 2] - np.mean(midprice_set[:, 2])
        else:
            midprice_set[:, 2] = (midprice_set[:, 2] - np.mean(midprice_set[:, 2])) / np.std(midprice_set[:, 2])
        if np.std(midprice_set[:, 4]) == 0:
            midprice_set[:, 4] = midprice_set[:, 4] - np.mean(midprice_set[:, 4])
        else:
            midprice_set[:, 4] = (midprice_set[:, 4] - np.mean(midprice_set[:, 4])) / np.std(midprice_set[:, 4])
        if np.std(midprice_set[:, 6]) == 0:
            midprice_set[:, 6] = midprice_set[:, 6] - np.mean(midprice_set[:, 6])
        else:
            midprice_set[:, 6] = (midprice_set[:, 6] - np.mean(midprice_set[:, 6])) / np.std(midprice_set[:, 6])
        midprice_set = midprice_set.flatten()
        data_x.append(midprice_set)
    data_x = np.array(data_x)
    data_y = np.array(data_y)
    print(data_x)
    print(data_y)
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
        midprice_set[:, 3] = midprice_set[:, 3] - midprice_set0[-1, 3]
        midprice_set[:, 5] = midprice_set[:, 5] - midprice_set0[-1, 5]
        if np.std(midprice_set[:, 2]) == 0:
            midprice_set[:, 2] = midprice_set[:, 2] - np.mean(midprice_set[:, 2])
        else:
            midprice_set[:, 2] = (midprice_set[:, 2] - np.mean(midprice_set[:, 2])) / np.std(midprice_set[:, 2])
        if np.std(midprice_set[:, 4]) == 0:
            midprice_set[:, 4] = midprice_set[:, 4] - np.mean(midprice_set[:, 4])
        else:
            midprice_set[:, 4] = (midprice_set[:, 4] - np.mean(midprice_set[:, 4])) / np.std(midprice_set[:, 4])
        if np.std(midprice_set[:, 6]) == 0:
            midprice_set[:, 6] = midprice_set[:, 6] - np.mean(midprice_set[:, 6])
        else:
            midprice_set[:, 6] = (midprice_set[:, 6] - np.mean(midprice_set[:, 6])) / np.std(midprice_set[:, 6])
        midprice_set = midprice_set.flatten()
        data_x.append(midprice_set)
        '''print(i)
        print(midprice_set)'''
    data_x = np.array(data_x)
    print(data_x)
    print(value_value)
    return data_x, value_value


if __name__ == "__main__":
    train_x, train_y = get_data_set_train()
    test_x, value_set = get_data_set_test()
    model = xgb.XGBRegressor(max_depth=8, learning_rate=0.1, n_estimators=150, silent=False, objective='reg:linear')
    model.fit(train_x, train_y)
    result = model.predict(test_x)
    size = len(result)
    for i in range(0, size):
        print(result[i])
        result[i] = result[i] + value_set[i]
    result_valid = result[142:]
    size_valid = np.size(result_valid, 0)
    result_number = []
    for i in range(0, size_valid):
        result_number.append(i + 143)
    result_number = np.array(result_number, dtype=int)
    resultpd = pd.DataFrame(data={"caseid":result_number, "midprice":result_valid})
    print(resultpd)
    name = 'resultxgboost' +  '.csv'
    resultpd.to_csv(name, index=False)
