import csv
import numpy as np


def create_train_dataset(file_name):
    csv_list = list(csv.reader(open(file_name)))

    data_set = []
    pred_set = []

    for i in range(1, len(csv_list)):
        if i >= 200000:
            break
        
        data = []
        pred = 0

        for j in range(i + 10, i + 30):
            pred += float(csv_list[j][3])
        pred /= 20.0
        pred -= float(csv_list[i + 9][3]) # average increment.

        pred_set.append(pred)

        for j in range(i, i + 10):
            data.append([float(csv_list[j][4])] + [float(num) for num in csv_list[j][6:]])
        data_set.append(data)

    data_set = normalization(np.array(data_set))
    # pred_set = normalization(np.array(pred_set))
    pred_set = np.array(pred_set)
    return data_set, pred_set


def create_test_dataset(file_name):
    csv_list = list(csv.reader(open(file_name)))

    data_set = []
    base_set = [] # base + predict increment = predict mid price.

    for i in range(1, 10991, 11):
        data = []
        for j in range(i, i + 10):
            data.append([float(csv_list[j][4])] + [float(num) for num in csv_list[j][6:]])
        data_set.append(data)
        base_set.append(float(csv_list[i + 9][3]))

    data_set = normalization(np.array(data_set))
    base_set = np.array(base_set)
    return data_set, base_set


def create_datasets(train_file_name, test_file_name):
    x_train, y_train = create_train_dataset(train_file_name)
    x_test, base_dataset = create_test_dataset(test_file_name)

    return x_train, y_train, x_test, base_dataset


def normalization(array):
    mean = np.mean(array, axis=0)
    std = np.std(array, axis=0)

    new_array = (array - mean) / std

    return new_array


def rMSE_Loss(pred, label):
    tmp = np.mean(np.power(pred - label, 2))
    return np.sqrt(tmp)


