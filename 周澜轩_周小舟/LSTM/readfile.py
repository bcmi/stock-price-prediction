import csv
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset


BATCH_SIZE = 50


def create_train_dataset(file_name):
    csv_file = csv.reader(open(file_name))
    csv_list = list(csv_file)
    data_set = []
    pred_set = []
    for i in range(len(csv_list)):
        if i >= 100000:
            break
        if i == 0:
            continue

        data = []
        pred = 0
        for j in range(i + 10, i + 30):
            pred += float(csv_list[j][3])
        pred /= 20.0
        pred -= float(csv_list[i + 9][3])
        pred_set.append(pred)
        for j in range(i, i + 10):
            data.append([float(num) for num in csv_list[j][4:]])
        data_set.append(data)
    data_set = torch.from_numpy(normalization(np.array(data_set))).float()
    pred_set = torch.from_numpy(np.array(pred_set)).float()
    return data_set, pred_set


def create_test_dataset(file_name):
    csv_file = csv.reader(open(file_name))
    csv_list = list(csv_file)
    data_set = []
    base_set = []
    for i in range(1, 10991, 11):
        data = []
        for j in range(i, i + 10):
            data.append([float(num) for num in csv_list[j][4:]])
        data_set.append(data)
        base_set.append(float(csv_list[i + 9][3]))
    data_set = torch.from_numpy(normalization(np.array(data_set))).float()
    return data_set, base_set


def create_dsataset(train_file_name, test_file_name):
    x_train, y_train = create_train_dataset(train_file_name)
    x_test, base_dataset = create_test_dataset(test_file_name)
    train = TensorDataset(x_train, y_train)
    test = TensorDataset(x_test)
    train_dataset = DataLoader(dataset=train, batch_size=BATCH_SIZE, shuffle=True)
    test_dateset = DataLoader(dataset=test, batch_size=1, shuffle=False)
    return (train_dataset, test_dateset, base_dataset)


def normalization(array):
    # amax, amin = array.max(), array.min()
    # new_array = (array - amin) / (amax - amin)
    amean = np.mean(array, axis=0)
    astd = np.std(array, axis=0)
    new_array = (array - amean) / astd

    return new_array
