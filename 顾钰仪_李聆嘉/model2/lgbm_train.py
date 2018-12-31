# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import lightgbm as lgb

params = {
    'task': 'train',
    'objective': 'binary',
    'metrics': 'auc',
    'early_stopping_rounds': 100,
    'n_estimators': 500,
    'learning_rate': 0.05,
}

def train():
    # Load file
    file = pd.read_csv('train_data.csv')
    data = file[['MidPrice', 'LastPrice', 'BidPrice1', 'BidVolume1', 'AskPrice1', 'AskVolume1']]
    labels = file[['MidPrice']]

    # Add new feature
    imbalance = data['BidVolume1'] - data['AskVolume1']
    new_col = pd.DataFrame(imbalance, columns=["Imbalance"])
    res = pd.concat([data, new_col], axis=1)
    volumes = res[['BidVolume1', 'AskVolume1', "Imbalance"]]
    volumes = np.array(volumes)
    new_vol = pd.DataFrame(volumes, columns=['BidVolume1', 'AskVolume1', "Imbalance"])
    res[['BidVolume1', 'AskVolume1', "Imbalance"]] = new_vol

    data = np.array(res)
    labels = np.array(labels)

    L = len(labels) - 30
    f = list()
    l = list()
    for i in range(L):
        label_for_case = np.mean(labels[i + 10: i + 30]) - data[i + 9][0]
        if (label_for_case >= 0):
            label_for_case = 1
        elif (label_for_case < 0):
            label_for_case = 0
        l.append(label_for_case)
    for i in range(L):
        pivot = data[i + 9][0]
        pivot_array = np.array([pivot, pivot, pivot, 0, pivot, 0, 0])
        feature_for_case = data[i: i + 10] - pivot_array
        f.append(feature_for_case.reshape(70))

    # Train
    split = int(len(labels) * 0.8)
    lgb_train = lgb.Dataset(np.array(f)[:split], np.array(l)[:split])
    lgb_val = lgb.Dataset(np.array(f)[split:], np.array(l)[split:])
    gbm = lgb.train(params, lgb_train, 50, verbose_eval=100, valid_sets=[lgb_train, lgb_val], early_stopping_rounds=200)
    gbm.save_model("model.h5")

if __name__ == '__main__':
    train()

