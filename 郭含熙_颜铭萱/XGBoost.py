# This method uses xgboost
import pandas as pd
import csv
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb

def load_csv_data(filename):
    file = pd.read_csv(filename)
    data = file.loc[:, ['caseid', 'LastPrice', 'Volume', 'BidPrice1', 'BidVolume1', 'AskPrice1', 'AskVolume1', 'MidPrice']].values
    predict = file.loc[:, ['LastPrice', 'Volume', 'BidPrice1', 'BidVolume1', 'AskPrice1', 'AskVolume1', 'MidPrice']].values
    data = data.astype(np.float32)
    predict = predict.astype(np.float32)
    return data, predict

def load_csv_test_data(filename):
    file = pd.read_csv(filename)
    data = file.loc[:, ['LastPrice', 'Volume', 'BidPrice1', 'BidVolume1', 'AskPrice1', 'AskVolume1', 'MidPrice']].values
    data = data.astype(np.float32)
    return data

def load_csv_test_data2(filename):
    file = pd.read_csv(filename)
    data = file.loc[:, ['LastPrice', 'Volume', 'BidPrice1', 'BidVolume1', 'AskPrice1', 'AskVolume1', 'MidPrice']].values
    data = data.astype(np.float32)
    return data


def process_data_test(data):
    feature = []
    for j in range(0, len(data), 10):
        mean = np.mean(data[j:j+10], axis=0)
        std = np.mean(data[j:j+10], axis=0)
        data[j:j+10] = (data[j:j+10] - mean) / (std + 0.000000000000001)
        tmp2 = []
        for k in range(10):
            for item in data[j+k]:
                tmp2.append(item)
        feature.append(tmp2)
    return feature


def process_data(data, predict):
    group = []
    for i in range(len(data)):
        group.append(data[i][0])
    data = np.delete(data, 0, 1)
    data_feature = []
    data_predict = []
    for j in range(0, len(data) - 30 - 1, 1):
        flag = 1
        tmp3 = group[j]
        for m in range(10):
            if (group[j + m] != tmp3):
                flag = 0
        if flag:
            tmp = data[j:j+10]
            mean = np.mean(tmp, axis=0)
            std = np.std(tmp, axis=0)
            tmp = (tmp - mean) / (std + 0.000000000000001)
            tmp2 = []
            for k in range(10):
                for item in tmp[k]:
                    tmp2.append(item)
            data_feature.append(tmp2)
            pre = predict[j+10:j+30, 6]
            data_predict.append((np.mean(pre)-predict[j+9, 6]))
    data_predict = np.array(data_predict)
    feature_val = data_feature[len(data_feature)-2000:]
    predict_val = data_predict[len(data_predict) - 2000:]
    data_feature = data_feature[:len(data_feature)-2000]
    data_predict = data_predict[:len(data_predict)-2000]
    return data_feature, data_predict, feature_val, predict_val

data, predict = load_csv_data('train_data_processed.csv')
feature, predict1, feature_val, predict_val = process_data(data, predict)
x_test = load_csv_test_data('test_data.csv')
x_test = process_data_test(x_test)

# train model
print("\nTraining XGBoost ...\n")

params = {'max_depth': 5, 'learning_rate': 0.1, 'n_estimators': 500, 'silent': 1, 'min_child_weight': 1,
          'subsample': 0.8, 'colsample_bytree': 0.8, 'gamma': 0, 'reg_alpha': 0, 'reg_lambda': 1, 'nthread': 4,
          'loss': 'rmse'}

best_params = {'max_depth': 5, 'learning_rate': 0.1, 'n_estimators': 500, 'silent': 1, 'min_child_weight': 1,
          'subsample': 0.8, 'colsample_bytree': 0.8, 'gamma': 0, 'reg_alpha': 0, 'reg_lambda': 1, 'nthread': 4,
               'loss': 'rmse'}
current_loss = 100000
epoch = 0
plt.figure()
ax = []
ay = []
for estimator in [100]:
    for min_child_weight in [1]:
        for max_depth in [3]:
            for gamma in [0.0]:
                for subsample in [0.8]:
                    for colsample in [0.8]:
                        for reg_alpha in [0]:
                            for reg_lambda in [1.0]:
                                for lr in [0.1]:
                                    print("epoch: ", epoch)
                                    params['max_depth'] = max_depth
                                    params['learning_rate'] = lr
                                    params['n_estimators'] = estimator
                                    params['min_child_weight'] = min_child_weight
                                    params['subsample'] = subsample
                                    params['colsample_bytree'] = colsample
                                    params['gamma'] = gamma
                                    params['reg_lambda'] = reg_lambda
                                    params['reg_alpha'] = reg_alpha
                                    print("try: ", params)
                                    model = xgb.XGBRegressor(**params)
                                    model.fit(feature, predict1)
                                    pred_val = model.predict(feature_val)
                                    error = 0
                                    for i in range(len(pred_val)):
                                        error += float(pred_val[i] - predict_val[i])*(pred_val[i] - predict_val[i])
                                    loss = float(error)/len(pred_val)
                                    ax.append(reg_lambda)
                                    ay.append(loss)
                                    if loss < current_loss:
                                        current_loss = loss
                                        best_params = params
                                        print("updated!")
                                        y_pred = model.predict(x_test)
                                        x_test2 = load_csv_test_data2('test_data.csv')
                                        with open('sample.csv', 'w', newline='') as fout:
                                            fieldnames = ['caseid', 'midprice']
                                            writer = csv.DictWriter(fout, fieldnames=fieldnames)
                                            writer.writeheader()
                                            for i in range(142, len(y_pred)):
                                                tmp = float(y_pred[i])
                                                writer.writerow({'caseid': str(i + 1),
                                                                 'midprice': float(tmp + x_test2[(i + 1) * 10 - 1][6])})
                                    else:
                                        print("Not updated!")
                                    print("current loss: ", current_loss)
                                    print("current best params: ", best_params)
                                    epoch = epoch + 1
plt.plot(ax, ay, 'b-')
plt.xlabel('reg_lambda')
plt.ylabel('Loss')
plt.show()
