import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_validate
import csv
import matplotlib.pyplot as plt


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


def k-fold(reg, features_train, offset_train):
    # test 5-fold, 10-fold, 15-fold
    for i, color in zip([5, 10, 15], ['r', 'g', 'b']):
        scores = cross_validate(reg, bid_ask, mid, cv=i, scoring='neg_mean_squared_error')
        plt.plot(np.arange(1, i+1), scores['test_score'], color)
        plt.scatter(np.arange(1, i+1), scores['test_score'], facecolor=color)
        print(scores['test_score'])
        print(np.mean(scores['test_score']))

    plt.legend(('5-fold', '10-fold', '15-fold'))
    plt.xlabel('iteration')
    plt.ylabel('neg MSE')
    plt.savefig('k_fold.png')
    plt.show()


def grid_search(reg, features_train, offset_train):
    tuned_parameters = {'n_estimators': range(60, 74, 5), 'max_features': range(10, 40, 5)}

    grid = GridSearchCV(reg, param_grid=tuned_parameters, cv=3,
                       scoring='neg_mean_squared_error')
    grid.fit(features_train, offset_train)

    print("Best parameters set:")
    print(grid.best_params_)
    print()
    print("scores:")
    means = grid.cv_results_['mean_test_score']
    stds = grid.cv_results_['std_test_score']
    times = grid.cv_results_['mean_fit_time']
    for mean, std, time, params in zip(means, stds, times, grid.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r using %0.3f s"
              % (mean, std * 2, params, time))
    print()


def gradient_tree_boosting(x_train, y_train, x_test, y_test, features_test_flat):
    reg = GradientBoostingRegressor(learning_rate=0.1, n_estimators=250, max_depth=5, loss='ls')
    reg.fit(x_train, y_train)
    y_predict = reg.predict(x_test)
    mse = mean_squared_error(y_test, y_predict)
    print(mse)

    offset_predict = est.predict(features_test_flat)
    return offset_predict



if __name__ == '__main__':
    train_file = pd.read_csv('train_data.csv')
    test_file = pd.read_csv('test_data.csv')

    raw_train_data = train_file[['MidPrice', 'LastPrice', 'Volume', 'BidPrice1', 'BidVolume1', 'AskPrice1', 'AskVolume1']].values
    raw_test_data = test_file[['MidPrice', 'LastPrice', 'Volume', 'BidPrice1', 'BidVolume1', 'AskPrice1', 'AskVolume1']].values

    # scaler = StandardScaler()
    # # scaler = MinMaxScaler()

    # raw_train_data_scaled = scaler.fit_transform(raw_train_data)
    # raw_test_data_scaled = scaler.transform(raw_test_data)

    raw_train_data_time_and_date = train_file[['Date', 'Time']].values

    features_train, offset_train, base_train = get_train_data(raw_train_data_time_and_date, raw_train_data_scaled, raw_train_data)
    features_test, base_test = get_test_data(raw_test_data_scaled, raw_test_data)

    features_train_flat = list(map(lambda i: list(features_train[i].flatten()), range(len(features_train))))
    features_test_flat = list(map(lambda i: list(features_test[i].flatten()), range(len(features_test))))

    x_train, x_test = features_train_flat[4000:], features_train_flat[:4000]
    y_train, y_test = offset_train[4000:], offset_train[:4000]

    offset_predict = gradient_tree_boosting(x_train, y_train, x_test, y_test, features_test_flat)
    final_predict = list(np.array(base_test) + np.array(offset_predict))
    write_file('gt_boosting_result.csv', final_predict)
