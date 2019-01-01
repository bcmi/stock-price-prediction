import numpy as np
import pandas as pd
import lightgbm as lgb
import csv

def test():
    # Load file
    file = pd.read_csv('test_data.csv')
    data = file[['MidPrice', 'LastPrice', 'BidPrice1', 'BidVolume1', 'AskPrice1', 'AskVolume1']]

    # Add the new feature
    imbalance = data['BidVolume1'] - data['AskVolume1']
    new_col = pd.DataFrame(imbalance, columns=["Imbalance"])
    res = pd.concat([data, new_col], axis=1)
    volumes = res[['BidVolume1', 'AskVolume1', "Imbalance"]]
    volumes = np.array(volumes)
    new_vol = pd.DataFrame(volumes, columns=['BidVolume1', 'AskVolume1', "Imbalance"])
    res[['BidVolume1', 'AskVolume1', "Imbalance"]] = new_vol

    data = np.array(res)

    L = len(data) // 10
    dat = list()
    pvs = list()
    for i in range(L):
        pvs.append(data[i * 10 + 9][0])
    for i in range(L):
        pivot = data[i * 10 + 9][0]
        pivot_array = np.array([pivot, pivot, pivot, 0, pivot, 0, 0])
        feature_for_case = data[i * 10: i * 10 + 10] - pivot_array
        dat.append(feature_for_case.reshape(70))

    # Load model
    filepath = 'model.h5'
    gbm = lgb.Booster(model_file=filepath)

    # Predict
    predict = gbm.predict(np.array(dat), num_iteration=gbm.best_iteration)

    result = list()
    displacement = 0.0005

    for i in range(len(predict)):
        if (predict[i] < 0.5):
            result.append(pvs[i] - displacement)
        elif (predict[i] >= 0.5):
            result.append(pvs[i] + displacement)

    # Write result
    with open("result.csv", "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["caseid", "midprice"])
        for i in range(142, len(result)):
            writer.writerow([i + 1, float(result[i])])

if __name__ == '__main__':
    test()


