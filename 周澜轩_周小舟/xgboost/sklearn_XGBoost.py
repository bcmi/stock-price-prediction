from utils import create_datasets, rMSE_Loss
import numpy as np
import csv

import xgboost as xgb

train = "data/train_data.csv"
test = "data/test_data.csv"

x, y, test, base = create_datasets(train, test)
x = x.reshape(x.shape[0], 50)

valid_x = x[:1000]
valid_y = y[:1000]
train_x = x[1000:]
train_y = y[1000:]

'''
min_loss = 1.0 # 0.0012749498997552062

print("training.")
parameters = {
    "max_depth": [3, 4, 5, 6],
    "learning_rate": [0.1, 0.03, 0.01],
    "n_estimators": [100, 120, 160, 200],
    "gamma": [0.0, 0.1, 0.03, 0.01],
    "min_child_weight": [1, 2, 0, 3],
    "max_delta_step": [0, 1, 2, 3],
    "subsample": [1.0, 0.95, 0.9],
    "reg_alpha": [0.0, 0.01, 0.03, 0.1],
    "reg_lambda": [1.0, 0.95, 0.9],
}

for a in parameters["max_depth"]:
    for b in parameters["learning_rate"]:
        for c in parameters["n_estimators"]:
            for d in parameters["gamma"]:
                for e in parameters["min_child_weight"]:
                    for f in parameters["max_delta_step"]:
                        for g in parameters["subsample"]:
                            for h in parameters["reg_alpha"]:
                                for i in parameters["reg_lambda"]:
                                    test_model = xgb.XGBRegressor(
                                        max_depth=a,
                                        learning_rate=b,
                                        n_estimators=c,
                                        gamma=d,
                                        min_child_weight=e,
                                        max_delta_step=f,
                                        subsample=g,
                                        reg_alpha=h,
                                        reg_lambda=i
                                    )
                                    test_model.fit(train_x, train_y)

                                    print("validating.")
                                    check_y = test_model.predict(valid_x)
                                    tmp_loss = rMSE_Loss(valid_y, check_y)
                                    if tmp_loss < min_loss:
                                        min_loss = tmp_loss
                                        print("************************************************")
                                        print(tmp_loss, "\n\n")
                                        print("max_depth = ", a)
                                        print("learning_rate = ", b)
                                        print("n_estimators = ", c)
                                        print("gamma = ", d)
                                        print("min_child_weight = ", e)
                                        print("max_delta_step = ", f)
                                        print("subsample = ", g)
                                        print("reg_alpha = ", h)
                                        print("reg_lambda = ", i)
                                        print("************************************************\n\n")
                                    else:
                                        print("not good.")



'''
test_model = xgb.XGBRegressor(
    max_depth=3,
    learning_rate=0.1,
    n_estimators=100,
    gamma=0.0,
    min_child_weight=3,
    max_delta_step=0,
    subsample=0.9,
    reg_alpha=0.03,
    reg_lambda=0.9
)
test_model.fit(train_x, train_y)

print("validating.")
check_y = test_model.predict(valid_x)
print(rMSE_Loss(valid_y, check_y))

print("predicting.")
test = test.reshape(test.shape[0], 50)
predict = base + test_model.predict(test)

with open("sample1.csv", 'w') as fout:
    fieldnames = ['caseid', 'midprice']
    writer = csv.DictWriter(fout, fieldnames=fieldnames)
    writer.writeheader()
    for i in range(len(predict)):
        if i < 142:
            continue
        writer.writerow({'caseid': str(i + 1), 'midprice': float(predict[i])})
