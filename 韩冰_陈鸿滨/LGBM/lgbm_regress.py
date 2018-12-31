import pickle
import numpy as np

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV

import lightgbm as lgb 
import pandas as pd


def loaddata(filename = 'data/train_data.csv'):
	df = pd.read_csv(filename)
	train_data = df.iloc[:,3:10]
	return train_data

def get_dataset(cut_data):
    X = []
    Y = []
    for piece in cut_data:
        p = np.array(piece)
        p = np.hstack((p[:,0:2], p[:, 3:]))
        for i in range(0, len(p) - 29):
            x = p[i:i+10, :].reshape(-1)  
            x = x[59:70]
            # x = x[0:7]
            X.append(x)
            Y.append(p[i+9, 0] - np.mean(p[i+10:i+30, 0]))
    return np.array(X), np.array(Y)      

def splitDataset(X, Y, rate = 0.1):
    l = len(X)
    return X[int(rate*l):], X[:int(rate*l)], Y[int(rate*l):], Y[:int(rate*l)]


f = open('data/cut_data', 'rb')
cut_data = pickle.load(f)
train_x, train_y = get_dataset(cut_data)
print(train_x.shape, train_y.shape)

randlist = np.arange(len(train_x))
np.random.shuffle(randlist)
X = train_x[randlist]
Y = train_y[randlist]
X_train, X_test, y_train, y_test = splitDataset(X, Y)

print('Start training...')
# train
gbm = lgb.LGBMRegressor(objective='regression',
                        num_leaves=25,
                        learning_rate=0.1,
                        n_estimators=500)
gbm.fit(X_train, y_train,
        eval_set=[(X_test, y_test)],
        eval_metric='l1',
        early_stopping_rounds=5)
 
print('Start predicting...')
# predict
y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration_)
# eval
print('The rmse of prediction is:', mean_squared_error(y_test, y_pred) ** 0.5)
 
# feature importances
print('Feature importances:', list(gbm.feature_importances_))
 
# other scikit-learn modules
# estimator = lgb.LGBMRegressor(num_leaves=31)
 
# param_grid = {
#     'learning_rate': [0.01, 0.1, 1],
#     'n_estimators': [40, 70]
# }
 
# gbm = GridSearchCV(estimator, param_grid)
 
# gbm.fit(X_train, y_train)
 
# print('Best parameters found by grid search are:', gbm.best_params_)


print('load test data...')
test_X_pd = loaddata('data/test_data.csv').iloc[1420:,:]
_test_X_pd = test_X_pd.values
test_diff_volume = []
for i in range(len(_test_X_pd)-1):
    test_diff_volume.append(_test_X_pd[i+1,2] - _test_X_pd[i,2])
test_diff_volume.append(test_diff_volume[-1])
for i in range(len(test_diff_volume)):
    if(test_diff_volume[i]<0):
        test_diff_volume[i] = 0
test_X_pd['diff_volume'] = np.array(test_diff_volume)
test_X_raw = test_X_pd.values
test_X_raw = np.hstack((test_X_raw[:, 0:2], test_X_raw[:, 3:]))
test_X_raw = np.reshape(test_X_raw, (-1, 70))
test_X = np.zeros((len(test_X_raw), 11))
for i in range(len(test_X_raw)):
    test_X[i] = test_X_raw[i][59:70]
print("test input shape:", test_X.shape)

print('start predicting...')
preds = gbm.predict(test_X, num_iteration=gbm.best_iteration_)

for i in range(len(preds)):
    preds[i] = preds[i] + test_X[i][-7]

caseid = list(range(143,1001))
df = pd.DataFrame({'caseid':caseid, 'midprice':preds})
df.to_csv('data/lgbm_result_v6.csv',index = False, sep = ',')
