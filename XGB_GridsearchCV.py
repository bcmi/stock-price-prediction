import pandas as pd
import numpy as np
from sklearn.metrics import make_scorer
from xgboost.sklearn import XGBRegressor
from sklearn.model_selection import train_test_split
from xgboost import plot_importance
import csv
import random
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt

def rmse(y, y_pred):
        return np.sqrt(np.mean(np.square(y-y_pred)))
rmse_scorer = make_scorer(rmse, greater_is_better=False)


# use GridSearchCV search for best parameters
'''
xgb1 = XGBRegressor()

parameters = {'learning_rate': [0.05], 'n_estimators': [400], 'objective': ['reg:linear'],
              'max_depth': range(1, 10, 2),
              'min_child_weight': range(1, 10, 2),
              'gamma': [i/10.0 for i in range(0,5)],
              'subsample': [i/10.0 for i in range(6,10)],
              'colsample_bytree': [i/10.0 for i in range(6,10)],
              'scale_pos_weight': [0, 1],
              'eval_metric': ['rmse']}
xgb_grid = GridSearchCV(xgb1,
                        parameters,
                        cv=2,
                        n_jobs=4,
                        verbose=5,
                        refit=1,
                        scoring=rmse_scorer)
xgb_grid.fit(X, Y)
print(xgb_grid.best_score_)
print(xgb_grid.best_params_)
xgb = xgb_grid.best_estimator_
'''

print("START")
X = np.load('X_train.npy')
Y = np.load('Y_train.npy')
X_test = np.load('X_test.npy')

# add randomness
for i in range(Y.shape[0]):
        Y[i] += (random.random()-0.5)/200

parameters = {'colsample_bytree': 0.8,
              'learning_rate': 0.1,
              'max_depth': 6,
              'min_child_weight': 1,
              'n_estimators': 1000,
              'objective': 'reg:linear',
              'silent': 1}
xgb = XGBRegressor(**parameters)
X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.01, random_state=42)  # use 1% train data for validation
xgb.fit(X_train, Y_train, eval_set=[(X_val, Y_val)], eval_metric='rmse', early_stopping_rounds=50)

# plot feature importance
'''
fig, ax = plt.subplots(figsize=(15,15))
plot_importance(xgb, height=0.9, ax=ax)
plt.show()
'''

# make prediction
pred = xgb.predict(X_test)

# write csv
test_data = np.array(pd.read_csv('test_data.csv').drop(['Date', 'Time'], axis=1).iloc[:, 1:])
with open('Prediction.csv', 'w') as fout:
        fieldnames=['caseid', 'midprice']
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()
        for i in range(142, 1000):
                writer.writerow({'caseid': str(i+1), 'midprice': str(pred[i]+test_data[i*10+9, 0])})
