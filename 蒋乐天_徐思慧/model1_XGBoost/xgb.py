import pandas as pd
import numpy as np
import random
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
from math import sqrt

import xgboost as xgb
from xgboost import XGBRegressor

with open("X_train.pkl", "rb") as f:
	X_train = pickle.load(f)

with open("y_train.pkl", "rb") as f:
	y_train = pickle.load(f)

with open("X_test.pkl", "rb") as f:
	X_test = pickle.load(f)

with open("m_test.pkl", "rb") as f:
	m_test = pickle.load(f)

X_train_filter = []
y_train_filter = []
bound = 0.003
for i in range(len(X_train)):
	if abs(y_train[i]) < bound:
		X_train_filter.append(X_train[i])
		y_train_filter.append(y_train[i])

#xgboost model
model = xgb.XGBRegressor(max_depth=7, learning_rate=0.1, n_estimators=1000, silent=False, objective='reg:linear')
model.fit(X_train_filter, y_train_filter)
model.save_model("xgb.model")

y_predict = model.predict(X_test)
y_predict = np.array(y_predict)

#save the prediction
caseid = [i for i in range(143, 1001)]
midprice = np.array(y_predict) + np.array(m_test)

submit = pd.DataFrame({'caseid': caseid, 'midprice': midprice})
submit.to_csv('../submit/correct_v1.csv', index=False)

