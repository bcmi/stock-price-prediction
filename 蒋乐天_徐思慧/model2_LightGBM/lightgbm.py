import pandas as pd
import lightgbm as lgbm
import matplotlib.pyplot as plt
import pickle
import numpy as np

with open("X_train.pkl", "rb") as f:
	X_train = pickle.load(f)

with open("y_train.pkl", "rb") as f:
	y_train = pickle.load(f)

with open("X_test.pkl", "rb") as f:
	X_test = pickle.load(f)

with open("m_test.pkl", "rb") as f:
	m_test = pickle.load(f)

X_train = np.array(X_train)
y_train = np.array(y_train)
X_test = np.array(X_test)

#LightGBM model
split = int(len(X_train)*0.8)
lgbm_train_set = lgbm.Dataset(X_train[:split], y_train[:split])
lgbm_valid_set = lgbm.Dataset(X_train[split:], y_train[split:])
lgbm_params = { 'boosting_type': 'gbdt', 'objective': 'regression','n_estimators': 20000, 'metric': 'mse','num_leaves': 30,'learning_rate': 0.002,'early_stopping_rounds': 200}
model = lgbm.train(lgbm_params,lgbm_train_set, 2,verbose_eval=100,valid_sets=[lgbm_train_set,lgbm_valid_set])
predict = model.predict(X_test,num_iteration=model.best_iteration)

#plot the decision tree
plt.figure(figsize=(100, 50))
lgbm.plot_tree(model, tree_index=1)
plt.savefig("lgbm_tree_demonstration.png")

#save the prediction
caseid = [i for i in range(143, 1001)]
midprice = np.array(predict) + np.array(m_test)
submit = pd.DataFrame({'caseid': caseid, 'midprice': midprice})
submit.to_csv('lgbm_final.csv', index=False)
