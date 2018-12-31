import pickle
import numpy as np
from sklearn.metrics import roc_auc_score  
from sklearn.model_selection import train_test_split  
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
            X.append(x)
            if np.mean(p[i+9:i+10, 0]) > np.mean(p[i+10:i+30, 0]):
                y = 0
            else:
                y = 1
            Y.append(y)
    return np.array(X), np.array(Y)      

# f = open('data/cut_data', 'rb')
# cut_data = pickle.load(f)
# train_x, train_y = get_dataset(cut_data)
# print(train_x.shape, train_y.shape)

# X, val_X, y, val_y = train_test_split(  
#     train_x,  
#     train_y,  
#     test_size=0.1,  
#     random_state=1,  
#     stratify=train_y  
# )
# print(X.shape, y.shape)
# print(val_X.shape, val_y.shape)

# # create dataset for lightgbm  
# lgb_train = lgb.Dataset(X, y)  
# lgb_eval = lgb.Dataset(val_X, val_y, reference=lgb_train)

# # specify your configurations as a dict  
# params = {  
#     'boosting_type': 'gbdt',  
#     'objective': 'binary',  
#     'metric': {'binary_logloss', 'auc'},  
#     'num_leaves': 5,  
#     'max_depth': 6,  
#     'min_data_in_leaf': 450,  
#     'learning_rate': 0.1,  
#     'feature_fraction': 0.9,  
#     'bagging_fraction': 0.95,  
#     'bagging_freq': 5,  
#     'lambda_l1': 1,    
#     'lambda_l2': 0.001,  # 越小l2正则程度越高  
#     'min_gain_to_split': 0.2,  
#     'verbose': 5,  
#     'is_unbalance': True  
# }  

# # train  
# print('Start training...')  
# gbm = lgb.train(params,  
#                 lgb_train,  
#                 num_boost_round=8000,  
#                 valid_sets=lgb_eval,  
#                 early_stopping_rounds=500) 

# importance = gbm.feature_importance()  
# names = gbm.feature_name()  

# print(importance)
# print(names)

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
print("test input shape:", test_X_raw.shape)

# print("start predicting...")
# preds = gbm.predict(test_X_raw, num_iteration=gbm.best_iteration) 

# threshold = 0.5
# predictions = preds.copy()
# for i in range(len(preds)):  
#     predictions[i] = 1 if preds[i] > threshold else 0  

res = open('data/lgbm_class_res.pkl', 'rb')
# pickle.dump(predictions, res)
preds = pickle.load(res)
print('test output shape', preds.shape)
pred_price = preds.copy()
for i in range(len(preds)):
    pred_price[i] = test_X_raw[i][63]
    if preds[i] == 0:
        pred_price[i] -= 5e-4
    else:
        pred_price[i] += 5e-4

# fluctation 3.8e-4

caseid = list(range(143,1001))
df = pd.DataFrame({'caseid':caseid, 'midprice':pred_price})
df.to_csv('data/lgbm_result_v2.csv',index = False, sep = ',')
