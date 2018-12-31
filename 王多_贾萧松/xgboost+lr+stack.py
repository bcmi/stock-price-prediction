import pandas as pd
from sklearn import preprocessing
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn import metrics
import time
import os,sys
import csv
from sklearn.preprocessing import PolynomialFeatures
import operator
from xgboost import plot_importance
from sklearn.feature_selection import SelectFromModel
import warnings
from numpy import sort
from sklearn.model_selection import GridSearchCV   #Perforing grid search
from sklearn.preprocessing import  OneHotEncoder
from sklearn.linear_model import LinearRegression
from scipy.sparse import hstack
from sklearn.model_selection import KFold
import random

#########log!!!!!!
class Logger():
    def __init__(self, logname):
        self.terminal = sys.stdout
        self.log = open(logname, 'a')
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
    def flush(self):
        pass

now = time.strftime("%Y-%m-%d-%H_%M_%S",time.localtime(time.time()))
dir = str(now)
os.mkdir(dir)
sys.stdout = Logger("%s/%s.log"%(dir,dir))

#####read data
train_df = pd.read_csv("train_data_3.csv", index_col=0)
X = train_df.loc[:,train_df.columns!="target"]
Y = train_df.loc[:,train_df.columns=="target"]
print(np.isnan(Y).any().any())
val_df = pd.read_csv("test_data_3.csv", index_col=0)
print("Read CSV Complete")
trainX, testX, trainY, testY = train_test_split(X, Y, test_size=0.2)


############PRETRAIN
xgb_params = {
            'max_depth': 7,
            "min_child_weight": 7,
            "gamma": 0,
            'colsample_bytree': 0.6,
            "subsample": 0.4,
            "reg_alpha": 0.1,
            "reg_lambda": 0.01,
            "learning_rate": 0.01,
            "n_estimators": 6500,
} 
print(xgb_params)

xlf = xgb.XGBRegressor(**xgb_params)
start = time.time()
xlf.fit(trainX, trainY, eval_metric="rmse", verbose=False, eval_set=[(testX, testY)], early_stopping_rounds=50)
end = time.time()
print("Time Comsuming: "+str(float(end-start)))

##############get feature importance
train_feature = list(train_df.columns)
train_feature.remove("target")
imp = list(zip(train_feature, xlf.feature_importances_))
imp.sort(key=lambda k:k[1], reverse=True)

thresholds = []
name = []
for item in imp:
    print(item)
    name.append(item[0])
    thresholds.append(item[1])
feature_rank = name
print(name)
print(thresholds)

###################feature selection
# for i in range(50, len(feature_rank)):
#     start = time.time()
#     select_trainX = trainX[feature_rank[:i]]
#     selection_model = xgb.XGBRegressor(**xgb_params)
#     #print(select_trainX)
#     #print(trainY)
#     select_testX = testX[feature_rank[:i]]
#     selection_model.fit(select_trainX, trainY, eval_metric="rmse", verbose=True, eval_set=[(select_testX, testY)], early_stopping_rounds=20)
#     preds = selection_model.predict(select_testX)
#     loss = np.sqrt(metrics.mean_squared_error(testY, preds))
#     end = time.time()
#     print("n=%d, RMSE: %f"%(i, loss) + " Time Comsuming: "+str(float(end-start)))
feature_rank = ['Volume(t-20)', 'MidPrice(t-20)', 'LastPrice(t-20)', 'BidVolume1(t-20)', 'AskVolume1(t-20)', 'RelaPriceDif(t-20)', 'TotalDif(t-20)', 'BidVolume1Dif(t-21)', 'BidPrice1(t-20)', 'VolumeDif(t-21)', 'BidVolume1Dif(t-22)', 'AskVolume1(t-21)', 'Volume(t-21)', 'AskVolume1Dif(t-21)', 'BidTotal(t-20)', 'AskVolume1Dif(t-22)', 'RelaPriceDif(t-21)', 'LastPrice(t-21)', 'BidVolume1(t-21)', 'MidPrice(t-21)', 'VolumeDif(t-22)', 'BidVolume1Dif(t-23)', 'K(t-20)', 'Depth(t-20)', 'RelaPriceDif(t-27)', 'AskVolume1Dif(t-23)', 'AskTotal(t-20)', 'RelaPriceDif(t-25)', 'AskPrice1(t-20)', 'AskVolume1(t-22)', 'BidVolume1Dif(t-24)', 'AskVolume1(t-29)', 'LastPrice(t-22)', 'AskVolume1(t-23)', 'RelaPriceDif(t-29)', 'BidVolume1(t-22)', 'AskVolume1Dif(t-24)', 'Depth(t-21)', 'MidPrice(t-22)', 'Depth(t-29)', 'RelaPriceDif(t-22)', 'BidPrice1(t-21)', 'RelaPriceDif(t-28)', 'LastPrice(t-29)', 'AskVolume1Dif(t-26)', 'Depth(t-23)', 'RelaPriceDif(t-24)', 'BidVolume1Dif(t-27)', 'TotalDif(t-21)', 'VolumeDif(t-29)', 'VolumeDif(t-23)', 'AskVolume1(t-26)', 'BidVolume1(t-29)', 'BidVolume1(t-28)', 'MidPrice(t-29)', 'Depth(t-22)', 'BidVolume1(t-27)', 'BidVolume1(t-23)', 'VolumeDif(t-28)', 'BidVolume1Dif(t-29)', 'VolumeDif(t-24)', 'LastPrice(t-23)', 'RelaPriceDif(t-23)', 'BidVolume1(t-24)', 'AskVolume1Dif(t-25)', 'RelMid(t-28)', 'BidVolume1Dif(t-25)', 'Depth(t-27)', 'Depth(t-24)', 'RelMid(t-27)', 'AskVolume1(t-24)', 'BidVolume1Dif(t-28)', 'AskVolume1Dif(t-29)', 'AskVolume1(t-27)', 'RelMid(t-23)', 'RelaPriceDif(t-26)', 'AskVolume1Dif(t-27)', 'AskVolume1Dif(t-28)', 'MidPrice(t-23)', 'BidTotal(t-21)', 'K(t-21)', 'MidPrice(t-25)', 'VolumeDif(t-25)', 'Depth(t-26)', 'VolumeDif(t-26)', 'LastPrice(t-27)', 'K(t-29)', 'RelMid(t-22)', 'RelMid(t-26)', 'LastPrice(t-26)', 'BidVolume1(t-25)', 'BidVolume1(t-26)', 'VolumeDif(t-27)', 'LastPrice(t-24)', 'RelMid(t-29)', 'LastPrice(t-28)', 'RelMid(t-25)', 'Depth(t-25)', 'BidVolume1Dif(t-26)', 'BidPrice1(t-22)', 'K(t-24)', 'RelMid(t-21)', 'Depth(t-28)', 'K(t-22)', 'AskVolume1(t-28)', 'TotalDif(t-25)', 'LastPrice(t-25)', 'K(t-23)', 'MidPrice(t-24)', 'VolDif(t-20)', 'TotalDif(t-23)', 'AskPrice1(t-21)', 'TotalDif(t-29)', 'BidTotal(t-22)', 'K(t-27)', 'AskVolume1(t-25)', 'Volume(t-29)', 'TotalDif(t-22)', 'MidPrice(t-26)', 'TotalDif(t-27)', 'K(t-28)', 'BidPrice1(t-29)', 'TotalDif(t-26)', 'K(t-26)', 'MidPrice(t-28)', 'BidPrice1(t-24)', 'TotalDif(t-24)', 'K(t-25)', 'BidPrice1(t-23)', 'RelMid(t-24)', 'TotalDif(t-28)', 'AskPrice1(t-23)', 'AskTotal(t-22)', 'MidPrice(t-27)', 'AskPrice1(t-24)', 'Volume(t-22)', 'Volume(t-23)', 'AskTotal(t-21)', 'AskPrice1(t-29)', 'AskPrice1(t-26)', 'AskPrice1(t-27)', 'BidPrice1(t-26)', 'AskPrice1(t-25)', 'AskPrice1(t-22)', 'AskTotal(t-23)', 'AskTotal(t-28)', 'BidTotal(t-23)', 'BidTotal(t-27)', 'BidTotal(t-26)', 'BidTotal(t-29)', 'BidPrice1(t-25)', 'AskPrice1(t-28)', 'BidTotal(t-28)', 'BidPrice1(t-28)', 'BidTotal(t-25)', 'Volume(t-24)', 'AskTotal(t-29)', 'AskTotal(t-25)', 'AskTotal(t-27)', 'Volume(t-28)', 'BidPrice1(t-27)', 'BidTotal(t-24)', 'AskTotal(t-26)', 'AskTotal(t-24)', 'Volume(t-25)', 'RelBid(t-20)', 'VolDif(t-26)', 'Volume(t-27)', 'VolDif(t-28)', 'VolDif(t-29)', 'VolDif(t-21)', 'VolDif(t-23)', 'Volume(t-26)', 'VolDif(t-24)', 'VolDif(t-22)', 'VolDif(t-25)', 'VolDif(t-27)', 'RelBid(t-23)', 'BidPrice1Dif(t-21)', 'LastPriceDif(t-28)', 'day_hour1_9.0', 'LastPriceDif(t-21)', 'RelBid(t-29)', 'BidPrice1Dif(t-22)', 'RelBid(t-21)', 'RelBid(t-24)', 'AskPrice1Dif(t-22)', 'AskPrice1Dif(t-21)', 'LastPriceDif(t-29)', 'LastPriceDif(t-23)', 'weekday1_5.0', 'LastPriceDif(t-22)', 'RelBid(t-22)', 'LastPriceDif(t-27)', 'weekday1_4.0', 'RelBid(t-26)', 'BidPrice1Dif(t-23)', 'AskPrice1Dif(t-29)', 'weekday1_3.0', 'weekday1_2.0', 'BidPrice1Dif(t-29)', 'AskPrice1Dif(t-23)', 'weekday1_1.0', 'RelBid(t-28)', 'BidPrice1Dif(t-28)', 'AskPrice1Dif(t-28)', 'day_hour1_11.0', 'AskPrice1Dif(t-24)', 'LastPriceDif(t-25)', 'LastPriceDif(t-24)', 'BidPrice1Dif(t-24)', 'LastPriceDif(t-26)', 'BidPrice1Dif(t-26)', 'BidPrice1Dif(t-25)', 'day_hour1_10.0', 'AskPrice1Dif(t-25)', 'BidPrice1Dif(t-27)', 'RelAsk(t-22)', 'RelBid(t-25)', 'day_hour1_13.0', 'AskPrice1Dif(t-27)', 'RelBid(t-27)', 'AskPrice1Dif(t-26)', 'day_hour1_14.0', 'RelAsk(t-26)', 'RelAsk(t-20)', 'RelAsk(t-27)', 'RelAsk(t-28)', 'RelAsk(t-25)', 'RelAsk(t-21)', 'BidAskDif(t-20)', 'BidAskDif(t-22)', 'BidAskDif(t-29)', 'BidAskDif(t-25)', 'BidAskDif(t-24)', 'BidAskDif(t-26)', 'BidAskDif(t-21)', 'BidAskDif(t-27)', 'RelAsk(t-29)', 'BidAskDif(t-23)', 'BidAskDif(t-28)', 'RelAsk(t-23)', 'RelAsk(t-24)']

upper_bound_k = 220
lowwer_bound_k = 120

################# Grid Search Best Parameters
###############parameters
# start = time.time()
# select_trainX = X[feature_rank[:k]]
# #selection_model = xgb.XGBRegressor(**other_params)
# #print(select_trainX)
# #print(trainY)
# param_test = []
# param_test = []

# for i in [1e-3, 3.5e-3, 7e-3,1e-2]:
#     param_test.append({})
#     param_test[-1]['learning_rate']=[i]
#     param_test[-1]["n_estimators"]=[int(65/i)]
# print(param_test)
# gsearch = GridSearchCV(estimator=xgb.XGBRegressor(**other_params), param_grid=param_test, cv=5, verbose=1, scoring='neg_mean_squared_error')
# gsearch.fit(select_trainX, Y)
# end = time.time()

# print("grid:" + str(gsearch.cv_results_))
# print("best parameters:"+ str(gsearch.best_params_))
# print("best score:"+str(gsearch.best_score_))
# print("Time Comsuming:"+str(float(end-start)))



###number of model to stack
n = 36
splits = 5


folds = list(KFold(shuffle=True, n_splits=splits).split(X, y=Y))

S_train = np.zeros((train_df.shape[0], n))
S_val = np.zeros((val_df.shape[0], n))


for i in range(n):
    S_val_i = np.zeros((val_df.shape[0], splits))
    ####random features
    feature_num = random.randint(lowwer_bound_k, upper_bound_k)
    for j, (train_idx, test_idx) in enumerate(folds):
        X_train = X.iloc[train_idx][feature_rank[:feature_num]]
        Y_train = Y.iloc[train_idx]
        X_test = X.iloc[test_idx][feature_rank[:feature_num]]
        Y_test = Y.iloc[test_idx]
        X_val = val_df[feature_rank[:feature_num]]
        
        ###random params a little
        params = {}
        for key, value in xgb_params.items():
            if(key not in ["n_estimators", "max_depth"]):
                params[key] = value*random.normalvariate(1,1e-1)
                #xgb_params[key] = params[key]
            else:
                params[key] = int(value*random.normalvariate(1,1e-1))+1
                #xgb_params[key] = params[key]
        model = xgb.XGBRegressor(**params)

        #xgb fit
        print("Fit xgb model %d fold %d" % (i, j))
        model.fit(X_train, Y_train)
        #xgb + lr 
        lr = LinearRegression()
        x_train_leaves = model.apply(X_train)
        x_test_leaves = model.apply(X_test)
        x_val_leaves = model.apply(X_val)

        train_test_rows = X.shape[0]

        x_leaves = np.concatenate((x_train_leaves, x_test_leaves, x_val_leaves),axis=0)
        xgb_encoder = OneHotEncoder()
        x_leaves = xgb_encoder.fit_transform(x_leaves)
        #x_leaves = x_leaves.toarray()
        print("Leaves shape:"+str(x_leaves.shape))
        X_train = hstack([x_leaves[:len(train_idx),:],X_train])
        X_test = hstack([x_leaves[len(train_idx):train_test_rows,:],X_test])
        X_val = hstack([x_leaves[train_test_rows:,:],X_val])
        print(np.isnan(Y_train))
        lr.fit(X_train, Y_train)
        y_pred = lr.predict(X_test)
        S_train[test_idx, i] = y_pred
        S_val_i[:, j] = lr.predict(X_val)
    S_val[:,i] = S_val_i.mean(axis=1)

##### Output the stack result
lr = LinearRegression()
lr.fit(S_train, Y)
res = lr.predict(S_val).T.tolist()[0]


with open("%s/xgb+lr+stack_res_%s.csv"%(dir,str(now)), 'w', newline='') as fout:
        fieldnames = ['caseid', 'midprice']
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()
        for i in range(len(res)):
            writer.writerow({'caseid':str(i+1), 'midprice':float(res[i])})



