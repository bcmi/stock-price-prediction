import xgboost as xgb
import numpy as np
import pandas as pd
from sklearn import preprocessing, cross_validation,metrics
from sklearn.metrics import mean_squared_error
from sklearn.feature_selection import f_regression
from sklearn.cross_validation import *
from xgboost.sklearn import XGBRegressor
from sklearn.grid_search import GridSearchCV
import matplotlib.pylab as plt
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 12, 4
from xgboost import plot_importance

target='MidPrice'
def prepropressingData(filepath):
    data = pd.read_csv(filepath,usecols=['MidPrice','Volume','BidPrice1','BidVolume1','AskPrice1','AskVolume1'],dtype=np.float64)
    result = data['MidPrice'].copy()
    data['MidPrice'] = data['MidPrice'].shift(-1)-data['MidPrice']
    newToVolume = data['Volume'].shift(-1)-data['Volume']
    newToVolume.fillna(0,inplace=True)
    newPrice = data['BidPrice1'] - data['AskPrice1']
    newVolume= data['BidVolume1'] - data['AskVolume1']
    data.fillna(0,inplace=True)
    data.insert(0,'newPrice',newPrice)
    data.insert(0,'newVolume',newVolume)
    data.insert(0,'newToVolume',newToVolume)
    # predictors = [x for x in data.columns if x not in [target]]
    # for i in predictors:
    #      data[i] = preprocessing.scale(data[i])
    return data,result

def modelfit(model,parameters, dtrain,test,predictors,result):
    clf = GridSearchCV(xgb_model, parameters, n_jobs=5,
                       cv=StratifiedKFold(dtrain[target].values, n_folds=5, shuffle=True),
                       # scoring='roc_auc',
                       verbose=2, refit=True)
    clf.fit(dtrain[predictors].values,dtrain[target].values)
    best_parameters, score, _ = max(clf.grid_scores_, key=lambda x: x[1])
    print('Raw AUC score:', score)
    for param_name in sorted(best_parameters.keys()):
        print("%s: %r" % (param_name, best_parameters[param_name]))
    ans = clf.predict(test[predictors].values)
    f = open('result.csv',mode= 'w')
    f.write('caseid,midprice')
    f.write('\n')
    i = 0
    while i*10+9<10000:
        temp=ans[i*10+9]/10+result[i*10+9]
        f.write(str(i+1)+','+str(temp))
        f.write('\n')
        i += 1
    print(mean_squared_error(test[target].values,ans))
    f.close()
    plot_importance(model)
    plt.show()

train,_ = prepropressingData('train_data.csv')
test,result = prepropressingData('test_data.csv')
predictors = [x for x in train.columns if x not in [target]]
xgb_model = XGBRegressor()

parameter = {
    'nthread': [4],  # multi-threading
    'objective': ['reg:linear'],
    'learning_rate': [0.01,0.05,0.1,0.2],
    'max_depth': [2,4,6,8],
    'min_child_weight': [1,3,5,7,9,11],
    'silent': [1],
    'subsample': [0.8],
    'colsample_bytree': [0.7],
    'n_estimators': [1000],  # number of trees
    'missing': [-999],
    'seed': [1337]
}


modelfit(xgb_model, parameter,train,test, predictors,result)