import xgboost as xgb
import numpy as np
import pandas as pd
from sklearn import preprocessing, cross_validation,metrics
from sklearn.metrics import mean_squared_error
from sklearn.feature_selection import f_regression
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
    data.fillna(0,inplace=True)
    newPrice = data['BidPrice1'] - data['AskPrice1']
    newVolume= data['BidVolume1'] - data['AskVolume1']
    data.insert(0,'newPrice',newPrice)
    data.insert(0,'newVolume',newVolume)
    data.insert(0,'newToVolume',newToVolume)
    data.fillna(0,inplace=True)
    predictors = [x for x in data.columns if x not in [target]]
    for i in predictors:
        data[i] = preprocessing.scale(data[i])
    return data,result

def modelfit(model, dtrain,test,predictors,result):
    model.fit(dtrain[predictors].values,dtrain[target].values)
    ans = model.predict(test[predictors].values)
    f = open('result.csv',mode= 'w')
    f.write('caseid,midprice')
    f.write('\n')
    i = 142
    while i*10+9<10000:
        temp=ans[i*10+9]+result[i*10+9]
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
xgb1 = XGBRegressor(
     learning_rate =0.01,
     n_estimators=1000,
     max_depth=6,
     objective= 'reg:linear',
    silent=True
)
modelfit(xgb1, train,test, predictors,result)