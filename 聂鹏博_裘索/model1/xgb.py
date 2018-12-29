import pandas as pd
import xgboost as xgb
import numpy as np
from sklearn.linear_model import LinearRegression

def train_test(reframed):
    # split into train and test sets
    train = reframed[:, :]
    # split into input and outputs
    train_X, train_y = train[:, :-1], train[:, -1]
    # reshape input to be 3D [samples, timesteps, features]
    return train_X,train_y

def run(fileindex):
    '''reframed = np.load('reframed_train%s.npy'%str(fileindex))
    reframed_test = np.load('reframed_test%s.npy'%str(fileindex))

    train_X,train_y= train_test(reframed)'''
    train_X = pd.read_csv('train_X.csv',index_col = None, header = None)
    train_X = train_X.values
    train_y = pd.read_csv('train_Y.csv',index_col = None, header = None)
    train_y = train_y.values
    predict_X = pd.read_csv('predict_X.csv',index_col = None, header = None)
    predict_X = predict_X.values

    model = xgb.XGBRegressor(silent=False,max_depth=6, learning_rate=0.1, n_estimators=160,subsample =0.5)
    #model=LinearRegression()
    model.fit(train_X,train_y)

    dataset = pd.read_csv('test_data.csv',  parse_dates = [['Date', 'Time']], index_col=0)
    dataset.drop('Unnamed: 0', axis=1, inplace=True)
    datavalue = dataset.values
    
    test_Y = model.predict(predict_X)
    print(test_Y)
    print(len(test_Y))
    data = {'caseid':[i+1 for i in range(1000)], 'midprice':[datavalue[9+10*i, 0]+test_Y[i] for i in range(1000)]}
    df = pd.DataFrame(data)
    df = df[142:]
    df.to_csv("rest%s.csv"%str(fileindex), index = False)

for i in range(1):
    run(1)
