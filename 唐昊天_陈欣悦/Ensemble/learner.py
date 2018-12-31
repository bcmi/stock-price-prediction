import pandas as pd
from sklearn.linear_model import Ridge
import numpy as np
import copy
from sklearn import metrics
import csv
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.model_selection import GridSearchCV
from LSTM import LSTM
from new_data_proc import read_data, batch_formulation, read_data_test
import xgboost as xgb
from dataset import *
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class Learner(object):
    def __init__(self):
        pass
    
    def fit(self, X_train, y_train, X_test, y_test):
        pass
    
    def predict(self, X_test):
        pass

    
class XGBLearner(Learner):
    def __init__(self, booster="gbtree"):
        super(XGBLearner, self).__init__()
        params = {
        'booster': booster,
        'n_estimators': 150,
        'objective': 'reg:linear',
        'subsample': 0.8,
        'colsample_bytree': 0.85,
        'eta': 0.1,
        'alpha': 0.25,
        'lambda': 0.25,
        'gamma': 0,
        'max_depth': 8,
        'seed': 1,
        'silent': 0,
        #'scale_pos_weight': 1,
        'min_child_weight': 1,
        'eval_metric': 'rmse'
        }
        

        self.regressor = xgb.XGBRegressor(booster=params['booster'], n_estimators=params['n_estimators'], subsample = params['subsample'], objective=params['objective'],
                                    colsample_bytree=params['colsample_bytree'], learning_rate=params['eta'], reg_alpha=params['alpha'], reg_lambda=params['lambda'], gamma=params['gamma'], max_depth=params['max_depth'], seed=params['seed'], silent=params['silent'], eval_metric=params['eval_metric'])

    def fit(self, X_train, y_train, X_test, y_test):
        eval_set = [(X_train, y_train), (X_test, y_test)]
        self.regressor.fit(X_train, y_train, verbose=True, early_stopping_rounds=100, eval_metric="rmse",
                      eval_set=eval_set)
    
    
    def predict(self, X_predict):
        y_pred = self.regressor.predict(X_predict)
        return y_pred


class RFLearner(Learner):
    def __init__(self):
        super(RFLearner, self).__init__()
        self.regressor = RandomForestRegressor(n_estimators=100, max_features=8)
    
    def fit(self, X_train, y_train, X_test, y_test):
        self.regressor.fit(X_train, y_train)
        X_pred = self.regressor.predict(X_test)
        print("RMSE:",np.sqrt(metrics.mean_squared_error(X_pred, y_test)))
    
    def predict(self, X_predict):
        y_pred = self.regressor.predict(X_predict)
        return y_pred

class LRLearner(Learner):
    def __init__(self):
        super(LRLearner, self).__init__()
        self.regressor = Ridge(alpha=0.1)
    
    def fit(self, X_train, y_train, X_test, y_test):
        self.regressor.fit(X_train, y_train)
        X_pred = self.regressor.predict(X_test)
        print("RMSE:",np.sqrt(metrics.mean_squared_error(X_pred, y_test)))
    
    def predict(self, X_predict):
        y_pred = self.regressor.predict(X_predict)
        return y_pred
            
class ERTLearner(Learner):
    def __init__(self):
        super(ERTLearner, self).__init__()
        self.regressor = ExtraTreesRegressor(n_estimators=100, max_features=8)
    
    def fit(self, X_train, y_train, X_test, y_test):
        self.regressor.fit(X_train, y_train)
        X_pred = self.regressor.predict(X_test)
        print("RMSE:",np.sqrt(metrics.mean_squared_error(X_pred, y_test)))
    
    def predict(self, X_predict):
        y_pred = self.regressor.predict(X_predict)
        return y_pred

class LSTMLearner(Learner):
    def __init__(self):
        super(LSTMLearner, self).__init__()
        self.device = "cuda:0"
        self.regressor = LSTM(6, 128, 10, self.device).to(self.device)
        self.optimizer = torch.optim.Adam(self.regressor.parameters(), lr=0.001, betas=(0.9, 0.999))
        self.loss_fn = nn.L1Loss()
        self.NUM_EPOCHES = 15
        
    def fit(self, X_train, y_train, X_test, y_test):
        #Note: Here X_test and y_test are the validation set.
        self.regressor.train()
        for epoch in range(self.NUM_EPOCHES):
            print("========Start Epoch %d========" % (epoch + 1))
            loss_sum = 0
            self.predict(X_test)
            for idx, (data, gt) in enumerate(self.dataloader):
                data, gt = data.to(self.device), gt.to(self.device).view(-1)
                prediction = self.regressor(data) * 0.1
                #prediction = lstm(data)

                loss = 100 * self.loss_fn(prediction, gt)
                self.optimizer.zero_grad()
                loss.backward()

                self.optimizer.step()

                loss_sum += loss.item()
                if idx % 50 == 0:
                     print('Epoch %d, Iteration %d: loss = %.4f' % (epoch + 1, idx + 1, loss.item()))

            print('Epoch %d: loss = %.4f' % (epoch + 1, loss_sum))
            loss_sum = 0
            torch.save(self.regressor.state_dict(), "checkpoints/epoch_%s.pkl" % (epoch + 1))
            print("========End Epoch %d========" % (epoch + 1))
            
    def predict(self, X_test, cur_phase="val"):
        self.regressor.eval()
        if cur_phase == "val":
            self.dataset.phase = "val"
            self.dataloader = DataLoader(self.dataset, batch_size=32, shuffle=False)
            total_idx = 0
            total_loss = 0.
            y_pred = []
            for idx, (data, gt) in enumerate(self.dataloader):
                data, gt = data.to(self.device), gt.to(self.device).view(-1)
                prediction = self.regressor(data) * 0.1
                for i in range(prediction.size(0)):
                    y_pred.append(prediction[i])
                #prediction = lstm(data)
                loss = torch.sqrt(torch.mean((prediction - gt) ** 2))
                total_loss += loss
                total_idx += 1

            avg_loss = total_loss / total_idx
            print("========Start Evaluation========")
            print("Evaluation on validation set: %.8f" % avg_loss.item())
            print("========End Evaluation========")
            self.regressor.train()
            self.dataset.phase = "train"
            self.dataloader = DataLoader(self.dataset, batch_size=32, shuffle=True)
            y_pred = np.array(y_pred)
            return y_pred
        else:
            y_pred = []
            for idx, data in enumerate(self.test_dataloader):
                data = data.to(self.device)
                prediction = self.regressor(data) * 0.1
                for i in range(prediction.size(0)):
                    y_pred.append(prediction[i].item())
                #prediction = lstm(data)
            self.regressor.train()
            y_pred = np.array(y_pred)
            return y_pred
    
    def set_dataset(self, X_train, y_train, X_val, y_val, X_test):
        self.dataset = HFTDataset(X_train, y_train, X_val, y_val)
        self.dataloader = DataLoader(self.dataset, batch_size=32, shuffle=True)
        self.test_dataset = HFTTestDataset(X_test)
        self.test_dataloader = DataLoader(self.test_dataset, batch_size=32, shuffle=False)
    
    def re_initialize(self):
        self.regressor = LSTM(6, 128, 10, self.device).to(self.device)
        self.optimizer = torch.optim.Adam(self.regressor.parameters(), lr=0.001, betas=(0.9, 0.999))

if __name__ == '__main__':
    all_train_data = read_data()
    X_train, y_train, Train_restore, X_test, y_test, Test_restore = batch_formulation(all_train_data)[0][0]
    all_test_data = read_data_test()
    pred_data, pred_restore = batch_formulation(all_test_data, phase="test")
    
    LSTM = LSTMLearner()
    LSTM.set_dataset(X_train, y_train, X_test, y_test, pred_data)
    LSTM.fit(X_train, y_train, X_test, y_test)
    print(LSTM.predict(pred_data, "test")[142:])
    """
    
    print(RF.predict(pred_data)[142:])
    """