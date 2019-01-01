from sklearn.model_selection import KFold
import numpy as np
from learner import *
from sklearn.linear_model import LinearRegression

class StackEnsemble(object):
    def __init__(self, datasets, all_labels, test_dataset, holdout_indices, stacker, base_models):
        #N-fold dataset
        self.nfold = datasets
        #labels for all training data
        self.all_labels = all_labels
        #test data
        self.T = test_dataset[0]
        #test data restoration parameter (mid price of the last time point)
        self.T_restore = test_dataset[1]
        #holdout indices for each fold
        self.holdout_indices = holdout_indices
        #stacking predictor
        self.stacker = stacker
        #all the base models
        self.base_models = base_models

    def run(self):
        #N-fold training. Each model use N-1 folds as training samples in each run, and predict on fold N & the test dataset.
        #X: training data, y: training ground truth, T: test data
        
        X_train, y_train, Train_restore, X_test, y_test, Test_restore = self.nfold[0]
        S_train = np.zeros((X_train.shape[0]+X_test.shape[0], len(self.base_models)))
        y = self.all_labels
        
        S_test = np.zeros((self.T.shape[0], len(self.base_models)))

        for i in range(len(self.base_models)):
            S_test_i = np.zeros((self.T.shape[0], len(self.nfold)))
            
            for fold_idx in range(len(self.nfold)):
                X_train, y_train, Train_restore, X_test, y_test, Test_restore = self.nfold[fold_idx]
                
                if i == 0:
                    #LSTM
                    self.base_models[i].set_dataset(X_train, y_train, X_test, y_test, self.T)
                    if fold_idx != 0:
                        self.base_models[i].re_initialize()
                    
                self.base_models[i].fit(X_train, y_train, X_test, y_test)
                y_pred = self.base_models[i].predict(X_test)[:]
                S_train[self.holdout_indices[fold_idx], i] = y_pred
                
                if i == 0:
                    S_test_i[:, fold_idx] = self.base_models[i].predict(self.T, "test")[:]
                else:
                    S_test_i[:, fold_idx] = self.base_models[i].predict(self.T)[:]
                
                #S_test_i[:, fold_idx] = self.base_models[i].predict(self.T)[:]
            #for each model, its prediction on test is the average prediction of models trained with n folds.
            S_test[:, i] = S_test_i.mean(1)
        
        self.stacker.fit(S_train, y)
        S_train_pred = self.stacker.predict(S_train)
        print("Final RMSE:",np.sqrt(metrics.mean_squared_error(S_train_pred, y)))
        
        y_pred = self.stacker.predict(S_test)[:]
        return y_pred
    
    def give_prediction(self):
        y_pred = self.run()
        for i in range(len(y_pred)):
            y_pred[i] = y_pred[i]+self.T_restore[i][0]
        print(y_pred[142:242])
        with open('super_ensemble.csv','w') as fout:
            fieldnames = ['caseid','midprice']
            writer = csv.DictWriter(fout, fieldnames = fieldnames)
            writer.writeheader()
            for i in range(142,len(y_pred)):
                writer.writerow({'caseid':str(i+1),'midprice':float(y_pred[i])})
    
if __name__ == '__main__':
    import pickle
    XGB = XGBLearner()
    XGB_dart = XGBLearner("dart")
    RF = RFLearner()
    ERT = ERTLearner()
    LSTM = LSTMLearner()
    LR = LRLearner()
    #basemodels = [LSTM, RF, XGB, XGB_dart]
    basemodels = [LSTM, RF, XGB, XGB_dart]
    
    """
    f = open('DATA.pkl','wb')
    all_train_data = read_data()
    all_test_data = read_data_test()
    pickle.dump([all_train_data, all_test_data], f)
    f.close()
    """
    
    f = open("DATA.pkl","rb")
    lis = pickle.load(f)
    f.close()
    all_train_data, all_test_data = lis[0], lis[1]
    
    train_processed = batch_formulation(all_train_data)
    nfolds, holdout_indices, all_labels = train_processed[0], train_processed[1], train_processed[2]
    stacker = LinearRegression()
    
    test_dataset = batch_formulation(all_test_data, "test")
    ensemble = StackEnsemble(nfolds, all_labels, test_dataset, holdout_indices, stacker, basemodels)
    
    ensemble.run()
    ensemble.give_prediction()