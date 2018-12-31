# In[]
import imp
import lightgbm as lgb
import numpy as np
import pandas as pd
import pickle
import time
import csv

# In[]
def resultRenormalize(data,mean,sigma):
    return data * sigma[0] + mean[0]

def predict_writeout(predict):
    # write csv
    with open("./predict.csv", "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(
            ["caseid", "MidPrice"])
        for i in range(142, len(predict)):
            writer.writerow([(1 + i), predict[i].item()])
        
def shuffle(data,label):
    np.random.seed(int(time.time()))
    randomList = np.arange(data.shape[0])
    np.random.shuffle(randomList)
    return data[randomList], label[randomList]

def splitdata(data,label,rate = 0.3):
    train_data = data[int(data.shape[0] * rate):]
    train_label = label[int(label.shape[0] * rate):]
    val_data = data[:int(data.shape[0] * rate)]
    val_label = label[:int(label.shape[0] * rate)]
    return train_data, train_label, val_data, val_label


# In[]
data, label = pickle.load(open('generated_train_data_feature_plus.pkl','rb'))
data,label = shuffle(data,label)
train_data, train_label, val_data, val_label = splitdata(data,label,rate=0.05)
# test_data, mean2, std2 = pickle.load(open('generated_test_data_feature_plus.pkl','rb'))
print(train_data.shape,train_label.shape)
print(train_data[:5],train_label[:5])
# print(test_data[:2])
trainData = lgb.Dataset(train_data,train_label)
validData = lgb.Dataset(val_data,val_label)
# testData = lgb.Dataset(test_data)

# In[]
# params = {
#     'boosting': 'gbdt',
#     'objective': 'regression',
#     # 'min_data_in_leaf': 200,
#     'learning_rate': 0.01,
#     'max_bin': 32767,
#     'num_iterations': 1000,
#     'metric': 'l2',
#     'num_leaves': 511,
#     'early_stopping_round': 5,
#     # 'num_threads': 16
# }
params = {
    'num_leaves': 80,
    'objective': 'regression',
    'min_data_in_leaf': 200,
    'learning_rate': 0.02,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.7,
    'bagging_freq': 1,
    'metric': 'l2',
    'num_threads': 16,
    'early_stopping_rounds':20,
    'num_iterations':2000
}

print('trianing...')

model = lgb.train(
    params, trainData,
    valid_sets=[trainData,validData], valid_names=['train','valid'], 
    # valid_sets=[trainData,validData], valid_names=['train','valid'], 
    verbose_eval=1
)

# In[]
test = pd.DataFrame(pd.read_csv("test_data_feature_plus.csv"))
test = test.drop(columns=['ID', 'Date', 'Time']).values
print('test shape:',test.shape)
pred = np.zeros(test.shape[0]//10)
for i in range(test.shape[0]//10):
    item = test[i*10:(i+1)*10]
    item_mean = item.mean(axis=0)
    item_std = item.std(axis=0)
    item = (item-item_mean)/(item_std+1)
    item = item.reshape(-1,100)
    item_pred = model.predict(item)[0]
    # print(item.shape,item_pred)
    pred[i] = item_pred * (item_std[0] + 1) + item_mean[0]
    # pred = resultRenormalize(pred,mean2,std2)
print(pred[1:10])
print(pred[143].item())
predict_writeout(pred)