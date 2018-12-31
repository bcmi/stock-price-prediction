from sklearn import neighbors
from sklearn.ensemble import RandomForestRegressor
from sklearn import svm
from sklearn import ensemble
from sklearn.tree import DecisionTreeRegressor
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.externals import joblib
import numpy as np
import csv


tmp = np.loadtxt("30-train_data.csv", dtype=np.str, delimiter=",")
X = tmp[1:,0:30].astype(np.float)#加载数据部分
y = tmp[1:,30].astype(np.float)#加载类别标签部分
x_train, x_test, y_train, y_test = train_test_split(X, y)


'''params={}

rf =ensemble.RandomForestRegressor(n_estimators=40,max_depth=11,min_samples_split=10)
grid=GridSearchCV(rf,params,cv=5)

grid.fit(x_train,y_train)

print(grid.best_score_)
print(grid.best_params_)
'''
rf=ensemble.RandomForestRegressor()
rf.fit(x_train,y_train)
result1 = rf.predict(x_test)
print(rf.score(x_test,y_test))


tmp=np.loadtxt("30-test_data.csv", dtype=np.str, delimiter=",")
m = tmp[1:,0:30].astype(np.float)#加载数据部分
result2=rf.predict(m)
print(result2)


'''
out=open('result1.csv','a',newline='')
imf=['caseid','midprice']
csv_write=csv.writer(out,dialect='excel')

csv_write.writerow(imf)
for j in range(143,1001):
    imf=[j,result2[j-1]]
    csv_write.writerow(imf)
out.close()'''
        
