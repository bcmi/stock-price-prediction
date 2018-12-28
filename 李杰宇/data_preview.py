#-*- coding:utf-8 -*-
import csv
from dataprepro import data_iter, file2dict
import matplotlib
from matplotlib import pyplot as plt
import numpy as np
from dataprepro import *
from minepy import MINE
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, ShuffleSplit
from sklearn.linear_model import Ridge, LinearRegression

'''
am, pm = data_iter('train.csv', 1)

am_data = am[0]
am_center = am[1]
pm_data = pm[0]
pm_center = pm[1]


ava = []
center = []
index_list = []
rand_list= []
np.random.seed(0)
last = []
loss = []
ups = []
downs = []
eqs = []
up = 0
down = 0
eq = 0
data_min = 0
data_max = -1
for index, data in enumerate(am_data):
    if data.min() < data_min:
        data_min = data.min()
    if data.max() > data_max:
        data_max = data.max()
    if index % 100 != 0:
        for i in range(9):
            if data[i] < data[i + 1]:
                up  += 1
            elif data[i] == data[i + 1]:
                eq += 1
            else:
                down += 1
    else:
        ups.append(up)
        downs.append(down)
        eqs.append(eq)
        up = 0
        down = 0
        eq = 0
        index_list.append(index // 100)
    ava.append(data.mean())
    last.append(data[-1])
    center.append(am_center[index])
    loss.append(am_center[index] - data[-1])

ava = np.array(ava)
ava_line = ava.mean()

#k = np.array(loss) ** 2
#k = np.sqrt(k.mean())
#print(k)
'''
'''
for index, data in enumerate(am_data):
    if index % 100 != 0:
        pass
    if am_center[index] > float(ava_line):
        rand_list.append(float(ava_line) + k)
    else:
        rand_list.append(float(ava_line) - k)
'''
'''

#plt.plot(np.array(index_list), ava)
#plt.plot(np.array(index_list), np.array(last))
#plt.plot(np.array(index_list), np.array([ava_line] * len(index_list)))
#plt.plot(np.array(index_list), np.array(center))
#plt.plot(np.array(index_list), np.array(rand_list))
#plt.plot(np.array(index_list), np.array(loss))
#plt.bar(index_list, ups)
#plt.show()
#plt.bar(index_list, downs)
#plt.show()
#plt.bar(index_list, eqs)
#plt.show()

ava = []
center = []
#index_list = []
rand_list= []
last = []
loss = []
#ups = []
#downs = []
#eqs = []
up = 0
down = 0
eq = 0

print(data_min, data_max)
for index, data in enumerate(pm_data):
    if data.min() < data_min:
        data_min = data.min()
    if data.max() > data_max:
        data_max = data.max()
    if index % 100 != 0:
        for i in range(9):
            if data[i] < data[i + 1]:
                up  += 1
            elif data[i] == data[i + 1]:
                eq += 1
            else:
                down += 1
    else:
        ups.append(up)
        downs.append(down)
        eqs.append(eq)
        up = 0
        down = 0
        eq = 0
        index_list.append(62 + index // 100)
    ava.append(data.mean())
    last.append(data[-1])
    center.append(pm_center[index])
    loss.append(pm_center[index] - data[-1])

ava = np.array(ava)
ava_line = ava.mean()
#k = np.array(loss) ** 2
#k = np.sqrt(k.mean())
#print(k)
'''
'''
for index, data in enumerate(pm_data):
    if index % 100 != 0:
        pass
    if pm_center[index] > float(ava_line):
        rand_list.append(float(ava_line) + k)
    else:
        rand_list.append(float(ava_line) - k)
'''
'''

#plt.plot(np.array(index_list), ava)
#plt.plot(np.array(index_list), np.array(last))
#plt.plot(np.array(index_list), np.array([ava_line] * len(index_list)))
#plt.plot(np.array(index_list), np.array(center))
#plt.plot(np.array(index_list), np.array(rand_list))
#plt.plot(np.array(index_list), np.array(loss))
#plt.show()
print(data_min, data_max)
print(np.array(ups).mean())
print(np.array(downs).mean())
print(np.array(eqs).mean())
plt.bar(index_list, ups)
plt.show()
plt.bar(index_list, downs)
plt.show()
plt.bar(index_list, eqs)
plt.show()
'''
'''
csv_file = csv.reader(open('test_data.csv', 'r'))
flag = False
data_list = []
index = []
count = 0
tmp_dict = []
for stu in csv_file:
    if len(stu) == 0:
        tmp = np.array(tmp_dict).mean()
        #print(tmp_dict)
        data_list.append(tmp)
        count += 1
        tmp_dict = []
        index.append(count)
        continue
    if flag:
        #print(stu)
        tmp_dict.append(float(stu[3]))

    else:
        flag = True

mid = []
for i, data in enumerate(data_list):
    if i == len(data_list) - 1:
        mid.append([data,data])
    else:
        mid.append([0.7 * data + 0.3 * data_list[i + 1], data])


plt.plot(np.array(index[:100]), np.array(data_list[:100]))
plt.show()
plt.plot(np.array(index[100:200]), np.array(data_list[100:200]))
plt.show()
plt.plot(np.array(index[200:300]), np.array(data_list[200:300]))
plt.show()
plt.plot(np.array(index[300:400]), np.array(data_list[300:400]))
plt.show()
plt.plot(np.array(index[400:500]), np.array(data_list[400:500]))
plt.show()
plt.plot(np.array(index[500:600]), np.array(data_list[500:600]))
plt.show()
plt.plot(np.array(index[600:700]), np.array(data_list[600:700]))
plt.show()
plt.plot(np.array(index[700:800]), np.array(data_list[700:800]))
plt.show()
plt.plot(np.array(index[800:900]), np.array(data_list[800:900]))
plt.show()
plt.plot(np.array(index[900:]), np.array(data_list[900:]))
plt.show()
#plt.plot(np.array(index), np.array(mid))
plt.show()

csv_out = open('try.csv', 'w', newline='')
csv_write = csv.writer(csv_out, dialect='excel')
csv_write.writerow(['caseid', 'midprice'])
count = 1
for m in mid:
    if count < 143:
        count += 1
        continue
    csv_write.writerow([count, m[0], m[1]])
    count += 1
'''
'''

am, pm = data_iter('train.csv', 1)
amdata = am[0]
amcenter = am[1]
pmdata = pm[0]
pmcenter = pm[1]
def test(k):
    acc = 0
    total = 0

    for i, data in enumerate(amdata):
        total += 1
        if i < k:
            if amcenter[i] == 0:
                acc += 1
        else:
            if amcenter[i] == amcenter[i - k]:
                acc += 1

    for i, data in enumerate(pmdata):
        total += 1
        if i < k:
            if pmcenter[i] == 0:
                acc += 1
        else:
            if pmcenter[i] == pmcenter[i - k]:
                acc += 1

    print(k, acc, total, acc / total)

for i in range(3):
    test(i + 1)
'''

'''
f1 = csv.reader(open('try.csv', 'r'))
f2 = csv.reader(open('prediction.csv', 'r'))
flag = False
try_data = []
pred_data = []
real = []

for stu in f1:
    if flag:
        try_data.append(float(stu[1]))
        real.append(float(stu[2]))
    else:
        flag = True

flag = False
for stu in f2:
    if flag:
        pred_data.append(float(stu[1]))
    else:
        flag = True

ava = 0
dif = 0
for i in range(858):
    flag = (pred_data[i] - real[i]) * target[i] >= 0
    #print(flag)
    ava += flag
    dif += abs(pred_data[i] - try_data[i])

print(ava / 858, dif / 858)
#print(target)
plt.plot(range(858), try_data, 'r')
plt.plot(range(858), pred_data)
#plt.yticks([3, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8])
#plt.show()

'''

def classify():
    csv_file = csv.reader(open('new_dataset.csv', 'r'))
    flag = False
    up_bp = []
    up_ap = []
    down_bp = []
    down_ap = []
    eq_bp = []
    eq_ap = []
    count = 0
    up = []
    eq = []
    down = []
    last = 0
    firsttime = datetime.time(hour = 9, minute = 30, second = 3)
    for stu in csv_file:
        if flag:
            date = date_normalize(stu[1], stu[2])
            if date.time() < firsttime:
                #print(stu)
                continue
            mp = float(stu[3])
            ap = float(stu[9]) / 10000
            bp = float(stu[7]) / 10000
            dv = float(stu[10]) / 10000
            minp = float(stu[11])
            minv = float(stu[12]) / 10000
            if mp > last:
                if dv > 200 or minv > 100:
                    continue
                if 1 / minv > 150:
                    continue
                #print(1 / minv)
                up_bp.append(dv)
                up_ap.append(minv)
            elif mp == last:
                if dv > 300 or minv > 250:
                    continue
                if dv / minv > 20000:
                    continue
                if 1 /minv > 200:
                    continue
                eq_bp.append(dv)
                eq_ap.append(1 / minv)
            else:
                if dv > 200 or minv > 100:
                    continue
                if 1 / minv > 150:
                    continue
                down_bp.append(dv)
                down_ap.append(1 / minv)
            count += 1
            last = mp
        #if count == 500000:
        #    break
        else:
            flag = True

    plt.scatter(up_bp, up_ap)
    plt.show()
    plt.scatter(eq_bp, eq_ap)
    plt.show()
    plt.scatter(down_bp, down_ap)
    plt.show()

def distribution():
    dv = []
    dm = []
    mv = []
    csv_file = csv.reader(open('new_dataset.csv', 'r'))
    flag = False
    for stu in csv_file:
        if flag:
            if float(stu[10]) > 25:
                continue
            if float(stu[13]) > 100:
                continue
            dv.append(float(stu[10]))
            dm.append(float(stu[11]))
            mv.append(float(stu[13]))
        else:
            flag = True

    plt.hist(dv, bins = 25)
    plt.show()
    plt.hist(dm, bins = 100)
    plt.show()
    plt.hist(mv, bins = 100)
    plt.show()

def data2word():
    csv_file = csv.reader(open('new_dataset.csv', 'r'))
    flag = False
    worddict = {}
    count = 0
    max_count = 0
    for stu in csv_file:
        if flag:
            tmp = (int(float(stu[10]) / 100), round(float(stu[11])), int(float(stu[13]) / 100))
            worddict[tmp] = worddict.get(tmp, 0) + 1
            count += 1
            if worddict[tmp] > max_count:
                max_count = worddict[tmp]
        else:
            flag = True
    print(len(worddict), count, max_count)
    count_list = []
    mt1 = 0
    for k in worddict:
        count_list.append(worddict[k])
        if worddict[k] > 1:
            mt1 += 1
    print(mt1)
    print(count_list)



def MIC():
    train_x, train_y = MLdata('new_dataset.csv')
    mic = MINE()
    l = train_x.shape[1]
    print(l)
    for i in range(l):
        mic.compute_score(train_x[:,i], train_y)
        print(i, mic.mic())

def RFR():
    train_x, train_y = MLdata('new_dataset.csv')
    rf = RandomForestRegressor(n_estimators=20, max_depth=4)
    for i in range(train_x.shape[1]):
        score = cross_val_score(rf, train_x[:, i:i+1], train_y ,scoring='neg_mean_squared_error',
                              cv=ShuffleSplit(len(train_x), 3, .3))
        print(np.mean(score), i)

def AVA():
    x, y = MLdata('new_dataset.csv')
    index = np.linspace(0, y.shape[0] - 1, 200, dtype = int)
    y_ava = []
    y_mse = []
    y_dif = []
    last = 0
    for i in index:
        mse = np.sqrt((y[last:i] ** 2).mean())
        ava = abs(y[last:i]).mean()
        print(ava, mse)
        y_ava.append(ava)
        y_mse.append(mse)
        y_dif.append(mse - ava)
        last = i

    plt.plot(index, y_dif)
    plt.plot(index, y_ava)
    #plt.show()
    plt.plot(index, y_mse)
    plt.show()


target = []
f = csv.reader(open('new_testset.csv', 'r'))
flag = False
cand = {'u':0,'d':0}
tmp = []
for stu in f:
    if flag:
        if len(stu) == 0:
            for i in range(9):
                if tmp[i + 1] >= tmp[i]:
                    cand['u'] += 1
                else:
                    cand['d'] += 1
            target.append(cand['u'] - 3.1 * cand['d'])
            cand = {'u': 0, 'd': 0}
            tmp = []
        else:
            tmp.append(float(stu[3]))
    else:
        flag = True

target = target[142:]
#print(len(target))
#print(target)

