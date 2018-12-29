#-*- coding:utf-8 -*-
import csv
import torch
from torch.autograd import Variable
import datetime
import xgboost
import numpy as np
import random
import nltk
from nltk.util import ngrams
import gensim


def date_normalize(date, time):
    year, month, day = date.split('-')
    hour, minute, second = time.split(':')
    #print(date, time, month, day, year, hour, minute, second)
    normal_date = datetime.datetime(year=int(year), month=int(month), day=int(day), hour=int(hour), minute=int(minute), second=int(second))
    return normal_date

def file2dict(file_name):
    '''
    :param file_name:
    :data_list:[day[data{index,date,time,MP,LP,V.BP,BV,AP,AV}]]
    '''
    csv_file = csv.reader(open(file_name, 'r'))
    day_flag = True#True-a.m.,False-p.m.
    data_day_list = []
    flag = False
    data_list = []
    count = 0
    last = datetime.datetime(year=2000, month=1, day=1, hour=1, minute=1, second=30)
    for stu in csv_file:
        tmp_dict = {}
        '''
        count += 1
        if count == 10:
            break
        '''
        if len(stu) == 0:
            continue
        if flag:
            tmp_dict['index'] = int(stu[0])
            tmp_dict['date'] = date_normalize(stu[1], stu[2])
            tmp_dict['MP'] = float(stu[3])
            tmp_dict['LP'] = float(stu[4])
            tmp_dict['V'] = float(stu[5])
            tmp_dict['BP'] = float(stu[6])
            #tmp_dict['BV'] = round(float(stu[7]) / 1000)
            tmp_dict['BV'] = float(stu[7]) / 1000
            #tmp_dict['BV'] = 2 * math.atan(float(stu[7]) / 10000)/math.pi
            tmp_dict['AP'] = float(stu[8])
            #tmp_dict['AV'] = round(float(stu[9]) / 1000)
            tmp_dict['AV'] = float(stu[9]) / 1000
            tmp_dict['DV'] = float(stu[10])
            #tmp_dict['DP'] = round(float(stu[11]) * 10) / 10
            tmp_dict['DP'] = float(stu[11])
            tmp_dict['MinP'] = float(stu[12])
            #tmp_dict['MinV'] = round(float(stu[13]) / 1000)
            tmp_dict['MinV'] = float(stu[13]) / 1000
            #tmp_dict['AV'] = 2 * math.atan(float(stu[9]) / 10000)/math.pi

            if (tmp_dict['date'] - last).seconds < 0:
                continue
                #pass
            last = tmp_dict['date']
            if tmp_dict['date'].hour < 13:
                tmp_dict['time'] = 'am'
                if not day_flag:
                    data_list.append(data_day_list)
                    #print(len(data_day_list))
                    data_day_list = []
                    day_flag = True
            else:
                tmp_dict['time'] = 'pm'
                day_flag = False
            data_day_list.append(tmp_dict)
        else:
            flag = True
    data_list.append(data_day_list)
    #print(len(data_day_list))
    return data_list

def data_normalize(batch_size, data_list, model_type, generate):
    '''
    :param batch_size:
    :return: amdata[time_step[batch[V,BP,BV,AP,AV]]], amcenter[batch[MP]]
    :return: pmdata[time_step[batch[V,BP,BV,AP,AV]]], pmcenter[batch[MP]]
    '''
    amdata = [[] for i in range(10)]
    pmdata = [[] for i in range(10)]
    amcenter = []
    pmcenter = []
    for day_data in data_list:
        am_data, pm_data = day_data_normalize(day_data, generate)
        #print(len(am_data), len(pm_data))
        am_net_data, am_center = divide(am_data, batch_size, model_type, generate)
        pm_net_data, pm_center = divide(pm_data, batch_size, model_type, generate)
        #print(len(am_center), len(pm_center))
        for i in range(10):
            amdata[i] += am_net_data[i]
            pmdata[i] += pm_net_data[i]
        amcenter += am_center
        pmcenter += pm_center

    #print(amdata)
    random.seed(1)
    am_index = list(range(len(amdata[0])))
    pm_index = list(range(len(pmdata[0])))
    random.shuffle(am_index)
    random.shuffle(pm_index)
    ams_data = [[] for i in range(10)]
    pms_data = [[] for i in range(10)]
    ams_center = []
    pms_center = []
    for index in am_index:
        #print(index)
        ams_center.append(amcenter[index])
        for i in range(10):
            ams_data[i].append(amdata[i][index])
    for index in pm_index:
        pms_center.append(pmcenter[index])
        for i in range(10):
            pms_data[i].append(pmdata[i][index])

    #amdata = batch_normalize(amdata, amcenter, batch_size)
    #pmdata = batch_normalize(pmdata, pmcenter, batch_size)

    amdata = batch_normalize(ams_data, ams_center, batch_size)
    pmdata = batch_normalize(pms_data, pms_center, batch_size)

    return amdata, pmdata


def day_data_normalize(day_data, generate):
    am_data = []
    pm_data = []
    for data in day_data:
        if data['time'] == 'am':
            am_data.append(data)
        else:
            pm_data.append(data)
    aml = len(am_data)
    pml = len(pm_data)
    if generate:
        am_data = am_data[:(aml - aml % 10)]
        pm_data = pm_data[:(pml - pml % 10)]
    else:
        am_data = am_data[:(aml - aml % 30)]
        pm_data = pm_data[:(pml - pml % 30)]
    return am_data, pm_data

def divide(ap_data, batch_size, model_type, generate):

    count = 0
    net_data = [[] for i in range(10)]
    center = []
    c = []
    if batch_size >= 1:
        tmp = 0
        #print(ap_data)
        for data in ap_data:
        #ap_shuffle_index = list(range(len(ap_data)))
        #if batch_size > 1:
        #    random.shuffle(ap_shuffle_index)
        #print(ap_shuffle_index)
        #for index in ap_shuffle_index:
        #    data = ap_data[index]
            if count < 10:
                param = []
                param.append(data['MP'])
                param.append(data['BV'])
                param.append(data['AV'])
                #param.append(abs(data['AP'] - data['MP']))
                #param.append(data['MP'])
                net_data[count].append(param)
                # net_data[count].append([data['MP']])
                count += 1
                if count == 10:
                    tmp = data['MP']
            elif count < 30:
                if generate:
                    print(count, generate)
                c.append(data['MP'])
                count += 1
            if count == 30 and not generate:
                count = 0
                #c = c[9]
                c = torch.Tensor(c).mean()
                #########################################################################
                for i in range(10):
                    net_data[i][-1][0] -= tmp
                    #net_data[i][-1][0] = abs(net_data[i][-1][0])
                data_min = 100000000
                data_max = -100000000
                for index in range(len(net_data[0][0])):
                    for i in range(10):
                        if net_data[i][-1][index] > data_max:
                            data_max = net_data[i][-1][index]
                        if net_data[i][-1][index] < data_min:
                            data_min = net_data[i][-1][index]
                    for i in range(10):
                        if data_max == data_min:
                            net_data[i][-1][index] = 1
                        else:
                            net_data[i][-1][index] = (net_data[i][-1][index] - data_min) / (data_max - data_min)
                c -= tmp
                # print(ava, c)
                # print(c)
                #########################################################################
                if model_type == 'C':
                    if c < 0:
                        center.append(0)
                    #elif c == 0:
                    #    center.append(1)
                    else:
                        center.append(1)
                    # center.append(c)
                    # print(center)
                else:
                    center.append(c * 1000)
                c = []
            if count == 10 and generate:
                ava = 0
                data_min = 1
                data_max = -1
                c = data['index']
                tmp = net_data[9][-1][0]
                for i in range(10):
                    net_data[i][-1][0] -= tmp
                data_min = 100000000
                data_max = -100000000
                for index in range(len(net_data[0][0])):
                    for i in range(10):
                        if net_data[i][-1][index] > data_max:
                            data_max = net_data[i][-1][index]
                        if net_data[i][-1][index] < data_min:
                            data_min = net_data[i][-1][index]
                    for i in range(10):
                        if data_max == data_min:
                            net_data[i][-1][index] = 1
                        else:
                            net_data[i][-1][index] = (net_data[i][-1][index] - data_min) / (data_max - data_min)
                count = 0
                center.append([c, tmp])
        return net_data, center
    for index in range(len(ap_data) - 30):
        for data in ap_data[index:]:
            if count < 10:
                param = []
                param.append(data['MP'])
                param.append(data['BV'])
                param.append(data['AV'])
                # param.append(abs(data['AP'] - data['MP']))
                # param.append(data['MP'])
                net_data[count].append(param)
                # net_data[count].append([data['MP']])
                count += 1
                if count == 10:
                    tmp = data['MP']
            elif count < 30:
                if generate:
                    print(count, generate)
                c.append(data['MP'])
                count += 1
            if count == 30 and not generate:
                count = 0
                # c = c[9]
                c = torch.Tensor(c).mean()
                #########################################################################
                ava = 0
                for i in range(10):
                    ava += net_data[9][-1][0]
                tmp = ava / 10
                for i in range(10):
                    net_data[i][-1][0] -= tmp
                    # net_data[i][-1][0] = abs(net_data[i][-1][0])
                data_min = 100000000
                data_max = -100000000
                for index in range(len(net_data[0][0])):
                    for i in range(10):
                        if net_data[i][-1][index] > data_max:
                            data_max = net_data[i][-1][index]
                        if net_data[i][-1][index] < data_min:
                            data_min = net_data[i][-1][index]
                    for i in range(10):
                        if data_max == data_min:
                            net_data[i][-1][index] = 1
                        else:
                            net_data[i][-1][index] = (net_data[i][-1][index] - data_min) / (data_max - data_min)
                c -= tmp
                # print(ava, c)
                # print(c)
                #########################################################################
                if c < 0:
                    center.append(0)
                # elif c == 0:
                #    center.append(1)
                else:
                    center.append(1)
                # center.append(c)
                # print(center)
                c = []
                break
    return net_data, center

def batch_normalize(net_data, center, batch_size):
    count = 0
    l = len(center)
    data_batch = [[] for i in range(10)]
    center_batch = []
    data_tensors = []
    center_tensors = []
    for index in range(l):
        if count == batch_size:
            data_tensors.append(Variable(torch.Tensor(data_batch)))
            center_tensors.append(Variable(torch.Tensor(center_batch)))
            count = 0
            data_batch = [[] for j in range(10)]
            center_batch = []
        for i in range(10):
            data_batch[i].append(net_data[i][index])
        center_batch.append(center[index])
        count += 1
    if count == batch_size:
        data_tensors.append(Variable(torch.Tensor(data_batch)))
        center_tensors.append(Variable(torch.Tensor(center_batch)))
    print('batch_size: ', batch_size)
    print('total_batch: ', len(center_tensors))
    return [data_tensors, center_tensors]



def data_iter(file_name, batch_size, model_type='C', genrate = False):
    data_list = file2dict(file_name)
    return data_normalize(batch_size, data_list, model_type,genrate)

def divide_dataset():
    csv_file = csv.reader(open('new_dataset.csv', 'r'))
    csv_out_t = open('train.csv', 'w', newline='')
    csv_out_v = open('valid.csv', 'w', newline='')
    csv_write_t = csv.writer(csv_out_t, dialect='excel')
    csv_write_v = csv.writer(csv_out_v, dialect='excel')
    csv_write_t.writerow(['', 'Date', 'Time', 'MidPrice', 'LastPrice', 'Volume', 'BidPrice1',
                         'BidVolume1', 'AskPrice1', 'AskVolume1', 'DeltaVolume', 'DeltaMP',
                         'MinPrice', 'MinVolume'])
    csv_write_v.writerow(['', 'Date', 'Time', 'MidPrice', 'LastPrice', 'Volume', 'BidPrice1',
                         'BidVolume1', 'AskPrice1', 'AskVolume1', 'DeltaVolume', 'DeltaMP',
                         'MinPrice', 'MinVolume'])
    flag = False
    for stu in csv_file:
        tmp_dict = {}
        '''
        count += 1
        if count == 10:
            break
        '''
        if len(stu) == 0:
            continue
        if flag:
            tmp_dict['date'] = date_normalize(stu[1], stu[2])
            if tmp_dict['date'].month == 9:
                csv_write_v.writerow(stu)
            else:
                csv_write_t.writerow(stu)
        else:
            flag = True

########################################################################################################################
#TODO: XGBoost特征工程
def MLgenerate(file_name):
    csv_file = csv.reader(open(file_name, 'r'))
    flag = False
    res = []
    tmp_data = []
    center = []
    c = 0
    tmp = 0
    for stu in csv_file:
        if len(stu) == 0:
            res.append(tmp_data)
            tmp_data = []
            center.append([c, tmp])
            continue
        if flag:
            tmp_data += [round(float(stu[7]) / 1000),
                         round(float(stu[9]) / 1000),
                         float(stu[10]),
                         round(float(stu[11]) * 10) / 10,
                         round(float(stu[13]) / 1000)]
            c = int(stu[0])
            tmp = float(stu[3])

        else:
            flag = True
    return np.array(res), np.array(center)


def MLdata(file_name, generate = False):
    data_list = file2dict(file_name)
    count = 0
    tmp_data = []
    res = []
    c = []
    tmp = 0
    center = []
    total = 0
    for day_data in data_list:
        am_data, pm_data = day_data_normalize(day_data, generate)
        all_data = am_data + pm_data
        for data in all_data:
            total += 1
            if count < 10:
                tmp_data.append(data['BV'])
                tmp_data.append(data['AV'])
                tmp_data.append(data['DV'])
                #tmp_data.append(data['DP'])
                #tmp_data.append(data['AP'])
                #tmp_data.append(data['BP'])
                tmp_data.append(data['MP'])
                #tmp_data.append(data['MinP'])
                #tmp_data.append(data['MinV'])

                count += 1
                if count == 10:
                    tmp = data['MP']
            elif count < 30:
                if generate:
                    print(count, generate)
                c.append(data['MP'])
                count += 1
            if count == 30 and not generate:
                count = 0
                c = np.array(c).mean()
                c -= tmp
                c *= 1000
                #print(c)
                '''
                if c < 0:
                    center.append(0)
                else:
                    center.append(1)
                '''
                center.append(c)
                c = []
                res.append(np.array(tmp_data))
                tmp_data = []
            if count == 10 and generate:
                c = data['index']
                res.append(np.array(tmp_data))
                tmp_data = []
                count = 0
                #print(c, tmp)
                center.append([c, tmp])
    #print(total)
    if generate:
        return np.array(res), np.array(center)
    else:
        new_res = []
        new_center = []
        res_index = list(range(len(res)))
        random.seed(0)
        #random.shuffle(res_index)
        #print(center)
        for index in res_index:
            new_res.append(res[index])
            new_center.append(center[index])
        #return xgboost.DMatrix(np.array(new_res), label = np.array(center))
        return np.array(new_res), np.array(center)

def Formulize():
    csv_file = csv.reader(open('test_data.csv', 'r'))
    csv_out = open('new_testset.csv', 'w', newline = '')
    csv_writer = csv.writer(csv_out, dialect='excel')
    csv_writer.writerow(['', 'Date', 'Time', 'MidPrice', 'LastPrice', 'Volume', 'BidPrice1',
                         'BidVolume1', 'AskPrice1', 'AskVolume1', 'DeltaVolume', 'DeltaMP',
                         'MinPrice', 'MinVolume'])
    flag = False
    milestone = datetime.datetime(year=2000, month=1, day=1, hour=1, minute=1, second=30)
    last = datetime.datetime(year=2000, month=1, day=1, hour=1, minute=1, second=30)
    lastday = last.date()
    milestoneday = milestone.date()
    lastDV = 0
    lastDMP = 0
    th1 = datetime.time(hour = 9, minute = 0, second = 0)
    th2 = datetime.time(hour = 11, minute = 30, second = 0)
    th3 = datetime.time(hour=13, minute=0, second=0)
    th4 = datetime.time(hour=15, minute=0, second=0)
    for stu in csv_file:
        if len(stu) == 0:
            last = milestone
            lastday = milestoneday
            csv_writer.writerow(stu)
            continue
        if flag:
            d = date_normalize(stu[1], stu[2])
            dt = d.time()
            dd = d.date()
            if (dt >= th1 and dt <= th2) or (dt >= th3 and dt <= th4):
                if (d - last).seconds >= 3:
                    #print(last, d, d - last)
                    if dd != lastday:
                        stu.append(0)
                        stu.append(0)
                    else:
                        stu.append((float(stu[5]) - lastDV) / 100)
                        stu.append((float(stu[3]) - lastDMP) * 1000)
                    stu.append(min(float(stu[6]), float(stu[8])))
                    stu.append(min(float(stu[7]), float(stu[9])) / 100)
                    csv_writer.writerow(stu)
                    last = d
                    lastday = dd
                    lastDV = float(stu[5])
                    lastDMP = float(stu[3])
        else:
            flag = True

########################################################################################################################
#TODO:训练词向量数据处理

def get_word_dict():
    csv_file = csv.reader(open('new_dataset.csv', 'r'))
    flag = False
    word_dict = {}
    for stu in csv_file:
        if flag:
            tmp = (int(float(stu[10]) / 100), round(float(stu[11])), int(float(stu[13]) / 100))
            word_dict[tmp] = word_dict.get(tmp, 0) + 1
        else:
            flag = True
    return word_dict

def get_longsent():
    csv_file = csv.reader(open('new_dataset.csv', 'r'))
    flag = False
    begintime = datetime.datetime(year=2018, month=6, day=1, hour=9, minute=30, second=0)
    sents = []
    sent = []
    for stu in csv_file:
        if flag:
            d = date_normalize(stu[1], stu[2])
            tmp = (round(float(stu[10]) / 100), round(float(stu[11])), round(float(stu[7]) / 100), round(float(stu[9]) / 100))
            if (d - begintime).seconds > 60:
                sents.append(sent)
                sent = []
            else:
                sent.append(tmp)
            begintime = d
        else:
            flag = True
    return sents

def generate_sent(sents, l):
    g_file = open('gramma.txt', 'w')
    for sent in sents:
        short_sent = list(ngrams(sent, l))
        for s in short_sent:
            for word in s:
                tu = str(word)
                tu = ''.join(tu.split(' '))
                tu = 'b'.join(tu.split('('))
                tu = 'b'.join(tu.split(')'))
                tu = 'm'.join(tu.split('-'))
                tu = 'd'.join(tu.split(','))
                tu = 'p'.join(tu.split('.'))
                g_file.write(''.join(tu))
                g_file.write(' ')
            g_file.write('\n')
    g_file.close()

def nlp_data_iter(file_name, batch_size):
    csv_file = csv.reader(open(file_name, 'r'))
    vocable = gensim.models.Word2Vec.load('word2vec.bin')

    flag = False
    begintime = datetime.datetime(year=2018, month=6, day=1, hour=9, minute=30, second=0)
    sents = []
    sent = []
    c = 0
    for stu in csv_file:
        #c += 1
        #if c == 10000:
        #    break
        if flag:
            d = date_normalize(stu[1], stu[2])
            tmp = (float(stu[3]), int(float(stu[10]) / 100), round(float(stu[11])), round(float(stu[7]) / 100), round(float(stu[9]) / 100))
            word = str(tmp[1:])
            word = ''.join(word.split(' '))
            word = 'b'.join(word.split('('))
            word = 'b'.join(word.split(')'))
            word = 'm'.join(word.split('-'))
            word = 'd'.join(word.split(','))
            word = 'p'.join(word.split('.'))
            #print(word)
            if word not in vocable:
                vector = Variable(torch.zeros((100)))
            else:
                vector = Variable(torch.Tensor(vocable[word]))

            if (d - begintime).seconds > 60:
                sents.append(sent)
                sent = []
            else:
                sent.append((tmp[0], vector))
            begintime = d
        else:
            flag = True
    #print(sents)

    dataset = []
    lp = 0
    for day_ap in sents:

        if batch_size >= 1:
            c = 0
            all_sents = []
            tmp_sent = []
            for data in day_ap:
                if c == 30:
                    all_sents.append(tmp_sent)
                    c = 0
                    tmp_sent = []
                tmp_sent.append(data)
                c += 1
        else:
            all_sents = ngrams(day_ap, 30)

        for data in all_sents:
            mp = 0
            single_data = []
            for i, tp in enumerate(data):
                if i < 10:
                    single_data.append(tp[1])
                    if i == 9:
                        lp = tp[0]
                else:
                    mp += tp[0]
            ava = mp / 20
            #print(ava, lp)
            if ava < lp:
                label = Variable(torch.Tensor([0]))
            else:
                label = Variable(torch.Tensor([1]))
            dataset.append((single_data, label))

    random.shuffle(dataset)


    all_batch = []
    single_batch = []
    count = 0
    for data in dataset:
        if count == batch_size:
            count = 0
            net_data = [[] for i in range(10)]
            labels = []
            for d in single_batch:
                for i in range(10):
                    net_data[i].append(d[0][i])
                labels.append(d[1])
            for i in range(10):
                net_data[i] = torch.stack(net_data[i])
            labels = torch.cat(labels)
            single_batch = (net_data, labels)
            all_batch.append(single_batch)
            single_batch = []
        single_batch.append(data)
        count += 1
    print('batch_size:%d    total:%d'%(batch_size, len(all_batch)))

    return all_batch






if __name__ == '__main__':
    #divide_dataset()
    #MLdata('train_data.csv')
    #Formulize()
    #get_word_dict()
    sents = get_longsent()
    generate_sent(sents, 30)
    #g_file = open('gramma.txt', 'r')
    #for line in g_file:
    #    print(line)
    #    print(1)
    #nlp_data_iter('train.csv', 8)



