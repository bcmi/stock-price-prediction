#-*- coding:utf-8 -*-
import torch
import torch.nn as nn
from torch import autograd
from torch.autograd import Variable
from dataprepro import data_iter, MLdata, nlp_data_iter, MLgenerate
from module import DiaNet, ClassiNet
import torch.optim as optim
import time
from config import config_dict
import csv
import xgboost
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Lasso, LassoCV, LassoLarsCV
import math
import numpy as np
from matplotlib import pyplot as plt
from xgboost import plot_importance
from tensorboardX import SummaryWriter



class Trainer():
    def __init__(self, config):
        self.input_hidden_size = config['input_hidden_size']
        self.Rnn_hidden_layer = config['Rnn_hidden_layer']
        self.Rnn_hidden_size = config['Rnn_hidden_size']
        self.Rnn_type = config['Rnn_type']
        self.epoch= config['epoch']
        self.optimize_type = config['optimizer']
        self.lr = config['learning_rate']
        self.batch_size = config['batch_size']
        self.data_file = config['data_file']
        self.test_file = config['test_file']
        self.need_test = config['need_test']
        self.criterion_type = config['criterion']
        self.model_type = config['model']
        self.rparam = config_dict['rparam']
        self.cv_params = config_dict['cv_param']
        self.use_attention = config_dict['use_attention']
        self.use_RNN = config_dict['use_RNN']
        self.task_type = config_dict['task_type']


        #TODO:数据预处理
        start = time.time()
        if self.model_type == 'DiaNet':
            self.am, self.pm = data_iter(self.data_file, self.batch_size, self.task_type)
            self.amdata = self.am[0]
            self.amcenter = self.am[1]
            self.pmdata = self.pm[0]
            self.pmcenter = self.pm[1]
            self.input_size = self.amdata[0][0].shape[1]
        elif self.model_type == 'XGBoost':
            self.xgb_train, self.train_center = MLdata('train.csv')
            #print(self.xgb_train, self.train_center)
            self.xgb_test, self.test_center = MLdata('valid.csv')
        elif self.model_type == 'ClassiNet':
            self.train_dataset = nlp_data_iter(self.data_file, self.batch_size)
            if self.need_test:
                self.test_dataset = nlp_data_iter(self.test_file, 1)
        elif self.model_type == 'Lasso':
            self.lasso_train_x, self.lasso_train_y = MLdata('train.csv')
            self.lasso_test_x, self.lasso_test_y = MLdata('valid.csv')

        end = time.time()
        print('data already!    time:%f'%(end - start))

        #TODO:定义网络
        if self.model_type == 'DiaNet':
            self.net = DiaNet(self.input_hidden_size,
                              self.Rnn_hidden_layer,
                              self.Rnn_hidden_size,
                              self.Rnn_type,
                              self.input_size,
                              self.use_attention,
                              self.use_RNN,
                              self.task_type)

            if self.criterion_type == 'MSE':
                self.criterion = nn.MSELoss()
            elif self.criterion_type == 'L1':
                self.criterion = nn.L1Loss()
            self.class_criterion = nn.CrossEntropyLoss()

            self.test_criterion = nn.MSELoss()

            if self.optimize_type == 'Adam':
                self.optimizer = optim.Adam(self.net.parameters(), lr = self.lr)
                self.sub_optimizer = optim.Adam(self.net.parameters(), lr = self.lr)
            else:
                self.optimizer = optim.SGD(self.net.parameters(), lr = self.lr)
                self.sub_optimizer = optim.Adam(self.net.parameters(), lr=self.lr / 10)

            if self.need_test:
                self.test_am, self.test_pm = data_iter(self.test_file, 1)
                self.test_amdata = self.test_am[0]
                self.test_amcenter = self.test_am[1]
                self.test_pmdata = self.test_pm[0]
                self.test_pmcenter = self.test_pm[1]

            self.writer = SummaryWriter(comment = 'DiaNet')
        elif self.model_type == 'ClassiNet':
            self.net = ClassiNet(self.input_hidden_size, self.Rnn_hidden_size, self.Rnn_hidden_layer)
            self.criterion = nn.CrossEntropyLoss()
            if self.optimize_type == 'Adam':
                self.optimizer = optim.Adam(self.net.parameters(), lr = self.lr)
                self.sub_optimizer = optim.Adam(self.net.parameters(), lr = self.lr)
            else:
                self.optimizer = optim.SGD(self.net.parameters(), lr = self.lr)
                self.sub_optimizer = optim.Adam(self.net.parameters(), lr=self.lr / 10)



    def Diatrain(self):
        flag = True
        for e in range(self.epoch):
            self.net.hidden_init(self.batch_size)
            start = time.time()
            loss_list = []

            self.net.set_time('am')
            for index, batch in enumerate(self.amdata):
                #print(batch)
                self.optimizer.zero_grad()
                center = self.amcenter[index]
                output = self.net(batch)
                #print(output)
                #print(output, center)
                if self.task_type == 'C':
                    loss = self.class_criterion(output, center.long())
                else:
                    output = output.view_as(center)
                    loss = self.criterion(output, center)
                    loss = torch.sqrt(loss)
                loss_list.append(loss)
                loss.backward()
                if flag:
                    self.optimizer.step()
                else:
                    self.sub_optimizer.step()
                self.writer.add_scalar('Train', loss, e)

            self.net.set_time('pm')
            for index, batch in enumerate(self.pmdata):
                self.optimizer.zero_grad()
                center = self.pmcenter[index]
                output = self.net(batch)
                #print(output, center)
                if self.task_type == 'C':
                    loss = self.class_criterion(output, center.long())
                else:
                    output = output.view_as(center)
                    loss = self.criterion(output, center)
                    loss = torch.sqrt(loss)
                loss_list.append(loss)
                loss.backward()
                if flag:
                    self.optimizer.step()
                else:
                    self.sub_optimizer.step()
                self.writer.add_scalar('Train', loss, e)


            loss_list = torch.Tensor(loss_list)
            torch.save(self.net, 'param/param_%d'%(e + 1))
            end = time.time()
            print_loss = loss_list.mean()
            print('epoch: %d    loss: %f    time: %f'%(e + 1, print_loss, end - start))

            if print_loss < 0.0016:
                flag = False

            #if self.need_test and print_loss < 0.14:
            #    self.Diatest()
            if self.need_test:
                self.Diaclass_test()
        self.writer.add_graph(self.net, (self.amdata[0],))
        self.writer.add_graph(self.net, (self.pmdata[0],))

    def Diatest(self):
        start = time.time()
        output_list = []
        center_list = []

        # print(self.test_amdata)
        self.net.set_time('am')
        for index, batch in enumerate(self.test_amdata):
            # continue
            center = self.test_amcenter[index]
            output = self.net(batch)
            output_list.append(output.view_as(center))
            # output_list.append(batch[-1].view_as(center))
            center_list.append(center)

        self.net.set_time('pm')
        for index, batch in enumerate(self.test_pmdata):
            # continue
            center = self.test_pmcenter[index]
            output = self.net(batch)
            output_list.append(output.view_as(center))
            # output_list.append(batch[-1].view_as(center))
            center_list.append(center)
        # print(output_list)
        # print(output_list)
        op = torch.stack(output_list)
        ct = torch.stack(center_list)

        # print(op, ct)
        loss = self.criterion(op, ct)
        end = time.time()
        print('loss: %f    time: %f\n' % (torch.sqrt(loss), end - start))

    def Diaclass_test(self):
        start = time.time()
        output_list = []
        center_list = []

        # print(self.test_amdata)
        self.net.set_time('am')
        for index, batch in enumerate(self.test_amdata):
            # continue
            center = self.test_amcenter[index]
            output = self.net(batch)
            output_list.append(output)
            # output_list.append(batch[-1].view_as(center))
            center_list.append(center)

        self.net.set_time('pm')
        for index, batch in enumerate(self.test_pmdata):
            # continue
            center = self.test_pmcenter[index]
            output = self.net(batch)
            output_list.append(output)
            # output_list.append(batch[-1].view_as(center))
            center_list.append(center)
        # print(output_list)

        # print(output_list)

        op = torch.stack(output_list)
        ct = torch.stack(center_list).view(-1)

        if self.task_type == 'C':
            # print(op, ct)

            op = torch.max(op.view(-1, 2), dim=1)[1]

            # for i in range(ct.shape[0]):
            #    print(op[i], ct[i])
            loss = (op.float() == ct.float()).float()
            loss = loss.mean()
        else:
            op = op.view(-1)
            loss = self.criterion(op, ct)
            loss = loss.mean()
            loss = torch.sqrt(loss)
            # print(op, ct)

        end = time.time()
        # print('loss: %f    time: %f\n' % (torch.sqrt(loss), end - start))
        print('loss: %f    time: %f\n' % (loss, end - start))

    def Diagenerate(self, param):
        self.net = torch.load('param/param_%d' % param)
        am_data, pm_data = data_iter('new_testset.csv', 1, self.model_type, genrate=True)
        amdata = am_data[0]
        amindex = am_data[1]
        pmdata = pm_data[0]
        pmindex = pm_data[1]
        csv_out = open('prediction.csv', 'w', newline='')
        csv_write = csv.writer(csv_out, dialect='excel')
        write_list = {}
        self.net.set_time('am')
        for i, data in enumerate(amdata):
            pred = self.net(data)
            if amindex[i][0][0] in [478198, 473348, 468498, 463648, 453998, 449148, 439548, 433748]:
                write_list[amindex[i][0][0]] = amindex[i][0][1]
            # pred += amindex[i][0][1]
            else:
                if self.model_type == 'C':
                    pred = torch.max(pred.view(-1, 2), dim=1)[1]
                    write_list[amindex[i][0][0]] = (2 * int(pred) - 1) * 0.00028 + amindex[i][0][1]
                else:
                    write_list[amindex[i][0][0]] = pred / 1000 + amindex[i][0][1]
            # print(amindex[i][0][1])
            # sig = 1 if amindex[i][0][1] >= 0 else -1
            # write_list[amindex[i][0][0]] = abs(amindex[i][0][1]) + sig * torch.rand(1) / 50000

        self.net.set_time('pm')
        for i, data in enumerate(pmdata):
            pred = self.net(data)
            if pmindex[i][0][0] in [478198, 473348, 468498, 463648, 453998, 449148, 439548, 433748]:
                write_list[pmindex[i][0][0]] = pmindex[i][0][1]
            # pred += pmindex[i][0][1]
            else:
                if self.model_type == 'C':
                    pred = torch.max(pred.view(-1, 2), dim=1)[1]
                    write_list[pmindex[i][0][0]] = (2 * int(pred) - 1) * 0.00028 + pmindex[i][0][1]
                else:
                    write_list[pmindex[i][0][0]] = pred / 1000 + pmindex[i][0][1]
            # write_list[pmindex[i][0][0]] = float(pmindex[i][0][1])
            # print(pmindex[i][0][1])
            # sig = 1 if pmindex[i][0][1] >= 0 else -1
            # write_list[pmindex[i][0][0]] = abs(pmindex[i][0][1]) + sig * pmindex[i][0][1] * torch.rand(1) / 50000

        key_list = sorted(write_list.keys())
        print(len(key_list))
        count = 1
        csv_write.writerow(['caseid', 'midprice'])
        for k in key_list:
            if count < 143:
                count += 1
                continue
            csv_write.writerow([count, float(write_list[k])])
            count += 1

    def xgb(self):
        bst = xgboost.XGBRegressor(**self.rparam)
        bst.fit(self.xgb_train, self.train_center, eval_set=[(self.xgb_test, self.test_center)],
                early_stopping_rounds=30, eval_metric='rmse', verbose=True)#1000

        gen, center = MLgenerate('new_testset.csv')
        #print(gen.shape, center.shape)
        pred = bst.predict(gen, ntree_limit=bst.best_ntree_limit)
        #print(pred)


        #csv_out = open('prediction.csv', 'w', newline='')
        #csv_write = csv.writer(csv_out, dialect='excel')
        write_list = {}


        for i, data in enumerate(center):
            if data[0] in [478198, 473348, 468498, 463648, 453998, 449148, 439548, 433748]:
                pred[i] = 0
            write_list[data[0]] = pred[i] / 1000 + data[1]
            print(pred[i], data[1], pred[i] / 1000 + data[1])

        return
        key_list = sorted(write_list.keys())
        print(len(key_list))
        count = 1
        #csv_write.writerow(['caseid', 'midprice'])
        for k in key_list:
            if count < 143:
                count += 1
                continue
            #csv_write.writerow([count, float(write_list[k])])
            count += 1


    def rxgb(self):
        #print(self.xgb_train)
        #print(self.train_center)
        model = xgboost.XGBRegressor(**self.rparam)
        optimized_GBM = GridSearchCV(estimator=model, param_grid=self.cv_params, scoring='neg_mean_squared_error', cv=5, verbose=1, n_jobs=4)
        optimized_GBM.fit(self.xgb_train, self.train_center)
        evalute_result = optimized_GBM.cv_results_
        print('每轮迭代运行结果:{0}'.format(evalute_result['mean_test_score']))
        print('每轮迭代运行结果:{0}'.format(evalute_result['params']))
        print('参数的最佳取值：{0}'.format(optimized_GBM.best_params_))
        print('最佳模型得分:{0}'.format(math.sqrt(abs(optimized_GBM.best_score_))))
        model.fit(self.xgb_train, self.train_center, eval_set=[(self.xgb_test, self.test_center)],
                early_stopping_rounds=10, eval_metric='rmse', verbose=True)

    def cxgb(self):
        model = xgboost.XGBClassifier(**self.rparam)
        optimized_GBM = GridSearchCV(estimator=model, param_grid=self.cv_params, scoring='accuracy', cv=5,
                                     verbose=1, n_jobs=4)
        optimized_GBM.fit(self.xgb_train, self.train_center)
        evalute_result = optimized_GBM.cv_results_
        print('每轮迭代运行结果:{0}'.format(evalute_result['mean_test_score']))
        print('每轮迭代运行结果:{0}'.format(evalute_result['params']))
        print('参数的最佳取值：{0}'.format(optimized_GBM.best_params_))
        print('最佳模型得分:{0}'.format(optimized_GBM.best_score_))
        model.fit(self.xgb_train, self.train_center, eval_set=[(self.xgb_test, self.test_center)],
                  early_stopping_rounds=10, eval_metric='error', verbose=True)

    def lasso(self):
        #model = Lasso(normalize=True, alpha=0.01)
        model = LassoCV(cv=5,normalize=True,alphas=np.logspace(-3, 2, 50), max_iter=100)
        model.fit(self.lasso_train_x, self.lasso_train_y)
        print('系数矩阵:\n', model.coef_)
        print('线性回归模型:\n', model)
        predicted = model.predict(self.lasso_test_x)
        loss = mean_squared_error(self.lasso_test_y, predicted)
        print('loss:{0}'.format(math.sqrt(loss)))



    def Clatrain(self):
        for e in range(self.epoch):
            start = time.time()
            loss_list = []
            for batch in self.train_dataset:
                self.optimizer.zero_grad()
                data = batch[0]
                label = batch[1].long()
                output = self.net(data)
                loss = self.criterion(output, label)
                loss_list.append(loss)
                loss.backward()
                self.optimizer.step()
            loss_list = torch.Tensor(loss_list)
            torch.save(self.net, 'Claparam/param_%d' % (e + 1))
            end = time.time()
            print_loss = loss_list.mean()
            print('epoch: %d    loss: %f    time: %f' % (e + 1, print_loss, end - start))
            if self.need_test:
                self.Clatest()
    def Clatest(self):
        start = time.time()
        output_list = []
        label_list = []

        for index, batch in enumerate(self.test_dataset):
            data = batch[0]
            label = batch[1]
            output = self.net(data)
            output_list.append(output)
            label_list.append(label)

        op = torch.stack(output_list)
        ct = torch.stack(label_list).view(-1)

        op = torch.max(op.view(-1, 2), dim=1)[1]

        loss = (op.float() == ct.float()).float()
        loss = loss.mean()
        end = time.time()
        print('loss: %f    time: %f\n' % (loss, end - start))


    def train(self):
        if self.model_type == 'DiaNet':
            self.Diatrain()
        elif self.model_type == 'XGBoost':
            self.cxgb()
        elif self.model_type == 'ClassiNet':
            self.Clatrain()
        elif self.model_type == 'Lasso':
            self.lasso()




if __name__ == '__main__':
    torch.manual_seed(1)
    trainer = Trainer(config_dict)
    #trainer.xgb()
    #trainer.train()
    trainer.Diagenerate(76)