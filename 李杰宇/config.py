#-*- coding:utf-8 -*-
config_dict = {'input_hidden_size'  :   64,#64
               'Rnn_hidden_layer'   :   2,
               'Rnn_hidden_size'    :   64,#64
               'Rnn_type'           :   'LSTM',
               'epoch'              :   100,
               'optimizer'          :   'Adam',
               'learning_rate'      :   0.0001,
               'batch_size'         :   32,
               'data_file'          :   'train.csv',
               'test_file'          :   'valid.csv',
               'need_test'          :   True,
               'criterion'          :   'MSE',
               'model'              :   'DiaNet',#DiaNet, ClassiNet, XGBoost, Lasso
               'task_type'         :   'R',#R:regression,C:Classify
               'use_attention'      :   False,
               'use_RNN'            :   True,
               'rparam'             :   {'learning_rate'        : 0.1,
                                         'n_estimators'         : 1000,
                                         'max_depth'            : 10,
                                         'min_child_weight'     : 3,
                                         'seed'                 : 0,
                                         'subsample'            : 0.8,
                                         'colsample_bytree'     : 0.8,
                                         'gamma'                : 0,
                                         #'reg_alpha'            : 500,
                                         'objective'            : 'binary:logistic',
                                         'eval_metric'          : 'error'},
                'cv_param'          :   {'seed':[0]}
               }