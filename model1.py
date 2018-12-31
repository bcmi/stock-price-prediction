import pandas as pd
import numpy as np
import tensorflow as tf
import random

rnn_unit=10       #hidden layer units
input_size=5
output_size=1
lr=0.0006

def get_train_data(batch_size,time_step,train_begin,train_end):
    batch_index=[]
    data_train=data[train_begin:train_end]
    normalized_train_data=(data_train-np.mean(data_train,axis=0))/np.std(data_train,axis=0)  #标准化
    train_x,train_y=[],[]
    for i in range(len(normalized_train_data)-time_step):
       if i % batch_size==0:
           batch_index.append(i)
       x=normalized_train_data[i:i+time_step,1:6]
       y=normalized_train_data[i:i+time_step,0,np.newaxis]
       train_x.append(x.tolist())
       train_y.append(y.tolist())
    batch_index.append((len(normalized_train_data)-time_step))
    return batch_index,train_x,train_y

def get_test_data(time_step=10):
    data_test=testdata
    len0=len(data_test)
    mean=np.mean(data_test,axis=0)
    std=np.std(data_test,axis=0)
    normalized_test_data=(data_test-mean)/std  #标准化
    size=len0//time_step
    test_x=[]
    for i in range(size):
       x=normalized_test_data[i*time_step:(i+1)*time_step,1:6]
       test_x.append(x.tolist())
    return mean,std,test_x



weights={
         'in':tf.Variable(tf.random_normal([input_size,rnn_unit])),
         'out':tf.Variable(tf.random_normal([rnn_unit,1]))
        }
biases={
        'in':tf.Variable(tf.constant(0.1,shape=[rnn_unit,])),
        'out':tf.Variable(tf.constant(0.1,shape=[1,]))
       }


def lstm(X):
    batch_size=tf.shape(X)[0]
    time_step=tf.shape(X)[1]
    w_in=weights['in']
    b_in=biases['in']
    input=tf.reshape(X,[-1,input_size])  #2维
    input_rnn=tf.matmul(input,w_in)+b_in
    input_rnn=tf.reshape(input_rnn,[-1,time_step,rnn_unit])  #3维
    cell=tf.nn.rnn_cell.BasicLSTMCell(rnn_unit)
    init_state=cell.zero_state(batch_size,dtype=tf.float32)
    output_rnn,final_states=tf.nn.dynamic_rnn(cell, input_rnn,initial_state=init_state, dtype=tf.float32)  #output_rnn是记录lstm每个输出节点的结果，final_states是最后一个cell的结果
    output=tf.reshape(output_rnn,[-1,rnn_unit]) #作为输出层的输入
    w_out=weights['out']
    b_out=biases['out']
    pred=tf.matmul(output,w_out)+b_out
    return pred,final_states


def train_test(batch_size=10,time_step=10,train_begin=0,train_end=400000):
    X=tf.placeholder(tf.float32, shape=[None,time_step,input_size])
    Y=tf.placeholder(tf.float32, shape=[None,time_step,output_size])
    batch_index,train_x,train_y=get_train_data(batch_size,time_step,train_begin,train_end)
    pred,_=lstm(X)
    loss=tf.reduce_mean(tf.square(tf.reshape(pred,[-1])-tf.reshape(Y, [-1])))
    train_op=tf.train.AdamOptimizer(lr).minimize(loss)
    saver=tf.train.Saver(tf.global_variables(),max_to_keep=15)
    module_file = tf.train.latest_checkpoint('model99/')
    with tf.Session() as sess:
        temp_list=[]
        for step in range(len(batch_index) - 1):
            temp_list.append(step)
        sess.run(tf.global_variables_initializer())
        if module_file!=None:
            saver.restore(sess, module_file)
        for i in range(iter_time):
            random.shuffle(temp_list)
            for temp in temp_list:
                _,loss_=sess.run([train_op,loss],feed_dict={X:train_x[batch_index[temp]:batch_index[temp+1]],Y:train_y[batch_index[temp]:batch_index[temp+1]]})
            print(i,loss_)
            if i % 10==0:
                print("保存模型：",saver.save(sess,'model99/stock.model',global_step=i))
        #预测--------------------------------------------------------------------------------------
        mean, std, test_x = get_test_data(time_step)
        test_predict = []
        for step in range(len(test_x)):
            prob = sess.run(pred, feed_dict={X: [test_x[step]]})
            predict = prob[-1][-1]
            test_predict.append([predict])
        test_predict = np.array(test_predict) * std[0] + mean[0]
        pd.DataFrame(test_predict, columns=['midprice']).to_csv('result.csv')


def get_dataset(dataset, begin, len, step):
    data1, data2 = [], []
    for i in range(begin, begin + len, step):
        data1.append(dataset[i, 1:6])
        data2.append(dataset[i + 1:i + 21, 0])
    data_x = np.array(data1)
    data_y = np.array(data2)
    data_x = data_x.reshape([-1, 5])
    data_y = np.mean(data_y, axis=1)
    data_y = data_y[:, np.newaxis]
    return data_x, data_y


iter_time=100

dataset = pd.read_csv('train_data.csv', usecols=[3, 4, 6, 7, 8,9])
dataset = np.reshape(dataset.values, (len(dataset), 6))
data_xx, data_yy = get_dataset(dataset,0, 420000,1)


new_data=np.hstack((data_yy,data_xx))
data = new_data

testdata=pd.read_csv('test_data.csv', usecols=[3, 4, 6, 7, 8, 9])
testdata=testdata.values

train_test()
