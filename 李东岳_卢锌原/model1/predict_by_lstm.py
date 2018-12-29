import pandas as pd
import numpy as np
import tensorflow as tf
import os
import  random
from  time import strftime ,localtime
reg=0.00001
learning_rate=0.0006
EPOCHES=50
batch_size=32
rnn_unit=10
lstm_layers=1
nn_layers1=2
nn_layers2=2
nn_unit1=10
nn_unit2=10
input_size=5
output_size=1
time_step=10

train_x=[]
train_y=[]
test_x=[]
test_y=[]
infer_x=[]
infer_y=[]
ten_y=[]
features=[]
train_feature=[]
close=[]
infer_feature=[]
infer_close=[]

f=open('train_data.csv')
df=pd.read_csv(f)
data=df.iloc[:,[3,4,6,7,8,9]].values
data_y=[]
for i in range(data.shape[0]-20):
    y=np.mean(data[i+1:i+21,0])-data[i,0]
    data_y.append(y)
data_y=np.array(data_y)
data_y=np.reshape(data_y,[-1,1])

f2=open('test_data.csv')
df2=pd.read_csv(f2)
infer=df2.iloc[:,[3,4,6,7,8,9]].values

def cmp(x):
    return x[0]

def b_find(num):
    start = 0
    end = len(train_feature) - 1
    while start <= end:
        mid = int((start + end) / 2)
        # print(train_feature[mid])
        # print(mid,train_feature[mid][0])
        # exit()
        if train_feature[mid][0] < num:
            start = mid + 1
        else:
            end = mid - 1
    return end


def find(index):
    for i in range(len(train_feature)):
        if i == index:
            continue
        tmp = abs(train_feature[i] - train_feature[index])
        if tmp < minn:
            minn = tmp
            result = i
    return i

def get_train_data(train_begin=30000,train_end=430000,time_step=10,gap=5):
    data_train=data[train_begin:train_end,1:]
    mid_prices_train=data_y[train_begin:train_end]
    for i in range(len(data_train)):
        a=(data_train[i][1]/data_train[i][3])
        features.append(a)
    index=0
    for i in range(0,train_end-train_begin-time_step+1,gap):
        train_feature.append((np.mean(features[i:i + time_step]), index))
        index += 1
        vol_mean=np.mean(data_train[i:i + time_step,[2,4]])
        vol_std=np.std(data_train[i:i + time_step,[2,4]])
        price_mean = np.mean(data_train[i:i + time_step, [ 0,1,3]])
        price_std = np.std(data_train[i:i + time_step, [0,1,3]])
        mean=np.array([price_mean,price_mean,vol_mean,price_mean,vol_mean])
        std=np.array([price_std,price_std,vol_std,price_std,vol_std])
        epsilon=1e-12
        x=(data_train[i:i + time_step]-mean)/(epsilon+std)
        train_x.append(x.tolist())
        train_y.append(mid_prices_train[i:i+time_step].tolist())
    train_feature.sort(key=cmp)
    # print (train_feature)
    # exit()
    l = len(train_feature)
    for i in range(l):
        close.append(0)
    for i in range(len(train_feature)):
        former = train_feature[(i - 1 + l) % l][0]
        latter = train_feature[(i + 1) % l][0]
        if (abs(former - train_feature[i][0]) < abs(latter - train_feature[i][0])):
            close[train_feature[i][1]] = train_feature[(i - 1 + l) % l][1]
        else:
            close[train_feature[i][1]] = train_feature[(i + 1) % l][1]
    print(train_x[0])

def get_test_data(test_begin=0,test_end=30000,time_step=10,gap=30):
    data_test = data[test_begin:test_end, 1:]
    mid_prices_test = data_y[test_begin:test_end]
    for i in range(0, test_end -test_begin - time_step + 1,gap):
        vol_mean = np.mean(data_test[i:i + time_step, [ 2, 4]])
        vol_std = np.std(data_test[i:i + time_step, [2, 4]])
        price_mean = np.mean(data_test[i:i + time_step, [0,1, 3]])
        price_std = np.std(data_test[i:i + time_step, [0,1, 3]])
        mean = np.array([price_mean,price_mean,vol_mean,price_mean,vol_mean])
        std = np.array([price_std,price_std,vol_std,price_std,vol_std])
        epsilon = 1e-12
        x = (data_test[i:i + time_step] - mean) / (epsilon + std)
        test_x.append(x.tolist())
        test_y.append(mid_prices_test[i:i+time_step].tolist())
    print(test_x[0])
def get_infer_data(time_step=10):
    data_infer=infer[:,1:]
    for i in range(len(infer)):
        infer_feature.append(infer[i][2]/infer[i][4])
    for i in range(0,len(infer)-time_step+1,time_step):

        vol_mean = np.mean(data_infer[i:i + time_step, [ 2, 4]])
        vol_std = np.std(data_infer[i:i + time_step, [ 2, 4]])
        price_mean = np.mean(data_infer[i:i + time_step, [0,1, 3]])
        price_std = np.std(data_infer[i:i + time_step, [0,1, 3]])
        mean = np.array([price_mean,price_mean,vol_mean,price_mean,vol_mean])
        std = np.array([price_std,price_std,vol_std,price_std,vol_std])
        epsilon = 1e-12
        x = (data_infer[i:i + time_step] - mean) / (epsilon + std)
        infer_x.append(x.tolist())
        infer_close.append(train_feature[b_find(float(np.mean(np.array(infer_feature[i:i + time_step]))))][1])
    for i in range(0,len(infer),10):
        ten_y.append(infer[i+9,0])
    print(infer_x[0])

def write_result(filename):
    case=[]
    for i in range(143,1001):
        case.append(i)

    result=np.array(infer_y)[142:]+ np.array(ten_y)[142:]
    print(result)
    Data={'caseid':case,'midprice':result}
    save=pd.DataFrame(Data, columns=["caseid","midprice"])
    save.to_csv(filename+'.csv', index=False, encoding="utf-8")

class LSTM_HFT:
    def __init__(self):
        self.X1 = tf.placeholder(shape=[None, time_step, input_size], dtype=tf.float32)
        self.X2 = tf.placeholder(shape=[None, time_step, input_size], dtype=tf.float32)
        self.Y = tf.placeholder(shape=[None, time_step, output_size], dtype=tf.float32)
        with tf.variable_scope('main'):
            cell1 = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.LSTMCell(rnn_unit) for i in range(lstm_layers)])
            #init_state=cells.zero_state(batch_size,dtype=tf.float32)
            output_rnn1, final_state1 = tf.nn.dynamic_rnn(cell1, self.X1, dtype=tf.float32)
        with tf.variable_scope('second'):
            cell2 = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.LSTMCell(rnn_unit) for i in range(lstm_layers)])
            output_rnn2, final_state2 = tf.nn.dynamic_rnn(cell2, self.X2, dtype=tf.float32)
            #self.pred =tf.reshape(self.construct_nn_layers(self.final_state[-1][-1],output_size),[-1,])
        new_input=tf.concat([output_rnn1,output_rnn2],2)
        with tf.variable_scope('merge'):
            cell3 = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.LSTMCell(rnn_unit) for i in range(lstm_layers)])
            output_rnn, final_state = tf.nn.dynamic_rnn(cell3,new_input , dtype=tf.float32)
            output = tf.reshape(output_rnn, [-1, rnn_unit])
            w_out =tf.Variable(tf.random_normal([rnn_unit,1]))
            b_out =tf.Variable(tf.constant(0.1,shape=[1,]))
            self.pred=tf.matmul(output, w_out) + b_out
        #loss
        tv = tf.trainable_variables()
        self.loss = tf.reduce_mean(tf.square(tf.reshape(self.pred, [-1]) - tf.reshape(self.Y, [-1])))  # +reg*tf.reduce_sum([ tf.nn.l2_loss(v) for v in tv ])
        self.opt = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.loss)

    def construct_first_nn_layers(self,input,output_size):
        first_nn_layer = tf.layers.dense(inputs=input,units=nn_unit1,activation=tf.nn.relu)
        dense = first_nn_layer
        for i in range(nn_layers1 - 2):
            dense = tf.layers.dense(inputs=dense, units=nn_unit1, activation=tf.nn.relu)
        last_nn_layer = tf.layers.dense(inputs=dense, units=output_size)
        return last_nn_layer

    def construct_second_nn_layers(self,input,output_size):
        first_nn_layer = tf.layers.dense(inputs=input,units=nn_unit2,activation=tf.nn.selu)
        dense = first_nn_layer
        for i in range(nn_layers2 - 2):
            dense = tf.layers.dense(inputs=dense, units=nn_unit2, activation=tf.nn.selu)
        last_nn_layer = tf.layers.dense(inputs=dense, units=output_size)
        return last_nn_layer

def test_model(sess,lstm):
    test_size = len(test_x)
    pred_y=[]
    for index in range(0,test_size-batch_size+1,batch_size):
        pred = sess.run(lstm.pred, feed_dict={lstm.X:  test_x[index:index+batch_size]})
        pred_y.extend(pred.tolist())
    return np.sqrt(np.mean(np.square(np.array(pred_y)-np.array(test_y)[:,-1,0])))

def train_model(sess,lstm,ckpt_dir,restore=False,restore_dir=None):
    print("train start")
    """if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    if restore:
        ckpt = tf.train.get_checkpoint_state(restore_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess,ckpt.model_checkpoint_path)"""

    tmp=[]
    n_batch=len(train_x)//batch_size
    for i in range(n_batch):
        tmp.append(i*batch_size)
    for epoch in range(1,EPOCHES+1):
        train_size=len(train_x)
        loss_sum=0
        random.shuffle(tmp)
        inde=0
        for index in tmp:
            if inde%100==0:
                print(inde)
            inde+=1
            second = []
            batch_x=train_x[index:index+batch_size]
            for k in range(len(batch_x)):
                second.append(train_x[close[index + k]])
            loss,_=sess.run([lstm.loss,lstm.opt],feed_dict={lstm.X1:batch_x,lstm.X2:second, lstm.Y:train_y[index:index+batch_size]})
            loss_sum+=loss

        print("epoch:{0} \t loss :{1}".format(epoch,(loss_sum/n_batch)))
        #save model every 10 epoches
        if epoch % 2 == 0:
            saver.save(sess, ckpt_dir+"checkpoint", global_step=epoch)
        if epoch % 2 == 0:
            #print("epoch:{0}, MSE:{1}".format(epoch,test_model(sess,lstm)))
            predict(sess,lstm)
            write_result("epoch{0}".format(epoch))

def predict(sess,lstm,restore_dir=None):
    if restore_dir:
        ckpt = tf.train.get_checkpoint_state(restore_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
    infer_size = len(infer_x)
    global infer_y
    infer_y=[]
    for case in range(infer_size):
        batch_x=[infer_x[case]]
        second_x=[]
        for k in range(len(batch_x)):
            second_x.append(train_x[infer_close[k+case]])
        pred = sess.run(lstm.pred, feed_dict={lstm.X1: batch_x,lstm.X2:second_x})
        infer_y.append(pred[-1][-1])



time_stamp = strftime('%Y_%m_%d_%H_%M_%S', localtime())
get_train_data()
get_test_data()
get_infer_data()
sess=tf.Session()
lstm=LSTM_HFT()
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()

train_model(sess,lstm,"./"+time_stamp+"/")