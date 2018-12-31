import tensorflow as tf
import numpy as np
import pandas as pd

def add_layer(inputs,in_size,out_size,activation_function=None):
    Weights=tf.Variable(tf.random_normal([in_size,out_size]))
    biases=tf.Variable(tf.zeros([1,out_size])+0.1)
    Wx_plus_b=tf.matmul(inputs,Weights)+biases
    if activation_function is None:
        outputs=Wx_plus_b
    else:
        outputs=activation_function(Wx_plus_b)
    return outputs

def get_dataset(dataset,begin,len,step):
    data1,data2=[],[]
    for i in range(begin,begin+len,step):
        data1.append(dataset[i:i+10, 0])
        data2.append(dataset[i + 11:i + 31, 0])
    data_x = np.array(data1)
    data_y = np.array(data2)
    data_x=data_x.reshape([-1,10])
    data_y=np.mean(data_y, axis=1)
    data_y=data_y[:,np.newaxis]
    return data_x,data_y

def get_test(test_data):
    data1=[]
    for i in range(0,len(test_data),10):
        data1.append(test_data[i:i+10, 0])
    data_x = np.array(data1)
    data_x=data_x.reshape([-1,10])
    return data_x





dataset = pd.read_csv('train_data.csv', usecols=[3, 4, 6, 7, 8,9])
dataset = np.reshape(dataset.values, (len(dataset), 6))
test_x, test_y = get_dataset(dataset,350000, 80000,1)

test_data=pd.read_csv('test_data.csv', usecols=[3, 4, 6, 7, 8,9])
test_data = np.reshape(test_data.values, (len(test_data), 6))
real_test_x=get_test(test_data)

xs=tf.placeholder(tf.float32,[None,10])
ys=tf.placeholder(tf.float32,[None,1])


l1=add_layer(xs,10,20,activation_function=tf.nn.relu)
l2=add_layer(l1,20,10,activation_function=tf.nn.relu)
l3=add_layer(l2,10,5,activation_function=tf.nn.relu)
prediction=add_layer(l3,5,1,activation_function=None)


loss=tf.reduce_mean(tf.reduce_sum(tf.square(ys-prediction),reduction_indices=[1]))


train_step=tf.train.RMSPropOptimizer(0.01).minimize(loss)

init=tf.global_variables_initializer()



with tf.Session() as sess:
    sess.run(init)
    for i in range(20):
        for j in range(0,30000,400):
            data_x, data_y = get_dataset(dataset,j, 2000,1)
            sess.run(train_step,feed_dict={xs:data_x,ys: data_y})
    re=sess.run(prediction,feed_dict={xs:real_test_x})
    re=np.array(re[142:])
    re=np.reshape(re,-1)
    case=[]
    for i in range(143,1001):
        case.append(i)
    Data={'caseid':case,'midprice':re}
    save=pd.DataFrame(Data, columns=["caseid","midprice"])
    save.to_csv('res2.csv', index=False, encoding="utf-8")



