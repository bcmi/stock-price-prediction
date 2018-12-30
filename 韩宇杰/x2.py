import numpy as np
import tensorflow as tf
import tensorflow.contrib.rnn as rnn
import matplotlib.pyplot as plt
import pandas as pd
import os
import csv

TIME_STEPS=10
BATCH_SIZE=32
HIDDEN_UNITS=64
LEARNING_RATE=0.0006
EPOCH=10
input_size = 4
output_size = 1

TRAIN_EXAMPLES=11000
TEST_EXAMPLES=1100
TRAIN_PATH = './train_data.csv'
TEST_PATH  = './test_data.csv'

def read_data(path):
	f=open(path)
	df=pd.read_csv(f)
	data=df.iloc[:,3:10].values
	datas=[]
	p=0

	tmp = data[0][2]
	p=0
	_,grad = np.gradient(data)
	for i in range(len(data)):

		data[i][2] = grad[i][2]
	
	data=np.delete(data,5, axis=1)
	data=np.delete(data,3, axis=1)
	data=np.delete(data,1, axis=1)			

	return data

data = read_data(TRAIN_PATH)
tdata = read_data(TEST_PATH)

def get_train_data(batch_size=60,time_step=10,train_begin=15000,train_end=35000):
	batch_index=[]
	data_train=[]

	train_x,train_y=[],[]

	data_train = data[train_begin:train_end+20]#

	normalized_train_data= (data_train-np.mean(data_train,axis=0))/np.std(data_train,axis=0)

	length = (len(data_train)-time_step-20)
	for i in range(length):#
		
		if i % batch_size==0:
			batch_index.append(i)
					

		p=np.random.randint(0,length)

		x=normalized_train_data[p:p+time_step]
		

		y=normalized_train_data[p+time_step:p+time_step+20,0:1]#
		train_x.append(x.tolist())
		tm=np.mean(y)
		tmp=tm
		train_y.append([tmp])

		
	batch_index.append(length)#
	return batch_index,train_x,train_y

def get_test_data(time_step=10,test_begin=350000,test_end=351200):

	data_test=tdata

	mean = np.mean(data_test,axis=0)
	std=np.std(data_test,axis=0)

	normalized_test_data=(data_test-mean)/std
	test_x,test_y=[],[]
	for i in range(len(normalized_test_data)):#

		if i%10>0:#
			continue	
		x=normalized_test_data[i:i+time_step]

		test_x.append(x.tolist())

	return mean,std,test_x,test_y

weights={
         'in':tf.Variable(tf.random_normal([input_size,HIDDEN_UNITS])),
         'out':tf.Variable(tf.random_normal([HIDDEN_UNITS,output_size])),
	 'single':tf.Variable(tf.random_normal([output_size,1]))
         }
biases={
        'in':tf.Variable(tf.constant(0.1,shape=[HIDDEN_UNITS,])),
        'out':tf.Variable(tf.constant(0.1,shape=[output_size,])),
	'single':tf.Variable(tf.constant(0.1,shape=[output_size,1]))
        }

def lstm(X):
	batch_size=tf.shape(X)[0]
	time_step=tf.shape(X)[1]

	cell1=tf.contrib.rnn.BasicLSTMCell(HIDDEN_UNITS)
	cell2=tf.contrib.rnn.BasicLSTMCell(HIDDEN_UNITS)
	cell3=tf.contrib.rnn.BasicLSTMCell(HIDDEN_UNITS)
	cell = tf.nn.rnn_cell.MultiRNNCell(cells=[cell1,cell2,cell3])

	init_state=cell.zero_state(batch_size,dtype=tf.float32)
	with tf.variable_scope('scope',reuse=tf.AUTO_REUSE):
		output_rnn,final_states=tf.nn.dynamic_rnn(cell,X,initial_state=init_state, dtype=tf.float32)  
	output=output_rnn[:,-1,:]

	w_out=weights['out']
	b_out=biases['out']
	pred=tf.matmul(output,w_out)+b_out

	return pred,final_states

def train_lstm(batch_size=60,time_step=10,train_begin=0,train_end=430000):
	X=tf.placeholder(tf.float32, shape=[None,time_step,input_size])

	Y=tf.placeholder(tf.float32, shape=[None,output_size])

	batch_index,train_x,train_y = get_train_data(batch_size,time_step,train_begin,train_end)	
	pred,_ = lstm(X)

	loss=tf.losses.mean_squared_error(Y,pred)	

	train_op=tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)
	saver = tf.train.Saver(tf.global_variables(),max_to_keep = 4)
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		for i in range(EPOCH):
			for step in range(len(batch_index)-1):
				_,loss_=sess.run([train_op,loss],feed_dict={X:train_x[batch_index[step]:batch_index[step+1]],Y:train_y[batch_index[step]:batch_index[step+1]]})

			print('iter:',i,'loss:',loss_)
			print("saved:", saver.save(sess,os.getcwd()+"/save.sav",global_step=i))



def prediction(time_step=10):
	X=tf.placeholder(tf.float32, shape=[None,time_step,input_size])
	Y=tf.placeholder(tf.float32, shape=[None,output_size])

	mean,std,test_x,test_y=get_test_data(time_step)
	pred,_=lstm(X)
	loss=tf.losses.mean_squared_error(Y,pred)
	saver=tf.train.Saver(tf.global_variables())
	with tf.Session() as sess:
		module_file = tf.train.latest_checkpoint(os.getcwd())
		saver.restore(sess,module_file)
		test_predict=[]
		losssum=0
		for step in range(len(test_x)):
			prob=sess.run(pred,feed_dict={X:[test_x[step]]})

			predict=prob.reshape(-1,1)

			test_predict.extend(predict)

		predict_test=[]

		for i in range(len(test_predict)):
			predict_test.append(test_predict[i])

		y_test=[]
		for i in range(len(test_y)):

			y_test.append(test_y[i])
		test_y=np.array(y_test)*std[0]+mean[0]

		test_predict=np.array(predict_test)*std[0]+mean[0]

		cf = open('out.csv', 'w')
		fn =["caseid","midprice"]
		writer=csv.DictWriter(cf, fieldnames=fn)
		writer.writeheader()
		for i in range(len(test_predict)):
			if(i<142):
				continue
			writer.writerow({'caseid':str(i+1),'midprice':float(test_predict[i][0])})
		cf.close()
train_lstm(batch_size=BATCH_SIZE,time_step=TIME_STEPS)
prediction()

