import numpy as np
import tensorflow as tf
import tensorflow.contrib.rnn as rnn
import matplotlib.pyplot as plt
import pandas as pd
import os
import csv

class CSVReader:
    dataitems = ["MidPrice", "LastPrice", "Volume", "BidPrice1", "BidVolume1", "AskPrice1", "AskVolume1"]
    def __init__(self, training_set="./train_data.csv", testing_set="./test_data.csv"):
        self.Train = pd.read_csv(training_set,
                                 index_col="Date",
                                 usecols=[
                                     "Date", "Time",
                                     "MidPrice", "LastPrice",
                                     "Volume", "BidPrice1",
                                     "BidVolume1", "AskPrice1",
                                     "AskVolume1"
                                 ])

        self.Test = pd.read_csv(testing_set,
                                index_col="Date",
                                usecols=[
                                    "Date", "Time",
                                    "MidPrice", "LastPrice",
                                    "Volume", "BidPrice1",
                                    "BidVolume1", "AskPrice1",
                                    "AskVolume1"
                                ])


        self.Train = self.Train.sort_index()
        self.Test = self.Test.sort_index()
        def hour(s):
            q = [float(i) for i in s.split(":")]
            return q[0] + q[1] / 60

        self.Train["Hour"] = self.Train["Time"].map(hour)

        self.Test["Hour"] = self.Test["Time"].map(hour)

        TimeStampCount = self.Train["Time"].groupby("Date").count()
        TimeStampCount = TimeStampCount.sort_values()

        self.TrainDates = self.Train.index.unique().tolist()
        self.DangerousDates = TimeStampCount[TimeStampCount > 5000].index.tolist()
        self.FilteredDates = [date for date in self.TrainDates if date not in self.DangerousDates]

        self.TrainSet = {}

        self.AM = self.Train[self.Train["Hour"] < 11.70]
        self.PM = self.Train[self.Train["Hour"] > 12.70]

        for date in self.FilteredDates:
            self.TrainSet[f"{date}|AM"] = self.AM.loc[date]
            self.TrainSet[f"{date}|PM"] = self.PM.loc[date]

        # Splitting Testing Set
        self.TestingSet = []

        for begin in range(0, len(self.Test), 10):
            self.TestingSet.append(self.Test.iloc[begin: begin + 10])

    def training_dates(self):
        """
        Returning all training set dates
        """
        return self.FilteredDates

    def get_training_numpy(self, idx):
        """
        Returning training set at idx, also the am / pm / date infomation trailing it.
        [T, Feature]
        """
        key = list(self.TrainSet.keys())[idx]
        pandadb = self.TrainSet[key][self.dataitems]
        return key, pandadb.values

    def get_testing_numpy(self, idx):
        """
        Returning testing set at idx, also the am / pm / date infomation trailing it.
        [T, Feature]
        """
        pandadb = self.TestingSet[idx][self.dataitems]
        return pandadb.values

    def training_count(self):
        """
        Returning the count of all available sub training set, including morning and afternoon
        """
        return len(self.TrainSet)

    def testing_count(self):
        """
        Returning the count of all testing instance
        """
        return len(self.TestingSet)


TIME_STEPS=10
BATCH_SIZE=32
HIDDEN_UNITS=64
LEARNING_RATE=0.0006
EPOCH=10
input_size = 4
output_size = 1


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
		tmp=tm-x[time_step-1][0]
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



def main(time_step=10):
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
			test_predict[i][0]+=test_x[i][time_step-1][0]

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