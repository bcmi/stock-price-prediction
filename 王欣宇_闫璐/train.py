# encoding: utf-8
# file: main.py
# author: shawn233

from __future__ import print_function
import os
import sys
import tensorflow as tf
import numpy as np
import time

BASE_DIR = os.path.dirname (os.path.abspath (sys.argv[0]))
DATA_DIR = os.path.join (BASE_DIR, 'data')
UTIL_DIR = os.path.join (BASE_DIR, 'util')
LOG_DIR = os.path.join (BASE_DIR, 'log')
SAVE_DIR = os.path.join (LOG_DIR, 'save')
MODEL_FILENAME = 'order-book-model'
OUTPUT_FILENAME = 'output.csv'

# add time stamp to output file
time_array = time.localtime (time.time())
time_stamp = str(time_array.tm_mon)+'-'+str(time_array.tm_mday)+'-'+str(time_array.tm_hour)+'-'+str(time_array.tm_min)
OUTPUT_FILENAME_L = OUTPUT_FILENAME.split ('.')
OUTPUT_FILENAME_L[0] = OUTPUT_FILENAME_L[0]+'-'+time_stamp
OUTPUT_FILENAME = '.'.join (OUTPUT_FILENAME_L)
print ('output write to', OUTPUT_FILENAME)

sys.path.append (UTIL_DIR)
from data_util import OrderBook
# import my_tf_util as tf_util




def get_model (inputs, is_training):
    '''
    Get the RNN model

    Args:
    - inputs: tf tensor;
    - is_training: tf bool tensor;

    Returns:
    - pred: prediction;
    '''

    # create RNN cell
    num_units_list = [4, 8]
    cells = [get_cell(num_units) for num_units in num_units_list]
    cell = tf.nn.rnn_cell.MultiRNNCell (cells)
    #num_units = 32
    #cell = get_cell (num_units)

    output_seq, state = tf.nn.dynamic_rnn (cell=cell, inputs=inputs, dtype=tf.float32)
    outputs = tf.reshape (output_seq, shape=[-1, num_units_list[-1] * inputs.get_shape()[1]])
    #outputs = state.h
    #print (outputs.shape)

    # additional fully connected layer
    with tf.variable_scope ('output_layer') as sc:
        weight1 = tf.get_variable ('weight1', shape=[outputs.get_shape()[-1], 16], dtype=tf.float32, initializer=tf.truncated_normal_initializer())
        bias1 = tf.get_variable ('bias1', shape=[16], dtype=tf.float32, initializer=tf.zeros_initializer())

        outputs = tf.matmul (outputs, weight1) + bias1
        #outputs = dropout (outputs, is_training, 'dropout')
        outputs = tf.nn.relu (outputs)

        weight2 = tf.get_variable ('weight2', shape=[outputs.get_shape()[-1], 1], dtype=tf.float32, initializer=tf.truncated_normal_initializer())
        bias2 = tf.get_variable ('bias2', shape=[1], dtype=tf.float32, initializer=tf.zeros_initializer())

        outputs = tf.matmul (outputs, weight2) + bias2
        #outputs = tf.nn.relu(tf.matmul (outputs, weight2) + bias2)

    #print (outputs.shape)
    #input()

    return outputs



def get_dnn_model (inputs,is_training):
    '''
    Get the DNN model

    Args:
    - inputs: tf tensor, batch_size x len x n_features, will be reshaped;
    - is_training: tf bool tensor;

    Returns:
    - pred: tf op, prediction;

    Comment:
    Really terrible performance
    '''
    inputs = tf.reshape (inputs, shape=[-1, inputs.get_shape()[1] * inputs.get_shape()[2]])
    print (inputs.shape)

    #inputs = tf.layers.batch_normalization (inputs, training=is_training)

    with tf.variable_scope ('layer1'):
        weight = tf.get_variable ('weight', shape=[inputs.get_shape()[-1], 128], initializer=tf.contrib.layers.xavier_initializer())
        bias = tf.get_variable ('bias', shape=[128], initializer=tf.zeros_initializer())
        outputs = tf.nn.relu(tf.matmul (inputs, weight) + bias)
        
    
    with tf.variable_scope ('layer2'):
        weight = tf.get_variable ('weight', shape=[outputs.get_shape()[-1], 1], initializer=tf.contrib.layers.xavier_initializer())
        bias = tf.get_variable ('bias', shape=[1], initializer = tf.zeros_initializer())
        outputs = tf.matmul (outputs, weight) + bias
    
    '''
    with tf.variable_scope ('layer3'):
        weight = tf.get_variable ('weight', shape=[outputs.get_shape()[-1], 1], initializer=tf.truncated_normal_initializer(mean=mean, stddev=stddev))
        bias = tf.get_variable ('bias', shape=[1], initializer = tf.zeros_initializer())
        outputs = tf.matmul (outputs, weight) + bias
    '''
    '''
    with tf.variable_scope ('layer4'):
        weight = tf.get_variable ('weight', shape=[outputs.get_shape()[-1], 1], initializer=tf.truncated_normal_initializer(mean=mean, stddev=stddev))
        bias = tf.get_variable ('bias', shape=[1], initializer = tf.zeros_initializer())
        outputs = tf.matmul (outputs, weight) + bias
    '''
    print (outputs.shape)

    return outputs



def get_simple_lstm_model (inputs, is_training):
    '''
    Get the simple LSTM model

    Args:
    - inputs: tf tensor;
    - is_training: tf bool tensor;

    Returns:
    - pred: prediction;
    '''

    cell = tf.nn.rnn_cell.LSTMCell (
                    num_units=128, 
                    initializer=tf.contrib.layers.xavier_initializer(),
                    name="LSTM_cell")

    output_seq, state = tf.nn.dynamic_rnn (cell=cell, inputs=inputs, dtype=tf.float32)
    outputs = state.h

    with tf.variable_scope ('projection') as sc:
        weight = tf.get_variable ('weight', shape=[outputs.get_shape()[-1], 1], 
                    initializer=tf.contrib.layers.xavier_initializer())
        bias = tf.get_variable ('bias', shape=[1], initializer=tf.zeros_initializer())
        outputs = tf.matmul (outputs, weight) + bias

    print ('LSTM model outputs.shape', outputs.shape)
    
    return outputs




def get_cnn_lstm_model (inputs, is_training):
    '''
    Get the CNN-LSTM model

    Args:
    - inputs: tf tensor;
    - is_training: tf bool tensor;

    Returns:
    - pred: prediction;
    '''

    inputs = tf.reshape (inputs, shape=[-1, inputs.shape[1], inputs.shape[2]], name='input_reshape')

    conv1 = tf.layers.conv1d (
        inputs,
        filters=16,
        kernel_size=5,
        strides=1,
        padding='same',
        activation=tf.nn.leaky_relu,
        use_bias=False,
        bias_initializer=None,
        name='conv1'
    )
    '''
    conv2 = tf.layers.conv1d (
        conv1,
        filters=16,
        kernel_size=5,
        strides=1,
        padding='same',
        activation=tf.nn.leaky_relu,
        use_bias=True,
        bias_initializer=tf.contrib.layers.xavier_initializer(),
        name='conv2'
    )
    '''

    conv3 = tf.layers.conv1d (
        conv1,
        filters=32,
        kernel_size=5,
        strides=1,
        padding='same',
        activation=tf.nn.leaky_relu,
        use_bias=True,
        bias_initializer=tf.contrib.layers.xavier_initializer(),
        name='conv3'
    )
    '''
    conv4 = tf.layers.conv1d (
        conv3,
        filters=32,
        kernel_size=5,
        strides=1,
        padding='same',
        activation=tf.nn.leaky_relu,
        use_bias=True,
        bias_initializer=tf.contrib.layers.xavier_initializer(),
        name='conv4'
    )
    '''

    cell = tf.nn.rnn_cell.LSTMCell (
        num_units=256, 
        initializer=tf.contrib.layers.xavier_initializer(),
        name="LSTM_cell"
    )

    outputs_seq, state = tf.nn.dynamic_rnn (cell=cell, inputs=conv3, dtype=tf.float32)
    outputs = state.h

    with tf.variable_scope ('projection') as sc:
        weight = tf.get_variable ('weight', shape=[outputs.get_shape()[-1], 1], 
                    initializer=tf.contrib.layers.xavier_initializer())
        bias = tf.get_variable ('bias', shape=[1], initializer=tf.zeros_initializer())
        outputs = tf.matmul (outputs, weight) + bias

    print ('CNN-LSTM model outputs.shape', outputs.shape)

    return outputs




def train():
    n_inputs = 10
    n_outputs = 1
    n_features = None # get from data
    batch_size = 64
    n_epochs = 10

    # train and test data
    order_book = OrderBook (batch_size, DATA_DIR, data_regenerate_flag=False)
    #mean_list = order_book.mean_list
    #stddev_list = order_book.stddev_list
    num_batches = order_book.num_batches
    print ('num_batches', num_batches)
    n_features = order_book.num_features
    print ("n_features", n_features)

    inputs_pl = tf.placeholder (tf.float32, shape=[None, n_inputs, n_features], name='inputs_pl') # batch_size x len x n_features
    outputs_pl = tf.placeholder (tf.float32, shape=[None, n_outputs], name='outputs_pl') # batch_size x n_outputs
    means_pl = tf.placeholder (tf.float32, shape=[None], name='means_pl') # batch_size
    stddevs_pl = tf.placeholder (tf.float32, shape=[None], name='stddevs_pl') # batch_size
    is_training_pl = tf.placeholder (tf.bool, shape=[], name='is_training')
    
    means = tf.reshape (means_pl, shape=[-1, 1])
    stddevs = tf.reshape (stddevs_pl, shape=[-1, 1])
    outputs_for_acc = outputs_pl * stddevs + means

    pred = get_simple_lstm_model (inputs_pl, is_training_pl)
    test_pred = pred * stddevs + means
    loss = tf.losses.mean_squared_error (labels=outputs_pl, predictions=pred) #get_loss (pred, outputs_pl)
    tf.summary.scalar ('loss', loss)
    
    accuracy = tf.sqrt(tf.losses.mean_squared_error (outputs_for_acc, test_pred)) # already tested
    # accuracy_my = tf.reduce_mean (tf.square (tf.subtract (outputs_pl, pred)))
    tf.summary.scalar ('accuracy', accuracy)

    step = tf.Variable (0)
    learning_rate = 1e-3 #get_learning_rate (step, batch_size, base_learning_rate=1e-3, min_rate=1e-5, decay_rate=0.9)
    tf.summary.scalar ('learning rate', learning_rate)

    #update_ops = tf.get_collection (tf.GraphKeys.UPDATE_OPS)
    #with tf.control_dependencies (update_ops):
    train_op = tf.train.AdamOptimizer (learning_rate).minimize (loss, global_step=step)


    merged = tf.summary.merge_all()
    init = tf.global_variables_initializer()

    with tf.Session () as sess:
        sess.run (init)

        # delete previous data and create summary writers
        #os.system ('DEL /Q '+os.path.join(os.path.join(LOG_DIR, 'train'), '*'))
        train_writer = tf.summary.FileWriter (os.path.join (LOG_DIR, 'train'), graph=sess.graph)
        #test_writer = tf.summary.FileWriter (os.path.join (LOG_DIR, 'test'), graph=sess.graph)

        # create saver
        #saver = tf.train.Saver ()

        step_val = None
        for epoch in range (n_epochs):
            order_book.reset_batch()
            total_loss = 0.0
            total_acc = 0.0
            
            for i in range (num_batches):        
                batch_inputs, batch_labels, batch_means, batch_stddevs = order_book.next_batch_with_mean_and_stddev()
                feed_dict = {inputs_pl: batch_inputs.reshape (batch_size, n_inputs, n_features), 
                            outputs_pl: batch_labels.reshape(batch_size, n_outputs),
                            means_pl: batch_means,
                            stddevs_pl: batch_stddevs,
                            is_training_pl: True}
                #print ('batch_labels', batch_labels)
                _, loss_val, acc_val, step_val, summary = sess.run ([train_op, loss, accuracy, step, merged],
                            feed_dict=feed_dict)
                #print ('pred_val', pred_val)
                #input()

                # after every batch
                total_acc += acc_val
                total_loss += loss_val

                train_writer.add_summary (summary, global_step=step_val)

            print ('Epoch', epoch, 'train_loss', total_loss/num_batches, 'train_acc', total_acc/num_batches)
            
            
            dev_inputs, dev_labels, dev_means, dev_stddevs = order_book.dev_set()
            feed_dict = {inputs_pl: dev_inputs.reshape (-1, n_inputs, n_features),
                        outputs_pl: dev_labels.reshape (-1, n_outputs),
                        means_pl: np.asarray(dev_means),
                        stddevs_pl: np.asarray(dev_stddevs),
                        is_training_pl: False}
            acc_val, loss_val = sess.run ([accuracy, loss], feed_dict=feed_dict)

            print ('dev_loss', loss_val, 'dev_acc', acc_val)
            
            #saver.save (sess, os.path.join (SAVE_DIR, MODEL_FILENAME))
            

        output_f = open (os.path.join (BASE_DIR, OUTPUT_FILENAME), 'w')
        output_f.write ('caseid,midprice\n')
        
        print ("Number of test half days", len(order_book.test_inputs_list))
        base_index = 143
        for i in range (len(order_book.test_inputs_list)):
            test_data = order_book.test_inputs_list [i]
            test_means = np.zeros ([test_data.shape[0]])
            test_means.fill(order_book.test_means_list [i])
            test_stddevs = np.zeros ([test_data.shape[0]])
            test_stddevs.fill (order_book.test_stddevs_list [i])
            feed_dict = {inputs_pl: test_data, is_training_pl:False,
                         means_pl: test_means, stddevs_pl: test_stddevs}
                    
            pred_val = sess.run (test_pred, feed_dict=feed_dict)
            pred_val = np.asarray (pred_val)
            #print ('prediction shape', pred_val.shape)

            for j in range (len (pred_val)):
                output_f.write (str(j+base_index)+','+str(pred_val[j][0])+'\n')
            base_index += len (pred_val)

        output_f.close()
        



def prediction ():
    '''
    Predict via restoring trained models
    '''

    pass



def get_learning_rate (
    global_step, 
    batch_size,
    base_learning_rate=1e-4,
    decay_rate=0.7,
    decay_step=200000,
    min_rate=1e-5):
    '''
    Learning rate decay by global step

    Args:
        global_step: tf variable.
        base_learning_rate: float.
        batch_size: int. 
        decay_rate: float.
        decay_step: int.
        min_rate: float. lower bound of learning rate

    Returns:
        learning_rate: tf variable.
    '''

    '''
    exponential_decay(learning_rate, global_step,
    param learning_rate
    decay_steps,                        decay_rate,
    staircase=False,                        name=None)
    '''
    learning_rate = tf.train.exponential_decay (
                    base_learning_rate,
                    global_step * batch_size,
                    decay_step,
                    decay_rate)
    learning_rate = tf.maximum(learning_rate, min_rate)
    return learning_rate 



def dropout (
    inputs,
    is_training,
    scope,
    keep_prob=0.5
):
    '''
    Dropout layer, suitable for training and testing procedure

    Args:
        inputs: tensor;
        is_training: boolean tf variable;
        scope: string;
        keep_prob: float in [0, 1].

    Returns:
        tensor variable
    '''

    with tf.variable_scope (scope) as sc:
        outputs = tf.cond (is_training,
                lambda: tf.nn.dropout (inputs, keep_prob),
                lambda: inputs)
    return outputs



def get_loss (prediction, labels):
    return tf.losses.mean_squared_error (labels, prediction)
    tf.losses.mean_pairwise_squared_error()




def get_cell (num_units):
    '''
    Get a cell for recurrent NN

    Args:
        num_units: int, state_size
    
    Returns:
        an instance of a subclass of RNNCell 
    '''
    return tf.nn.rnn_cell.LSTMCell (
                    num_units=num_units, 
                    use_peepholes=True,
                    initializer=tf.contrib.layers.xavier_initializer(),
                    num_proj=1,
                    name="LSTM_cell")



def restore_answer ():
    '''
    Restore answer
    '''

    in_f = open (os.path.join (BASE_DIR, OUTPUT_FILENAME), 'r')
    out_f = open (os.path.join (BASE_DIR, OUTPUT_FILENAME + '.restore'), 'w')

    order_book = OrderBook (256, DATA_DIR)
    pre_mean = order_book.pre_mean[0]
    pre_stddev = order_book.pre_stddev[0]
    out_f.write (in_f.readline())
    for raw_line in in_f:
        line = raw_line.strip().split (',')
        line[1] = float(line[1])
        line[1] = line[1] * pre_stddev + pre_mean
        line[1] = str(line[1])

        out_f.write (str(','.join(line)) + '\n')

    in_f.close()
    out_f.close()




if __name__ == "__main__":
    train ()
    #restore_answer ()