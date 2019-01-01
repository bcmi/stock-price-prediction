import tensorflow as tf
import numpy as np

from data_process import Data


def lstm_model(inputs, n_units):
    '''
    inputs: batch_size x length x n_features
    outputs_seq: batch_size x length x n_units
    finalH: batch_size x n_units
    outputs: batch_size x 1
    '''
    cell = tf.nn.rnn_cell.LSTMCell (
                    num_units=n_units, 
                    initializer=tf.contrib.layers.xavier_initializer())
    outputs_seq, lastState = tf.nn.dynamic_rnn(cell, inputs, dtype = tf.float32)
    finalH = lastState.h
    
    with tf.variable_scope('projection') as sc:
        weight = tf.get_variable('weight', shape = [finalH.get_shape()[-1], 1], initializer=tf.contrib.layers.xavier_initializer())
        bias = tf.get_variable('bias', shape = [1], initializer=tf.zeros_initializer())
        outputs = tf.matmul (finalH, weight) + bias
    
    #print 'LSTM model shape:', outputs_seq.shape, finalH.shape, outputs.shape
    
    return outputs


def lstm_attention_model(inputs, n_units):
    '''
    inputs: batch_size x length x n_features
    outputs_seq: batch_size x length x n_units
    finalH: batch_size x n_units
    outputs: batch_size x 1
    '''
    #cell = tf.contrib.rnn.BasicLSTMCell(num_units = n_units, state_is_tuple = True)
    cell = tf.nn.rnn_cell.LSTMCell (
                    num_units=n_units, 
                    initializer=tf.contrib.layers.xavier_initializer())
    rnn_cell = tf.contrib.rnn.AttentionCellWrapper(
            cell, attn_length = 10)
    outputs_seq, lastState = tf.nn.dynamic_rnn(rnn_cell, inputs, dtype = tf.float32)
    finalH = lastState[0].h
    
    with tf.variable_scope('projection') as sc:
        weight = tf.get_variable('weight', shape = [finalH.get_shape()[-1], 1], initializer=tf.contrib.layers.xavier_initializer())
        bias = tf.get_variable('bias', shape = [1], initializer=tf.zeros_initializer())
        outputs = tf.matmul (finalH, weight) + bias
    
    #print 'LSTM model shape:', outputs_seq.shape, finalH.shape, outputs.shape
    
    return outputs


def train():
    epochs = 50
    length = 10
    n_units = 128
    n_features = 6
    batch_size = 64
    
    data = Data(batch_size)
    num_batches = data.num_batches()
    
    xplaceholder = tf.placeholder(tf.float32, shape = [None, length, n_features])
    yplaceholder = tf.placeholder(tf.float32, shape = [None, 1])
    midPrice_means = tf.placeholder(tf.float32, shape = [None, 1])
    midPrice_stddevs = tf.placeholder(tf.float32, shape = [None, 1])
    
    #origin midPrice
    origin_midPrice = yplaceholder * midPrice_stddevs + midPrice_means
    
    pred = lstm_model(xplaceholder, n_units)
    
    #pred midPrice
    pred_midPrice = pred * midPrice_stddevs + midPrice_means
    
    loss = tf.losses.mean_squared_error(labels = yplaceholder, predictions = pred)
    tf.summary.scalar('loss', loss)
    
    accuracy = tf.sqrt(tf.losses.mean_squared_error(origin_midPrice, pred_midPrice))
    tf.summary.scalar('accuracy', accuracy)
    
    step = tf.Variable (0)
    learning_rate = 1e-4
    tf.summary.scalar ('learning rate', learning_rate)
    optimizer = tf.train.AdamOptimizer (learning_rate).minimize (loss, global_step=step)
    
    merged = tf.summary.merge_all()
    init = tf.global_variables_initializer()
    
    with tf.Session() as sess:
        sess.run(init)
        
        train_writer = tf.summary.FileWriter('log', graph = sess.graph)
        
        saver = tf.train.Saver (max_to_keep=3)
        
        step_val = None
        for epoch in range(epochs):
            data.reset_batch()
            total_loss = 0.0
            total_acc = 0.0
            
            for i in range(num_batches):
                batch_inputs, batch_labels, batch_means, batch_stddevs = data.next_batch()
                feed_dict = {xplaceholder: batch_inputs, 
                             yplaceholder: batch_labels,
                             midPrice_means: batch_means,
                             midPrice_stddevs: batch_stddevs}
                _, loss_val, acc_val, step_val, summary = sess.run ([optimizer, loss, accuracy, step, merged], feed_dict=feed_dict)
                
                total_acc += acc_val
                total_loss += loss_val
                
                train_writer.add_summary(summary, global_step = step_val)
                
            print 'Epoch', epoch, 'train_loss', total_loss/num_batches, 'train_acc', total_acc/num_batches
            
            '''
            dev_inputs, dev_labels = data.get_dev_data()
            feed_dict = {xplaceholder: dev_inputs, yplaceholder: dev_labels}
            acc_val, loss_val = sess.run([accuracy, loss], feed_dict = feed_dict)
            print 'dev_loss', loss_val, 'dev_acc', acc_val
            '''
            
        outfile = open('outputs10.csv', 'w')
        outfile.write ('midprice\n')
        test_inputs_list, test_means_list, test_stddevs_list = data.get_test_data()
        for i in range(data.test_num_half_day):
            test_means = []
            test_stddevs = []
            test_inputs = test_inputs_list[i]
            mean = test_means_list[i][0]
            stddev = test_stddevs_list[i][0]
            for j in range(len(test_inputs)):
                test_means.append(mean)
                test_stddevs.append(stddev)
            test_inputs = np.asarray(test_inputs)
            test_means = np.asarray(test_means).reshape([-1,1])
            test_stddevs = np.asarray(test_stddevs).reshape([-1,1])
            feed_dict = {xplaceholder: test_inputs,
                         midPrice_means: test_means,
                         midPrice_stddevs: test_stddevs}
            pred_val = sess.run(pred_midPrice, feed_dict = feed_dict)
            pred_val = np.asarray(pred_val)
            #print pred_val.shape
            for i in range(len(pred_val)):
                outfile.write(str(pred_val[i][0]) + '\n')
        outfile.close()















if __name__ == "__main__":
    train()
