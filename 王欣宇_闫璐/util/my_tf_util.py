# encoding: utf-8
# file: my_tf_util.py
# author: shawn233

from __future__ import print_function
import tensorflow as tf
import numpy as np


def _variable_on_cpu (name, shape, initializer):
    '''

    '''
    pass


def _variable_on_gpu (name, shape, initializer):
    '''
    '''
    pass


def _variable (name, shape, initializer):
    return tf.get_variable (name, shape, initializer=initializer, dtype=tf.float32)


def conv2d (
    inputs,
    num_out_channels,
    kernel_size,
    scope,
    stride = [1, 1],
    padding = 'SAME',
    stddev = 1e-3,
    activation_fn = tf.nn.relu
):

    '''
    2D convolutional layer

    Args:
        inputs: 4-D tensor variable, batch_size x input_h x input_w x in_channel
        num_out_channels: int, # output channels
        kernel_size: a list of 2 ints, [kernel_h, kernel_w]
        scope: string, indicating variable_scope
        stride: a list of 2 ints, []
        padding: 'SAME' or 'VALID'
        stddev: float, stddev for truncated_normal_initializer
        activation_fn: function

    Returns:
        Variable tensor
    '''

    outputs = None
    with tf.variable_scope (scope) as sc:
        # create kernel
        kernel_h, kernel_w = kernel_size
        num_in_channels = inputs.get_shape()[-1].value
        kernel_shape = [kernel_h, kernel_w,
                        num_in_channels, num_out_channels]
        kernel = _variable ('weights', kernel_shape,
                            tf.truncated_normal_initializer(stddev=stddev))

        # create conv2d layer
        stride_h, stride_w = stride
        outputs = tf.nn.conv2d (inputs, kernel,
                                [1, stride_h, stride_w, 1],
                                padding=padding)
        biases = _variable ('biases', [num_out_channels],
                            tf.constant_initializer(0.0))
        outputs = tf.nn.bias_add (outputs, biases)

        # batch norm here

        # activation
        outputs = activation_fn (outputs)
    return outputs


def fully_connected (
    inputs,
    num_outputs,
    scope,
    stddev = 1e-3,
    activation_fn = tf.nn.relu
):
    '''
    Fully connected layer

    Args:
        inputs: 2-D tensor, batch_size x n_input
        num_outputs: int, # output neurons
        scope: string, variable_scope
        stddev: float, for truncated_normal_initializer
        activation_fn: function

    Returns:
        Variable tensor
    '''
    
    outputs = None
    with tf.variable_scope (scope) as sc:
        num_input_units = inputs.get_shape()[-1].value
        # weights and biases
        weights = _variable ('weights',
                            shape=[num_input_units, num_outputs],
                            initializer=tf.truncated_normal_initializer(stddev=stddev))

        outputs = tf.matmul (inputs, weights)
        biases = _variable ('biases', [num_outputs],
                            tf.constant_initializer(0.0))
        outputs = tf.nn.bias_add (outputs, biases)

        # batch norm here

        # activation
        if activation_fn is not None:
            outputs = activation_fn(outputs)
    return outputs


def max_pool2d (
    inputs, 
    kernel_size,
    scope,
    stride = [2,2],
    padding = 'VALID'
):
    '''
    2-D max pooling layer

    Args:
        inputs: 4-D tensor, batch_size x input_h x input_w x in_channels
        kernel_size: a list of 2 ints, kernel_h x kernel_w
        scope: string, variable scope
        stride: a list of 2 ints

    Returns:
        Variable tensor
    '''

    outputs = None
    with tf.variable_scope (scope) as sc:
        kernel_h, kernel_w = kernel_size
        stride_h, stride_w = stride
        outputs = tf.nn.max_pool (inputs,
                                ksize=[1, kernel_h, kernel_w, 1],
                                strides=[1, stride_h, stride_w, 1],
                                padding=padding,
                                name=sc.name)
    return outputs


def avg_pool2d (
    inputs,
    kernel_size,
    scope,
    stride = [2,2],
    padding='VALID'
):
    '''
    2-D average pooling
    
    Args:
        inputs: 4-D tensor, batch_size x input_h x input_w x in_channels
        kernel_size: a list of 2 ints, kernel_h x kernel_w
        scope: string, variable scope
        stride: a list of 2 ints

    Returns:
        Variable tensor
    '''

    outputs = None
    with tf.variable_scope (scope) as sc:
        kernel_h, kernel_w = kernel_size
        stride_h, stride_w = stride
        outputs = tf.nn.avg_pool (inputs,
                                ksize=[1, kernel_h, kernel_w, 1],
                                strides=[1, stride_h, stride_w, 1],
                                padding=padding,
                                name=sc.name)
    return outputs



def reshape_pool (
    pool,
    scope
):
    '''
    Reshape the output of some pooling layer for later fully connected layer

    Args:
        pool: a 4-D tensor, output of pooling layer. batch_size x width x height x n_channels
        scope: string. 

    Returns:
        reshaped_pool: a 2-D tensor, input for fully connected layer. batch_size x n_features
    '''

    reshape_pool = None
    with tf.variable_scope (scope) as sc:
        gv = pool.get_shape ()
        w, h, c = gv[1].value, gv[2].value, gv[3].value
        new_shape = w*h*c

        reshaped_pool = tf.reshape (pool, [-1, new_shape], name='reshape')
    
    return reshaped_pool



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

    outputs = None
    with tf.variable_scope (scope) as sc:
        outputs = tf.cond (is_training,
                lambda: tf.nn.dropout (inputs, keep_prob),
                lambda: inputs)
    return outputs


if __name__ == "__main__":
    pass