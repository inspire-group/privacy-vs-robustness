"""
The model is adapted from the tensorflow tutorial:
https://www.tensorflow.org/get_started/mnist/pros
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

class Model(object):
    def __init__(self, width_num=10):
        self.width_num = width_num
        self._build_model()
        
    def _fully_connected(self, x, inter_dim, out_dim):
        """FullyConnected layer for final output."""
        num_non_batch_dimensions = len(x.shape)
        prod_non_batch_dimensions = 1
        for ii in range(num_non_batch_dimensions - 1):
              prod_non_batch_dimensions *= int(x.shape[ii + 1])
        x = tf.reshape(x, [tf.shape(x)[0], -1])
        w = tf.get_variable(
            'DW_0', [prod_non_batch_dimensions, inter_dim],
            initializer=tf.uniform_unit_scaling_initializer(factor=1.0))
        b = tf.get_variable('biases_0', [inter_dim],
                        initializer=tf.constant_initializer())
        x = tf.nn.xw_plus_b(x, w, b)
        x = tf.nn.relu(x)
        w1 = tf.get_variable(
            'DW_1', [inter_dim, out_dim],
            initializer=tf.uniform_unit_scaling_initializer(factor=1.0))
        b1 = tf.get_variable('biases_1', [out_dim],
                        initializer=tf.constant_initializer())
        
        return tf.nn.xw_plus_b(x, w1, b1)
    
    def _conv(self, name, x, filter_size, in_filters, out_filters, strides):
        """Convolution."""
        with tf.variable_scope(name):
              n = filter_size * filter_size * out_filters
              kernel = tf.get_variable(
                  'DW', [filter_size, filter_size, in_filters, out_filters],
                  tf.float32, initializer=tf.random_normal_initializer(
                      stddev=np.sqrt(2.0/n)))
              return tf.nn.conv2d(x, kernel, strides, padding='SAME')
    def _decay(self):
        """L2 weight decay loss."""
        costs = []
        for var in tf.trainable_variables():
            if var.op.name.find('DW') > 0:
                costs.append(tf.nn.l2_loss(var))
        return tf.add_n(costs)
            
    def _build_model(self):
        self.x_input = tf.placeholder(tf.float32, shape = [None, 28, 28])
        self.y_input = tf.placeholder(tf.int64, shape = [None])
        self.x_image = tf.reshape(self.x_input, [-1, 28, 28, 1])
        
        x = self._conv('conv_0', self.x_image, 3, 1, 16*self.width_num, [1, 1, 1, 1])
        x = tf.nn.relu(x)
        x = self._conv('conv_1', x, 3, 16*self.width_num, 16*self.width_num, [1, 1, 1, 1])
        x = tf.nn.relu(x)
        x = self._conv('conv_2', x, 3, 16*self.width_num, 16*self.width_num, [1, 2, 2, 1])
        x = tf.nn.relu(x)
        #print(x.shape)
        x = self._conv('conv_3', x, 3, 16*self.width_num, 32*self.width_num, [1, 1, 1, 1])
        x = tf.nn.relu(x)
        x = self._conv('conv_4', x, 3, 32*self.width_num, 32*self.width_num, [1, 1, 1, 1])
        x = tf.nn.relu(x)
        x = self._conv('conv_5', x, 3, 32*self.width_num, 32*self.width_num, [1, 2, 2, 1])
        x = tf.nn.relu(x)
        #print(x.shape)
        
    
        with tf.variable_scope('logit'):
            self.pre_softmax = self._fully_connected(x, 200, 10)



        y_xent = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=self.y_input, logits=self.pre_softmax)

        self.xent = tf.reduce_sum(y_xent)
        self.weight_decay_loss = self._decay()

        self.y_pred = tf.argmax(self.pre_softmax, 1)

        correct_prediction = tf.equal(self.y_pred, self.y_input)

        self.num_correct = tf.reduce_sum(tf.cast(correct_prediction, tf.int64))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    
  
    

