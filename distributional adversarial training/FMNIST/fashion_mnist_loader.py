"""
Utilities for importing the FMNIST dataset.
Each image in the dataset is a numpy array of shape (28, 28, 1), with the values from 0 to 1.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import pickle
import sys
import tensorflow as tf

import numpy as np

class FASHION_MNIST_DATA(object):
    
    def __init__(self):
        (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.fashion_mnist.load_data()
        train_images = train_images/255.0
        test_images = test_images/255.0
        print(np.amax(train_images), np.amin(train_images))
        self.train_data = DataSubset(train_images, train_labels)
        self.test_data = DataSubset(test_images, test_labels)


class DataSubset(object):
    def __init__(self, xs, ys):
        self.xs = xs
        self.n = xs.shape[0]
        self.ys = ys
        self.batch_start = 0
        self.cur_order = np.random.permutation(self.n)

    def get_next_batch(self, batch_size, multiple_passes=True, reshuffle_after_pass=True):
        if self.n < batch_size:
            raise ValueError('Batch size can be at most the dataset size')
        if not multiple_passes:
            actual_batch_size = min(batch_size, self.n - self.batch_start)
            if actual_batch_size <= 0:
                raise ValueError('Pass through the dataset is complete.')
            batch_end = self.batch_start + actual_batch_size
            batch_xs = self.xs[self.cur_order[self.batch_start : batch_end], ...]
            batch_ys = self.ys[self.cur_order[self.batch_start : batch_end], ...]
            self.batch_start += actual_batch_size
            return batch_xs, batch_ys
        actual_batch_size = min(batch_size, self.n - self.batch_start)
        if actual_batch_size < batch_size:
            if reshuffle_after_pass:
                self.cur_order = np.random.permutation(self.n)
            self.batch_start = 0
        batch_end = self.batch_start + batch_size
        batch_xs = self.xs[self.cur_order[self.batch_start : batch_end], ...]
        batch_ys = self.ys[self.cur_order[self.batch_start : batch_end], ...]
        self.batch_start += batch_size
        return batch_xs, batch_ys
