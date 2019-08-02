import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import scipy.io as io
import os
import sys


def YALE_split(Yale_file, train_points_per_label = 50):
    YALE = io.loadmat(Yale_file) 
    X = YALE['X']
    Y = YALE['Y']
    X = X.T/255.0
    X = X.reshape((2414,  168, 192)).swapaxes(1,2)
    Y = Y.flatten()
    
    train_data, train_label, test_data, test_label = [], [], [], []
    np.random.seed(0)
    label_count = 0
    for label in set(Y):
        label_idx = np.argwhere(Y==label).flatten()
        tot_num = len(label_idx)
        idx_permute = np.random.permutation(label_idx)
        train_data.append(X[idx_permute[:train_points_per_label]])
        train_label.append(np.repeat(label_count, train_points_per_label))
        test_data.append(X[idx_permute[train_points_per_label:]])
        test_label.append(np.repeat(label_count, tot_num - train_points_per_label))
        label_count += 1
    train_data = np.concatenate(train_data)
    test_data = np.concatenate(test_data)
    train_label = np.concatenate(train_label)
    test_label = np.concatenate(test_label)
    
    train_idx_permute = np.random.permutation(len(train_label))
    train_data = np.expand_dims(train_data[train_idx_permute], 3)
    train_label = train_label[train_idx_permute]
    
    test_idx_permute = np.random.permutation(len(test_label))
    test_data = np.expand_dims(test_data[test_idx_permute],3)
    test_label = test_label[test_idx_permute]            
    return train_data, train_label, test_data, test_label