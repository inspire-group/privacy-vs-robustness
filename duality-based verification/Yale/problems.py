# a hack to ensure scripts search cwd
import sys
sys.path.append('.')

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn.functional as F
import numpy as np
import torch.utils.data as td
import argparse
from convex_adversarial import epsilon_from_model, DualNetBounds
from convex_adversarial import Dense, DenseSequential
import math
import os
import sys
sys.path.append('../../')
from utils import *




def YALEBXF_loader(batch_size=20, test_batch_size=2):
    train_data, train_label, test_data, test_label = YALE_split('../../datasets/yale/YALEBXF.mat') 
    train_data = train_data.transpose((0, 3, 1, 2) )
    test_data = test_data.transpose((0, 3, 1, 2) )
    train_data = np.pad(train_data,((0,0),(0,0),(0,0),(12,12)),'constant', constant_values=(0, 0))
    test_data = np.pad(test_data,((0,0),(0,0),(0,0),(12,12)),'constant', constant_values=(0, 0))
    train_data = (train_data)*1
    test_data = (test_data)*1
    print(np.amax(train_data), np.amin(test_data), train_data.shape, test_data.shape)
    tensor_x = torch.stack([torch.FloatTensor(i) for i in train_data]) # transform to torch tensors
    tensor_y = torch.stack([torch.LongTensor(np.array(i)) for i in train_label])
    train_dataset = torch.utils.data.TensorDataset(tensor_x,tensor_y)
    
    tensor_x = torch.stack([torch.FloatTensor(i) for i in test_data]) # transform to torch tensors
    tensor_y = torch.stack([torch.LongTensor(np.array(i)) for i in test_label])
    test_dataset = torch.utils.data.TensorDataset(tensor_x,tensor_y)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
        shuffle=True, pin_memory=True)
    
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=test_batch_size,
        shuffle=False, pin_memory=True)
    return train_loader, test_loader



def argparser(batch_size=50, epochs=20, seed=0, verbose=1, lr=1e-4, 
              epsilon=0.1, starting_epsilon=None, 
              proj=None, 
              norm_train='l1', norm_test='l1', 
              opt='sgd', momentum=0.9, weight_decay=5e-4): 

    parser = argparse.ArgumentParser()

    # optimizer settings
    parser.add_argument('--opt', default=opt)
    parser.add_argument('--momentum', type=float, default=momentum)
    parser.add_argument('--weight_decay', type=float, default=weight_decay)
    parser.add_argument('--batch_size', type=int, default=batch_size)
    parser.add_argument('--test_batch_size', type=int, default=batch_size)
    parser.add_argument('--epochs', type=int, default=epochs)
    parser.add_argument("--lr", type=float, default=lr)

    # epsilon settings
    parser.add_argument("--epsilon", type=float, default=epsilon)
    parser.add_argument("--starting_epsilon", type=float, default=starting_epsilon)
    parser.add_argument('--schedule_length', type=int, default=10)

    # projection settings
    parser.add_argument('--proj', type=int, default=proj)
    parser.add_argument('--norm_train', default=norm_train)
    parser.add_argument('--norm_test', default=norm_test)

    # model arguments
    parser.add_argument('--num_save', type=int, default=5)
    parser.add_argument('--width_num', type=int, default=2)
    parser.add_argument('--model', default=None)
    parser.add_argument('--model_factor', type=int, default=8)
    parser.add_argument('--cascade', type=int, default=1)
    parser.add_argument('--method', default=None)
    parser.add_argument('--resnet_N', type=int, default=1)
    parser.add_argument('--resnet_factor', type=int, default=1)


    # other arguments
    parser.add_argument('--prefix')
    parser.add_argument('--load')
    parser.add_argument('--real_time', action='store_true')
    parser.add_argument('--seed', type=int, default=seed)
    parser.add_argument('--verbose', type=int, default=verbose)
    parser.add_argument('--cuda_ids', default=None)

    
    args = parser.parse_args()
    if args.starting_epsilon is None:
        args.starting_epsilon = args.epsilon 
    if args.prefix: 
        if args.model is not None: 
            args.prefix += '_'+args.model

        if args.method is not None: 
            args.prefix += '_'+args.method

        banned = ['verbose', 'prefix',
                  'resume', 'baseline', 'eval', 
                  'method', 'model', 'cuda_ids', 'load', 'real_time', 
                  'test_batch_size']
        if args.method == 'baseline':
            banned += ['epsilon', 'starting_epsilon', 'schedule_length', 
                       'l1_test', 'l1_train', 'm', 'l1_proj']

        # Ignore these parameters for filename since we never change them
        banned += ['momentum', 'weight_decay']

        if args.cascade == 1: 
            banned += ['cascade']

        # if not using a model that uses model_factor, 
        # ignore model_factor
        if args.model not in ['wide', 'deep']: 
            banned += ['model_factor']

        # if args.model != 'resnet': 
        banned += ['resnet_N', 'resnet_factor']

        for arg in sorted(vars(args)): 
            if arg not in banned and getattr(args,arg) is not None: 
                args.prefix += '_' + arg + '_' +str(getattr(args, arg))

        if args.schedule_length > args.epochs: 
            raise ValueError('Schedule length for epsilon ({}) is greater than '
                             'number of epochs ({})'.format(args.schedule_length, args.epochs))
    else: 
        args.prefix = 'temporary'

    if args.cuda_ids is not None: 
        print('Setting CUDA_VISIBLE_DEVICES to {}'.format(args.cuda_ids))
        os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_ids


    return args

def args2kwargs(args, X=None): 

    if args.proj is not None: 
        kwargs = {
            'proj' : args.proj, 
        }
    else:
        kwargs = {
        }
    kwargs['parallel'] = (args.cuda_ids is not None)
    return kwargs


