# import waitGPU
# import setGPU
# waitGPU.wait(utilization=50, available_memory=10000, interval=60)
# waitGPU.wait(gpu_ids=[1,3], utilization=20, available_memory=10000, interval=60)

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
# cudnn.benchmark = True

import torchvision.transforms as transforms
import torchvision.datasets as datasets

import setproctitle

import problems as pblm
from trainer import *
from model import *
import math
import numpy as np
import os


if __name__ == "__main__": 
    args = pblm.argparser(opt='adam', verbose=200, starting_epsilon=0.01)
    print("saving file to {}".format(args.prefix))
    setproctitle.setproctitle(args.prefix)
    

    train_log = open(args.prefix + "_train.log", "w")
    test_log = open(args.prefix + "_test.log", "w")

    train_loader, test_loader = pblm.YALEBXF_loader(args.batch_size, args.test_batch_size)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    #for X,y in train_loader: 
    #    break
    kwargs = pblm.args2kwargs(args)
    best_err = 1

    sampler_indices = []
    model = [YALE_model(args.width_num).cuda()]
    
    ########################################
    NAN_EXIST = False
    #######################################

    for _ in range(0,args.cascade): 
        if _ > 0: 
            # reduce dataset to just uncertified examples
            print("Reducing dataset...")
            train_loader = sampler_robust_cascade(train_loader, model, args.epsilon,
                                                  args.test_batch_size, 
                                                  norm_type=args.norm_test, bounded_input=True, **kwargs)
            if train_loader is None: 
                print('No more examples, terminating')
                break
            sampler_indices.append(train_loader.sampler.indices)

            print("Adding a new model")
            model.append(select_model(args.model))
        print(args.opt)
        if args.opt == 'adam': 
            opt = optim.Adam(model[-1].parameters(), lr=args.lr, weight_decay=args.weight_decay)
        elif args.opt == 'sgd': 
            opt = optim.SGD(model[-1].parameters(), lr=args.lr, 
                            momentum=args.momentum,
                            weight_decay=args.weight_decay)
        else: 
            raise ValueError("Unknown optimizer")
        lr_scheduler = optim.lr_scheduler.StepLR(opt, step_size=25, gamma=0.5)
        eps_schedule = np.linspace(args.starting_epsilon, 
                                   args.epsilon, 
                                   args.schedule_length)

        for t in range(args.epochs):
            lr_scheduler.step(epoch=max(t-len(eps_schedule), 0))
            if t < len(eps_schedule) and args.starting_epsilon is not None: 
                epsilon = float(eps_schedule[t])
            else:
                epsilon = args.epsilon


            # standard training
            if args.method == 'baseline': 
                for param_group in opt.param_groups:
                    print( param_group['lr'])
                train_baseline(train_loader, model[0], opt, t, train_log,
                                args.verbose)
                err = evaluate_baseline(test_loader, model[0], t, test_log,
                                args.verbose)

            # madry training
            elif args.method=='madry':
                train_madry(train_loader, model[0], args.epsilon, 
                            opt, t, train_log, args.verbose)
                err = evaluate_madry(test_loader, model[0], args.epsilon, 
                                     t, test_log, args.verbose)
            
            # robust cascade training
            elif args.cascade > 1: 
                train_robust(train_loader, model[-1], opt, epsilon, t,
                                train_log, args.verbose, args.real_time,
                                norm_type=args.norm_train, bounded_input=True,
                                **kwargs)
                
                err = evaluate_robust_cascade(test_loader, model,
                   args.epsilon, t, test_log, args.verbose,
                   norm_type=args.norm_test, bounded_input=True,  **kwargs)
            
            # robust training
            else:
                for param_group in opt.param_groups:
                    print( param_group['lr'])
                train_robust(train_loader, model[0], opt, epsilon, t,
                   train_log, args.verbose, args.real_time,
                   norm_type=args.norm_train, bounded_input=False, clip_grad=1, **kwargs)
                ##############################################################################
                for p in model[0].parameters():
                    if np.isnan(p.grad).any():
                        print('Gradient just goes to NAN VALUE!!!!!!!!!!!!!!!!!!!!!!!!!!')
                        NAN_EXIST = True
                        break
                #############################################################################
                err = evaluate_robust(test_loader, model[0], args.epsilon,
                   t, test_log, args.verbose, args.real_time,
                   norm_type=args.norm_test, bounded_input=False, **kwargs)
            #########################################################################
            if NAN_EXIST == True:
                break
            #########################################################################
            if err < best_err: 
                best_err = err
                torch.save({
                    'state_dict' : [m.state_dict() for m in model], 
                    'err' : best_err,
                    'epoch' : t,
                    'sampler_indices' : sampler_indices
                    }, args.prefix + "_best.pth")
            
            if t%args.num_save == 0:    
                torch.save({ 
                    'state_dict': [m.state_dict() for m in model],
                    'err' : err,
                    'epoch' : t,
                    'sampler_indices' : sampler_indices
                    }, args.prefix + "_checkpoint_"+str(t)+".pth")
