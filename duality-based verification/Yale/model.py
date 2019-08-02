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
import math
import os



class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)



def YALE_model(width_num=2):
    model = nn.Sequential(
        nn.Conv2d(1, 4*width_num, 3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(4*width_num, 4*width_num, 4, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(4*width_num, 8*width_num, 3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(8*width_num, 8*width_num, 4, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(8*width_num, 16*width_num, 3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(16*width_num, 16*width_num, 4, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(16*width_num, 32*width_num, 3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(32*width_num, 32*width_num, 4, stride=2, padding=1),
        nn.ReLU(),
        
        Flatten(),
        nn.Linear(32*width_num*12*12,200),
        nn.ReLU(),
        nn.Linear(200,38)
    )
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
            m.bias.data.zero_()
    return model
    