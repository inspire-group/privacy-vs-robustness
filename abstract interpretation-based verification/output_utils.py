import os 
import torch
import torchvision
import torch.nn
from torch.autograd import Variable
from torch.autograd.gradcheck import zero_gradients
from torch.autograd import Variable
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
import math


entropy_loss = torch.nn.CrossEntropyLoss()

def PGD_perturb(model, image_tensor, img_variable,img_labels, n_steps, eps_max, eps_step, clip_min=0.0, clip_max=1.0):
    output = model.forward(img_variable)
    for i in range(n_steps):
        zero_gradients(img_variable)                       #flush gradients
        output = model.forward(img_variable)         #perform forward pass
        loss_cal= entropy_loss(output,img_labels)
        loss_cal.backward()
        x_grad = eps_step * torch.sign(img_variable.grad.data)   # as per the formula
        adv_temp = img_variable.data + x_grad                 
        
        total_grad = adv_temp - image_tensor                  #total perturbation
        total_grad = torch.clamp(total_grad, -eps_max, eps_max)
        x_adv = image_tensor + total_grad                      #add total perturbation to the original image
        x_adv = torch.clamp(torch.clamp(x_adv-image_tensor, -1*eps_max, eps_max)+image_tensor, clip_min, clip_max)
        img_variable.data = x_adv
    
    return img_variable

def softmax_by_row(logits, T = 1.0):
    mx = np.max(logits, axis=-1, keepdims=True)
    exp = np.exp((logits - mx)/T)
    denominator = np.sum(exp, axis=-1, keepdims=True)
    return exp/denominator

def classifier_performance(model, train_loader, test_loader, num_step, max_perturb, step_size, clip_min=0, clip_max=1.0):
    output_train_benign = []
    output_train_adversarial = []
    train_label = []
    for num, data in enumerate(train_loader):
        images,labels = data
        image_tensor= images.cuda()
        img_variable = Variable(image_tensor, requires_grad=True)
        output = model.forward(img_variable)

        train_label.append(labels.numpy())
        output_train_benign.append(softmax_by_row(output.data.cpu().numpy(),T = 1))

        label_var = Variable(labels.cuda(), requires_grad=False)
        img_perturb = PGD_perturb(model, image_tensor, img_variable, label_var,num_step,max_perturb,step_size,clip_min,clip_max)
        output = model.forward(img_perturb)
        output_train_adversarial.append(softmax_by_row(output.data.cpu().numpy(),T = 1))

    
    train_label = np.concatenate(train_label)
    output_train_benign=np.concatenate(output_train_benign)
    output_train_adversarial=np.concatenate(output_train_adversarial)

    test_label = []
    output_test_benign = []
    output_test_adversarial = []

    for num, data in enumerate(test_loader):
        images,labels = data

        image_tensor= images.cuda()
        img_variable = Variable(image_tensor, requires_grad=True)

        output = model.forward(img_variable)

        test_label.append(labels.numpy())
        output_test_benign.append(softmax_by_row(output.data.cpu().numpy(),T = 1))

        label_var = Variable(labels.cuda(), requires_grad=False)
        img_perturb = PGD_perturb(model, image_tensor,img_variable,label_var,num_step,max_perturb,step_size,clip_min, clip_max)
        output = model.forward(img_perturb)
        output_test_adversarial.append(softmax_by_row(output.data.cpu().numpy(),T = 1))

    test_label = np.concatenate(test_label)
    output_test_benign=np.concatenate(output_test_benign)
    output_test_adversarial=np.concatenate(output_test_adversarial)

    train_acc1 = np.sum(np.argmax(output_train_benign,axis=1) == train_label.flatten())/len(train_label)
    train_acc2 = np.sum(np.argmax(output_train_adversarial,axis=1) == train_label.flatten())/len(train_label)
    test_acc1 = np.sum(np.argmax(output_test_benign,axis=1) == test_label.flatten())/len(test_label)
    test_acc2 = np.sum(np.argmax(output_test_adversarial,axis=1) == test_label.flatten())/len(test_label)
    print('Benign accuracy: ', (train_acc1, test_acc1), ' Adversarial accuracy: ', (train_acc2, test_acc2))
    
    
    return output_train_benign, output_train_adversarial, output_test_benign, output_test_adversarial, train_label, test_label