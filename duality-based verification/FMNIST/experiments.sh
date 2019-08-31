#!/bin/bash 
# PYTHONPATH=/nethome/ericwong/convex_adversarial.preview.nips
# for L2 note that an L-infinity ball with radius eps
# has approximately the same volume as an L2 ball with radius
# sqrt(d/pi)*eps, where d is the number of dimensions. 
# For MNIST this is 
# sqrt(784/pi)*0.1=1.58
# sqrt(784/pi)*0.3=4.74
# and for cifar this is 
# sqrt(1024/pi)*0.0348=0.628
# sqrt(1024/pi)*0.139=2.51

# arguments that are universal across all experiments
cuda_ids=6,7
epochs=61
schedule_length=20

# L2 ball arguments
norm_type=l1_median
norm_eval=l1

# FMNIST parameters
prefix="./model"
starting_epsilon=0.003
parameters="--epochs ${epochs} --starting_epsilon ${starting_epsilon} --schedule_length ${schedule_length} --prefix ${prefix} --verbose 50 --cuda_ids ${cuda_ids}"

# [pick an epsilon]
# linf ball epsilons
eps=0.1
width_num=16
num_save=5
##############################I may change the proj number, lr, roubst training function to not include bound!
batch_size=30
lr=1e-4


# all remaining experiments use an approximation for training with 20 projections
parameters="--proj 20 --norm_train ${norm_type} --width_num ${width_num}  --num_save ${num_save} --batch_size ${batch_size} ${parameters}"

python train_FMNIST.py --method 'baseline' --epsilon ${eps} --lr ${lr} --norm_test ${norm_type} --test_batch_size 5 ${parameters}

python train_FMNIST.py --epsilon ${eps} --lr ${lr} --norm_test ${norm_type} --test_batch_size 5 ${parameters}


