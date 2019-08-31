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
cuda_ids=5,6,7
epochs=181
schedule_length=80

# L2 ball arguments
norm_type=l1_median
norm_eval=l1

# FMNIST parameters
prefix="./model"
starting_epsilon=0.0001
parameters="--epochs ${epochs} --starting_epsilon ${starting_epsilon} --schedule_length ${schedule_length} --prefix ${prefix} --verbose 20 --cuda_ids ${cuda_ids}"

# [pick an epsilon]
# linf ball epsilons
eps=0.06275
width_num=2
num_save=5
##############################I may change the proj number, lr, roubst training function to not include bound!
batch_size=20
lr=1e-4
# eps=0.3

# all remaining experiments use an approximation for training with 50 projections
parameters="--proj 50 --norm_train ${norm_type} --width_num ${width_num}  --num_save ${num_save} --batch_size ${batch_size} ${parameters}"

python train_Yale.py --method 'baseline' --lr ${lr} --epsilon ${eps} --norm_test ${norm_eval} --test_batch_size 50 ${parameters}

python train_Yale.py --epsilon ${eps} --lr ${lr} --norm_test ${norm_type} --test_batch_size 5 ${parameters}


