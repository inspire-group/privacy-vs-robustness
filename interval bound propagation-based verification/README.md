### About
This defense method is proposed by [Gowal et al.](https://github.com/deepmind/interval-bound-propagation)

### Additional dependencies
dm-sonnet==1.26;  tensorflow-probability==0.5.0

### Model training inside Yale/FMNIST folder
`python train_natural.py`: trains a natural classifier  
`python train_robust.py`: trains a robust classifier

### Pretrain models  
[robust Yale Face classifier](http://www.princeton.edu/~liweis/privacy-vs-robustness/IBP_based_verify_yale_robust.zip)   
[robust Fashion MNIST classifier](http://www.princeton.edu/~liweis/privacy-vs-robustness/IBP_based_verify_fmnist_robust.zip)
