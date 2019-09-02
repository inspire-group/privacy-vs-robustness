# privacy-vs-robustness

### About
This code accompanies the paper "Privacy Risks of Securing Machine Learning Models against Adversarial Examples", accepted by ACM CCS 2019
https://arxiv.org/abs/1905.10291.

We perform membership inference attacks against machine learning models which are trained to be robust against adversarial examples.  
In total, we evaluate the privacy leakage introduced by six state-of-the-art robust training algorithms: [PGD-based adversarial training](https://arxiv.org/abs/1706.06083), [distributional adversarial training](https://arxiv.org/abs/1710.10571), [difference-based adversarial training](https://arxiv.org/abs/1901.08573), [duality-based verification](https://arxiv.org/abs/1805.12514), [abstract interpretation-based verification](http://proceedings.mlr.press/v80/mirman18b.html), [interval bound propagation-based verification](https://arxiv.org/abs/1810.12715).  
We find out that robust training algorithms tend to increase the membership information leakage of trained models, compared to the natural training algorithm.

### Overview of the code
`inference_utils.py`: defined function of membership inference based on prediction confidence  
`utils.py`: defined function to prepare Yale Face dataset  
`membership_inference_results.ipynb`: lists membership inference results  
* *Inside the folder of each robust training method*  
  `output_utils.py`: defined function to obtain predictions of training test data, in both benign and adversarial settings  
  `README.md`: instructions on how to train a robust (or natural) classifier
   * *Inside the subfolder of each dataset*  
      `output_performance.ipynb`: obtains model predictions  

      
### Dependecies
Tensorflow-1.12; Pytorch-0.4

