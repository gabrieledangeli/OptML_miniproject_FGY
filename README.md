# On GANs optimization: Competitive Gradient Descent  
In this repository you can find our code for the mini-project of the course [CS-439 "Optimization for Machine Learning](https://edu.epfl.ch/coursebook/en/optimization-for-machine-learning-CS-439).  
In this project we explored how the optimization of a Generative Adversarial Network works; in particular, we focused on a new approach based on the Competitive Gradient Descent algorithm introduced in [1]. This optimizer was shown to have really good experimental results on CIFAR10 dataset \[2\]; this suggests that taking in some way in account the update of the opponent in GANs (i.e. Generator for Discriminator and vice versa) outperform the usual training with independent updates (e.g. [3]\[4\]).  
We then reinforce this conjecture by analysing different training procedures for a GAN's optimization problem on another dataset, namely the MNIST dataset.

#### Team  
Our team is composed by:  
* D'Angeli Gabriele: [@gabrieledangeli](https://github.com/gabrieledangeli)    
* Di Gennaro Federico: [@FedericoDiGennaro](https://github.com/FedericoDiGennaro)    
* Yuzhu Tian: [@MileyTian](https://github.com/MileyTian)   

#### Reproducibility of the repo  
We worked with the libraries and their relative versions that you can find in the file requirements.txt  
Our code it is also build to run using GPU if available. Please notice that we take the WGAN implementation from [this repository](), and we used the library [here](https://github.com/devzhk/Implicit-Competitive-Regularization) to train our NNs with the Adaptive Competitive Gradient Descent (ACGD) algorithm.

#### Description of folders and files  
Here you can find a detailed description of what each file in this repository contains.  
* main.py:  

### References  
[1] Implicit competitive regularization in GANs. Florian Schaefer, Hongkai Zheng, Animashree Anandkumar Proceedings of the 37th International Conference on Machine Learning, PMLR 119:8533-8544, 2020.  
[2] 