# On GANs optimization: Competitive Gradient Descent  
In this repository you can find our code for the mini-project of the course [CS-439 - Optimization for Machine Learning](https://edu.epfl.ch/coursebook/en/optimization-for-machine-learning-CS-439).  
In this project we explored how the optimization of a Generative Adversarial Network works; in particular, we focused on a new approach based on the Competitive Gradient Descent algorithm introduced in [1]. This optimizer was shown to have really good experimental results on CIFAR10 dataset [2]; this suggests that taking in some way in account the update of the opponent in GANs (i.e. Generator for Discriminator and vice versa) outperform the usual training with independent updates.  
We then reinforce this conjecture by analysing different training procedures for a GAN's optimization problem on another dataset, namely the MNIST dataset.

<p align="center">
  <figure>
    <img src="https://github.com/gabrieledangeli/OptML_miniproject_FGY/blob/main/PART%202/results/mnist/WGAN/WGAN_generate_animation.gif" alt="First GIF" width="300" height="200"/>
    <figcaption>WGAN with ADAM.</figcaption>
  </figure>
  &nbsp;
  <figure>
    <img src="https://github.com/gabrieledangeli/OptML_miniproject_FGY/blob/main/PART%202/results/mnist/WGAN_ACGD/WGAN_ACGD_generate_animation.gif" alt="Second GIF" width="300" height="200"/>
    <figcaption>WGAN with CGD.</figcaption>
  </figure>
</p>

## Team  
Our team is composed by:  
* D'Angeli Gabriele: [@gabrieledangeli](https://github.com/gabrieledangeli)    
* Di Gennaro Federico: [@FedericoDiGennaro](https://github.com/FedericoDiGennaro)    
* Tian Yuzhu: [@MileyTian](https://github.com/MileyTian)   

## Reproducibility of the code
We worked with the libraries and their relative versions that you can find in the file `requirements.txt`.   
Our code it is also build to run using GPU if available. Please notice that we take the WGAN implementation from [this repository](), and we used the library [here](https://github.com/devzhk/Implicit-Competitive-Regularization) to train our NNs with the Adaptive Competitive Gradient Descent (ACGD) algorithm.

#### Description of folders and files  
Here you can find a detailed description of what each file in this repository contains.  
**PART 1 folder:** It contains the code necessary to obtain the results on the section *Competitive Gradient Descent* of the report.  
|--- `plots.py`: python file with the functions required for plotting the trajectories.   
|--- `main.ipynb`: notebook file that once runned call all the functions and it produces the trajectories of the optimization problems we discussed in this project.  
In PART 1 folder, there is also the **optimizers** folder, that contains the following files:   
|--- `optimizers_conc_conv.py`: python file for the concave-convex problem.   
|--- `optimizers_conv_conc.py`: python file for the convex-concave problem.  
|--- `optimizers_xy.py`: python file for the bilinear problem.  
**PART 2 folder:** It contains the code necessary to obtain the results on the section *CGD on GANs optimization* of the report.  
|--- `helpers.py`: python file for ausiliary functions needed to train the GANs and to all the images.  
|--- `WGAN.ipynb`: notebook file that once runned trains all the WGANs with the different
optimizers we discussed in the project.  
|--- `score_analysis.ipynb`: notebook file that once runned plots all the scores and loss functions in our report.  
|--- **results folder:** folder with the results of our trainings (generated imagese and GIFs) for both MNIST and CIFAR10.  
|--- **models folder:** folder with the numerical results (losses and scores) of our our trainings for both MNIST and CIFAR10.   

## References

#### Code:

#### Main theoretical references: 
[1] [Implicit competitive regularization in GANs, *Florian Schaefer, Hongkai Zheng, Animashree Anandkumar*. Proceedings of the 37th International Conference on Machine Learning, 2020.](http://proceedings.mlr.press/v119/schaefer20a.html)  
[2] [Competitive Gradient Descent, *Florian Schäfer, Anima Anandkumar*. NeurIPS, 2019.](https://arxiv.org/abs/1905.12103)
