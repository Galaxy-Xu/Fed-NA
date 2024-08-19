## Fed-NA: Federated Learning Noise Allocation Algorithm

## Description
This is the official implementation for our paper [Defending Federated Learning Against Gradient Leakage Attack with Noise Allocation]. 
We have submitted our paper for publication and it is under review. The source codes will be updated here once the paper is accepted.

## Abstract
Federated learning serves as a widely adopted secure distributed learning framework, facing the threat of gradient leakage attacks. As an efficient privacy-preserving technology, the noise-adding mechanism is typically introduced to safeguard data privacy. However, the introduction of noise inevitably results in a reduction in the training accuracy of the global model. Existing methods mainly focus on how to compensate the training accuracy loss caused by noise. In order to mitigate the impact of noise during the noise addition phase rather than compensating for accuracy after the introduction of noise, we propose a Federated Learning Noise Allocation (Fed-NA) algorithm based on a double-noise-adding mechanism. The double-noise-adding mechanism adds zero-sum noise to the convolutional layer of gradient and adds differential privacy noise to the fully connected layer of gradient, where the noise has less influence on the model’s training accuracy. Furthermore, we allocate zero-sum noise to the convolutional layer of the gradient information based on the client’s privacy leakage weight. The stronger zero-sum noise is allocated to the more vulnerable client, which enhances the defense ability of the client. We rigorously analyze the convergence when applying Fed-NA, and the result shows that our proposed Fed-NA eventually converges with the increase in training rounds. We carry out experiments on the federated learning hardware platform and use simulation experiments to verify the scheme. The results show that our proposed algorithm can improve accuracy by up to 12% and enhance resistance capability by up to 10% compared to existing methods.

## Models
## Prerequisites
- Pytorch >= 1.12.0
- Tensorflow >= 2.3.0
- Python >= 3.7
- NVIDIA GPU + CUDA


## Model and Datasets
- datasets referenced in this paper
  - [MNIST](https://www.tensorflow.org/datasets/catalog/mnist)
  - [CIFAR-100](https://www.tensorflow.org/datasets/catalog/cifar100)
  - [FASHION-MNIST](https://www.tensorflow.org/datasets/catalog/fashion_mnist)

- model referenced in this paper
  - LeNet -> convolutional layer 1 (3 × 6 × 5), convolutional layer 2 (6 × 16 × 5), fully connected layer 1 (400 × 120), fully connected layer 2 (120 × 84) and fully connected layer 3 (84 × class_num).
  
  - MLP ->  input layer (784 × 256), first hidden layer (256 × 64), second hidden layer (64 × 10) and output layer (10)


## Running and Configuration
- configuration in this paper
  - Fed-NA_test_accuracy
    For configuration of hyperparameters, change at /utils/options

  - Fed-NA_defense_ability
    For configuration of parameters, change at /main/GRNN.py    
  
- running the code
  - Fed-NA_test_accuracy
    Run /main.py to start the code, the result will be stored in /log/file
    
  - Fed-NA_defense_ability
    Run /GRNN-main/GRNN.py to start the code, the result will be stored in /Results
    

