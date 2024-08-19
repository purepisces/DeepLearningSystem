Emergency To do:

1. Check summation gradient make hw1's solution to hw2's summation

Make all the axes to be tuples, such like in summation, remove the check if isinstance(self.axes, int):, because according to the notebook, the axes should be tuples.

# Deep Learning System

This repository contains a series of homework assignments for the CMU Deep Learning System course. There are 4 homework assignments in total, from hw0 to hw4.

## Homework Assignments Overview

### hw0
**I still haven't finish question 6, the c++ part.**

Hw0 includes functions to read and parse the MNIST dataset, compute softmax loss, and perform Stochastic Gradient Descent (SGD) for both softmax regression and a simple two-layer neural network, providing a foundational framework for the MNIST digit classification problem.

Pay attention to 
- **src/simple.ml.py**

### hw1

**I still haven't understand matmul backward pass, summation backward pass, broadcastto backward pass, reshape backward pass, negate backward pass, transpose backward pass.**

HW1 expands on HW0 by implementing forward and backward computations for various operators within a computational graph framework, enabling automatic differentiation. And also implement a topological sort for reverse-mode backpropagation, adapt the softmax loss function for tensors, and train a two-layer neural network using stochastic gradient descent (SGD). 

Pay attention to 
- **python/needle/ops/ops_mathematic.py**

