**I still haven't understand matmul backward pass, summation backward pass, broadcastto backward pass, reshape backward pass, negate backward pass, transpose backward pass.**

Hw1: This homework will get you started with your implementation of the **needle** (**ne**cessary **e**lements of **d**eep **le**arning) library that you will develop throughout this course.  In particular, the goal of this assignment is to build a basic **automatic differentiation** frameowrk, then use this to re-implement the simple two-layer neural network you used for the MNIST digit classification problem in HW0.

For an introduction to the `needle` framework, refer to Lecture 5 in class and [this Jupyter notebook](https://github.com/dlsys10714/notebooks/blob/main/5_automatic_differentiation_implementation.ipynb) from the lecture. For this homework, you will be implementing the basics of automatic differentiation using a `numpy` CPU backend (in later assignments, you will move to your own linear algebra library including GPU code). All code for this assignment will be written in Python.

For the purposes of this assignment, there are two important files in the `needle` library, the `python/needle/autograd.py` file (which defines the basics of the computational graph framework, and also will form the basis of the automatic differentation framework), and the `python/needle/ops/ops_mathematic.py`.file (which contains implementations of various operators that you will use implement throughout the assignment and the course).

HW1 expands on HW0 by implementing forward and backward computations for various operators within a computational graph framework, enabling automatic differentiation. And also implement a topological sort for reverse-mode backpropagation, adapt the softmax loss function for tensors, and train a two-layer neural network using stochastic gradient descent (SGD). 

- **hw1_combined.ipynb**: The original Jupyter notebook.
- **hw1_part1.md**: homework explanation
- **hw1_part2.md**: homework explanation
- **hw1_tech.md**: homework python technical
- **hw1_original**: This folder contains the original homework without solutions. It includes the initial code and the original notebook to help you get started. **You just need the hw1-main.zip and hw1_combined.ipynb to get start from scratch.**
- **hw1_solution**: Completed solution.
- **xxxx.pdf**: Slides in course.
- **xxxx.pdf**: Slides in course.

