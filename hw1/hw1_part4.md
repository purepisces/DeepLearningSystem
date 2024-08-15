## Question 6: SGD for a two-layer neural network [10 pts]

As you did in Homework 0, you will now implement stochastic gradient descent (SGD) for a simple two-layer neural network as defined in Question 5 of Homework 0.

Specifically, for input $x \in \mathbb{R}^n$, we'll consider a two-layer neural network (without bias terms) of the form

$$z = W_2^T \text{ReLU}(W_1^T x)$$


where $W_1 \in \mathbb{R}^{n \times d}$ and $W_2 \in \mathbb{R}^{d \times k}$ represent the weights of the network (which has a $d$-dimensional hidden unit), and where $z \in \mathbb{R}^k$ represents the logits output by the network. We again use the softmax / cross-entropy loss, meaning that we want to solve the optimization problem, overloading the notation to describe the batch form with matrix $X \in \mathbb{R}^{m \times n}$:


$$\min_{W_1, W_2} \;\; \ell_{\mathrm{softmax}}(\text{ReLU}(X W_1) W_2, y).$$

First, you will need to implement the forward and backward passes of the `relu` operator.

1. Begin by filling out the function `ReLU` operator in `python/needle/ops/ops_mathematic.py`.

2. Then fill out the `gradient` function of the class `ReLU` in `python/needle/ops/ops_mathematic.py`. **Note that in this one case it's acceptable to access the `.realize_cached_data()` call on the output tensor, since the ReLU function is not twice differentiable anyway**.

Then,

3. Fill out the `nn_epoch` method in the `apps/simple_ml.py` file.

Again, you can use your solution in Homework 0 for the `nn_epoch` function as a starting point. Note that unlike in Homework 0, the inputs `W1` and `W2` are `Tensors`. Inputs `X` and `y` however are still numpy arrays - you should iterate over minibatches of the numpy arrays `X` and `y` as you did in Homework 0, and then cast each `X_batch` as a `Tensor`, and one hot encode `y_batch` and cast as a `Tensor`. While last time we derived the backpropagation equations for this two-layer ReLU network directly, this time we will be using our autodifferentiation engine to compute the gradients generically by calling the `.backward()` method of the `Tensor` class. For each minibatch, after calling `.backward`, you should compute the updated values for `W1` and `W2` in `numpy`, and then create new `Tensors` for `W1` and `W2` with these `numpy` values. Your solution should return the final `W1` and `W2`  `Tensors`.

