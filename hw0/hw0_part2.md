Refer to slides - [CMU 10714 Manual-Neural-Nets PDF](manual_neural_nets.pdf)

## Question 5: SGD for a two-layer neural network

Now that you've written SGD for a linear classifier, let's consider the case of a simple two-layer neural network. Specifically, for input $x \in \mathbb{R}^n$, we'll consider a two-layer neural network (without bias terms) of the form

$$\begin{equation}
z = W_2^T \mathrm{ReLU}(W_1^T x)
\end{equation}$$

where $W_1 \in \mathbb{R}^{n \times d}$ and $W_2 \in \mathbb{R}^{d \times k}$ represent the weights of the network (which has a $d$-dimensional hidden unit), and where $z \in \mathbb{R}^k$ represents the logits output by the network. We again use the softmax / cross-entropy loss, meaning that we want to solve the optimization problem

$$\begin{equation}
minimize_{W_1, W_2} \;\; \frac{1}{m} \sum_{i=1}^m \ell_{\mathrm{softmax}}(W_2^T \mathrm{ReLU}(W_1^T x^{(i)}), y^{(i)}).
\end{equation}$$

Or alternatively, overloading the notation to describe the batch form with matrix $X \in \mathbb{R}^{m \times n}$, this can also be written

$$\begin{equation}
minimize_{W_1, W_2} \;\; \ell_{\mathrm{softmax}}(\mathrm{ReLU}(X W_1) W_2, y).
\end{equation}$$

Using the chain rule, we can derive the backpropagation updates for this network (we'll briefly cover these in class, on 9/8, but also provide the final form here for ease of implementation). Specifically, let

$$\begin{equation}
\begin{split}
Z_1 \in \mathbb{R}^{m \times d} & = \mathrm{ReLU}(X W_1) \\
G_2 \in \mathbb{R}^{m \times k} & = normalize(\exp(Z_1 W_2)) - I_y \\
G_1 \in \mathbb{R}^{m \times d} & = \mathrm{1}\{Z_1 > 0\} \circ (G_2 W_2^T)
\end{split}
\end{equation}$$

where $\mathrm{1}\{Z_1 > 0\}$ is a binary matrix with entries equal to zero or one depending on whether each term in $Z_1$ is strictly positive and where $\circ$ denotes elementwise multiplication. Then the gradients of the objective are given by

$$\begin{equation}
\begin{split}
\nabla_{W_1} \ell_{\mathrm{softmax}}(\mathrm{ReLU}(X W_1) W_2, y) & = \frac{1}{m} X^T G_1 \\
\nabla_{W_2} \ell_{\mathrm{softmax}}(\mathrm{ReLU}(X W_1) W_2, y) & = \frac{1}{m} Z_1^T G_2. \\
\end{split}
\end{equation}$$

  

**Note:** If the details of these precise equations seem a bit cryptic to you (prior to the 9/8 lecture), don't worry too much. These _are_ just the standard backpropagation equations for a two-layer ReLU network: the $Z_1$ term just computes the "forward" pass while the $G_2$ and $G_1$ terms denote the backward pass. But the precise form of the updates can vary depending upon the notation you've used for neural networks, the precise ways you formulate the losses, if you've derived these previously in matrix form, etc. If the notation seems like it might be familiar from when you've seen deep networks in the past, and makes more sense after the 9/8 lecture, that is more than sufficient in terms of background (after all, the whole _point_ of deep learning systems, to some extent, is that we don't need to bother with these manual calculations). But if these entire concepts are _completely_ foreign to you, then it may be better to take a separate course on ML and neural networks prior to this course, or at least be aware that there will be substantial catch-up work to do for the course.

Using these gradients, now write the `nn_epoch()` function in the `src/simple_ml.py` file. As with the previous question, your solution should modify the `W1` and `W2` arrays in place. After implementing the function, run the following test. Be sure to use matrix operations as indicated by the expresssions above to implement the function: this will be _much_ faster, and more efficient, than attempting to use loops (and it requires far less code).

------------------------------------
### Key Terms

- **Logits**: These are the **raw, unnormalized scores output by a neural network's final layer before applying an activation function like softmax**. Logits are used to compute probabilities.
- **Probabilities**: These are the normalized scores obtained by applying the softmax function to the logits. They represent the predicted probabilities for each class.

### Layers in the Two-Layer (layers with trainable parameters (i.e., weights)) Neural Network

Network Structure: **Input Layer**, **First Layer: Hidden Layer (Input to Hidden)**, **Second Layer: Output Layer (Hidden to Output)**.

#### Input to Hidden Layer

- **Weight Matrix**: 
  $W_1 \in \mathbb{R}^{n \times d}$ where $n$ is the number of input features and $d$ is the number of hidden units.
- **Hidden Layer Activations**: 
  $Z_1 = \mathrm{ReLU}(X W_1)$
  - $X \in \mathbb{R}^{m \times n}$ is the input matrix.
  - $Z_1 \in \mathbb{R}^{m \times d}$ is the output of the hidden layer after applying the ReLU activation function. These are the activations, not logits or probabilities.

#### Hidden to Output Layer

- **Weight Matrix**: 
  $W_2 \in \mathbb{R}^{d \times k}$ where $d$ is the number of hidden units and $k$ is the number of output classes.
- **Output Logits**: 
  $Z_2 = Z_1 W_2$
  - $Z_2 \in \mathbb{R}^{m \times k}$ are the logits for the output layer.

#### Softmax Probabilities

- **Probabilities**: 
$P = \frac{\exp(Z_2)}{\sum \exp(Z_2)}$
  - $P \in \mathbb{R}^{m \times k}$ are the probabilities after applying the softmax function to the logits.

### Summary

- $Z_1 = \mathrm{ReLU}(X W_1)$ are the activations of the hidden layer after applying the ReLU function.
- $Z_2 = Z_1 W_2$ are the logits of the output layer.
- $P = \frac{\exp(Z_2)}{\sum \exp(Z_2)}$ are the probabilities after applying the softmax function to the logits $Z_2$.

## Backpropagation Overview
Backpropagation is the algorithm used to calculate the gradient of the loss function with respect to each parameter (weight) in a neural network. It allows the network to update these weights in a way that minimizes the loss function, enabling the network to learn from the data.

#### Forward Pass:
- Input data passes through the network, and the output is calculated.
- Activations and intermediate values (such as $Z_i$) are stored for use in the backward pass.

#### Backward Pass:
- The loss is computed using the output of the network and the true labels.
- Gradients of the loss with respect to each parameter are calculated using the chain rule of calculus.
- These gradients are used to update the parameters (weights) using an optimization algorithm like gradient descent.

### Backpropagation with Respect to $Z_i$ and $W_i$

In backpropagation, we need to calculate gradients with respect to both the activations $Z_i$ and the weights $W_i$. Here’s how this is done:

#### Gradients with Respect to Activations $Z_i$
The gradients with respect to the activations $Z_i$ are intermediate steps in the backpropagation process. They are used to calculate the gradients with respect to the weights. Specifically, for layer $i$, we compute $\frac{\partial \ell}{\partial Z_i}$, which represents how the loss changes with respect to the activations of that layer.

#### Gradients with Respect to Weights $W_i$
The ultimate goal is to compute the gradients of the loss with respect to the weights $W_i$, denoted as $\frac{\partial \ell}{\partial W_i}$. These gradients tell us how to adjust the weights to minimize the loss.

## Neural networks in machine learning

Recall that neural networks just specify one of the "three" ingredients of a machine learning algorithm, also need:
- Loss function: still cross entropy loss, like last time
- Optimization procedure: still SGD, like last time

In other words, we still want to solve the optimization problem

$$\min_{\theta} \frac{1}{m} \sum_{i=1}^{m} \ell_{ce}(h_{\theta}(x^{(i)}), y^{(i)})$$

using SGD, just with $h_{\theta}(x)$ now being a neural network.

Requires computing the gradients $\nabla_{\theta} \ell_{ce}(h_{\theta}(x^{(i)}), y^{(i)})$ for each element of $\theta$.

## The gradient(s) of a two-layer network part1

Let's work through the derivation of the gradients for a simple two-layer network, written in batch matrix form, i.e.,

$$\nabla_{\{W_1, W_2\}} \ell_{ce}(\sigma(XW_1)W_2, y)$$

The gradient w.r.t. $W_2$ looks identical to the softmax regression case:

$$\frac{\partial \ell_{ce}(\sigma(XW_1)W_2, y)}{\partial W_2} = \frac{\partial \ell_{ce}}{\partial \sigma(XW_1)W_2} \cdot \frac{\partial \sigma(XW_1)W_2}{\partial W_2}
= (S - I_y) \cdot \sigma(XW_1), \quad [S = \text{softmax}(\sigma(XW_1)W_2)]$$

so (matching sizes) the gradient is

$$\nabla_{W_2} \ell_{ce}(\sigma(XW_1)W_2, y) = \sigma(XW_1)^T (S - I_y)$$


> **Prove  $S - I_y$**
> 
> **1. First Prove $\ell_{ce}(h(x), y) = -h_y(x) + \log \sum_{j=1}^{k} \exp(h_j(x))$**
> 
> Let's convert the hypothesis function to a "probability" by exponentiating and normalizing its entries (to make them all positive and sum to one)
> 
> $$z_i = p(\text{label} = i) = \frac{\exp(h_i(x))}{\sum_{j=1}^{k} \exp(h_j(x))} \quad \Longleftarrow \quad z \equiv \text{softmax}(h(x))$$
> 
> Then let's define a loss to be the (negative) log probability of the true class: this is called softmax or cross-entropy loss
> 
> $$\ell_{ce}(h(x), y) = -\log p(\text{label} = y) = -h_y(x) + \log \sum_{j=1}^{k} \exp(h_j(x))$$
> 
> **2. Second prove $S - I_y$**
> 
>Let's start by deriving the gradient of the softmax loss itself: for vector $h \in \mathbb{R}^k$
> 
>$$\frac{\partial \ell_{ce} (h, y)}{\partial h_i} = \frac{\partial}{\partial h_i} \left( -h_y + \log \sum_{j=1}^{k} \exp h_j \right)$$
>$$= -1 \{ i = y \} + \frac{\exp h_i}{\sum_{j=1}^{k} \exp h_j}$$
> 
>So, in vector form:
> 
>$$\nabla_h \ell_{ce} (h, y) = z - e_y, \text{ where } z = \text{softmax}(h)$$
> 
>In “matrix batch” form:
> 
>$$\nabla_h \ell_{ce} (X\theta, y) = S - I_y, \text{ where } S = \text{softmax}(X\theta)$$

## The gradient(s) of a two-layer network part2

Deep breath and let's do the gradient w.r.t. \( W_1 \)...

$$\frac{\partial \ell_{ce} (\sigma(XW_1) W_2, y)}{\partial W_1} = \frac{\partial \ell_{ce} (\sigma(XW_1) W_2, y)}{\partial \sigma(XW_1) W_2} \cdot \frac{\partial \sigma(XW_1) W_2}{\partial \sigma(XW_1)} \cdot \frac{\partial \sigma(XW_1)}{\partial XW_1} \cdot \frac{\partial XW_1}{\partial W_1}$$

$$= (S - I_y) \cdot W_2 \cdot \sigma'(XW_1) \cdot X$$

and so the gradient is

$$\nabla_{W_1} \ell_{ce} (\sigma(XW_1) W_2, y) = X^T \left( (S - I_y) W_2^T \circ \sigma'(XW_1) \right)$$

where $\circ$ denotes elementwise multiplication

$$\sigma'(x) = \begin{cases}
1 & \text{if } x > 0 \\
0 & \text{if } x \leq 0
\end{cases}$$

## Backpropagation "in general"

> 🌟🌟🌟🌟🌟🌟 In the slides explanation, the  neural network is Z1->W1->Z2->W2->Z3, however, in the assignment, the neural network is X->W1->Z1->W2->Z2. So the formula will be index different, but the logic is same. For example in assignment, $G_1 \in \mathbb{R}^{m \times d}  = \mathrm{1}\{Z_1 > 0\} \circ (G_2 W_2^T)$, but in slides $G_i$ is $G_{i+1}W_i$.

There is a method to this madness ... consider our fully-connected network:

$$ Z_{i+1} = \sigma_i(Z_i W_i), \quad i = 1, \ldots, L $$

Then (now being a bit terse with notation)

$$ \frac{\partial \ell(Z_{L+1}, y)}{\partial W_i} = \frac{\partial \ell}{\partial Z_{L+1}} \cdot \frac{\partial Z_{L+1}}{\partial Z_L} \cdot \frac{\partial Z_L}{\partial Z_{L-1}} \cdot \ldots \cdot \frac{\partial Z_{i+2}}{\partial Z_{i+1}} \cdot \frac{\partial Z_{i+1}}{\partial W_i} $$

$$ G_{i+1} =\frac{\partial \ell}{\partial Z_{L+1}} \cdot \frac{\partial Z_{L+1}}{\partial Z_L} \cdot \frac{\partial Z_L}{\partial Z_{L-1}} \cdot \ldots \cdot \frac{\partial Z_{i+2}}{\partial Z_{i+1}} $$

Then we have a simple "backward" iteration to compute the $G_i$'s

$$ G_i = G_{i+1} \cdot \frac{\partial Z_{i+1}}{\partial Z_i} = G_{i+1} \cdot \frac{\partial \sigma_i(Z_i W_i)}{\partial Z_i W_i} \cdot \frac{\partial Z_i W_i}{\partial Z_i} = G_{i+1} \cdot \sigma'(Z_i W_i) \cdot W_i $$

## Computing the real gradients

To convert these quantities to "real" gradients, consider matrix sizes

$$G_i = \frac{\partial \ell(Z_{L+1}, y)}{\partial Z_i} = \nabla_{Z_i} \ell(Z_{L+1}, y) \in \mathbb{R}^{m \times n_i}$$

so with "real" matrix operations

$$G_i = G_{i+1} \cdot \sigma'(Z_i W_i) \cdot W_i = \left( G_{i+1} \circ \sigma'(Z_i W_i) \right) W_i^T$$

Similar formula for actual parameter gradients $\nabla_{W_i} \ell(Z_{L+1}, y) \in \mathbb{R}^{n_i \times n_{i+1}}$

$$\frac{\partial \ell(Z_{L+1}, y)}{\partial W_i} = G_{i+1} \cdot \frac{\partial \sigma_i(Z_i W_i)}{\partial Z_i W_i} \cdot \frac{\partial Z_i W_i}{\partial W_i} = G_{i+1} \cdot \sigma'(Z_i W_i) \cdot Z_i$$

$$\implies \nabla_{W_i} \ell(Z_{L+1}, y) = Z_i^T \left( G_{i+1} \circ \sigma'(Z_i W_i) \right)$$

> 🌟🌟🌟🌟🌟🌟 Note that in the assignment, $G_1 \in \mathbb{R}^{m \times d}  = \mathrm{1}\{Z_1 > 0\} \circ (G_2 W_2^T)$. This is because in the slides explanation, the  neural network is Z1->W1->Z2->W2->Z3, however, in the assignment, the neural network is X->W1->Z1->W2->Z2. So $G_i$ is not $G_{i+1}W_i$ but $G_{i+1} W_{i+1}$

## Backpropagation: Forward and backward passes

Putting it all together, we can efficiently compute all the gradients we need for a neural network by following the procedure below

**Forward pass**
1. Initialize: $Z_1 = X$
   
   Iterate: $Z_{i+1} = \sigma_i(Z_i W_i), \quad i = 1, \ldots, L$

**Backward pass}**

2. Initialize: $G_{L+1} = \nabla_{Z_{L+1}} \ell(Z_{L+1}, y) = S - I_y$
   
   Iterate: $G_i = \left( G_{i+1} \circ \sigma'_i(Z_i W_i) \right) W_i^T, \quad i = L, \ldots, 1$

And we can compute all the needed gradients along the way

$$\nabla_{W_i} \ell(Z_{L+1}, y) = Z_i^T \left( G_{i+1} \circ \sigma'_i(Z_i W_i) \right)$$

"Backpropagation" is just chain rule + intelligent caching of intermediate results


## A closer look at these operations

What is really happening with the backward iteration?

$$\frac{\partial \ell(Z_{L+1}, y)}{\partial W_i} = \frac{\partial \ell}{\partial Z_{L+1}} \cdot \frac{\partial Z_{L+1}}{\partial Z_L} \cdot \ldots \cdot \frac{\partial Z_{i+2}}{\partial Z_{i+1}} \cdot \frac{\partial Z_{i+1}}{\partial W_i}$$

$$ G_{i+1} =\frac{\partial \ell}{\partial Z_{L+1}} \cdot \frac{\partial Z_{L+1}}{\partial Z_L} \cdot \frac{\partial Z_L}{\partial Z_{L-1}} \cdot \ldots \cdot \frac{\partial Z_{i+2}}{\partial Z_{i+1}} $$

Each layer needs to be able to multiply the "incoming backward" gradient $G_{i+1}$ by its derivatives, $\frac{\partial Z_{i+1}}{\partial W_i}$, an operation called the "vector Jacobian product."

This process can be generalized to arbitrary computation graphs: this is exactly the process of automatic differentiation we will discuss in the next lecture.

## Code Implementation
```python
def nn_epoch(X, y, W1, W2, lr = 0.1, batch=100):
    """ Run a single epoch of SGD for a two-layer neural network defined by the
    weights W1 and W2 (with no bias terms):
        logits = ReLU(X * W1) * W2
    The function should use the step size lr, and the specified batch size (and
    again, without randomizing the order of X).  It should modify the
    W1 and W2 matrices in place.

    Args:
        X (np.ndarray[np.float32]): 2D input array of size
            (num_examples x input_dim).
        y (np.ndarray[np.uint8]): 1D class label array of size (num_examples,)
        W1 (np.ndarray[np.float32]): 2D array of first layer weights, of shape
            (input_dim, hidden_dim)
        W2 (np.ndarray[np.float32]): 2D array of second layer weights, of shape
            (hidden_dim, num_classes)
        lr (float): step size (learning rate) for SGD
        batch (int): size of SGD minibatch

    Returns:
        None
    """
    ### BEGIN YOUR CODE
    num_examples = X.shape[0]
    num_classes = W2.shape[1]

    for start in range(0, num_examples, batch):
        end = min(start + batch, num_examples)
        X_batch = X[start:end]
        y_batch = y[start:end]

        # Forward pass: Compute Z1 and Z2 (the output logits)
        Z1 = np.maximum(0, X_batch @ W1)  # ReLU activation
        Z2 = Z1 @ W2

        # Compute softmax probabilities
        exp_logits = np.exp(Z2)
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

        # Create a one-hot encoded matrix of the true labels
        I_y = np.zeros((len(y_batch), num_classes))
        I_y[np.arange(len(y_batch)), y_batch] = 1

        # Backward pass: Compute gradients G2 and G1
        G2 = probs - I_y
        G1 = (Z1 > 0).astype(np.float32) * (G2 @ W2.T)
        # Compute the gradients for W1 and W2
        grad_W1 = X_batch.T @ G1 / batch
        grad_W2 = Z1.T @ G2 / batch

        # Perform the gradient descent step(Update the weights)
        W1 -= lr * grad_W1
        W2 -= lr * grad_W2
    ### END YOUR CODE
```

-----------------------
### Training a full neural network

```python3
def train_nn(X_tr, y_tr, X_te, y_te, hidden_dim = 500,
             epochs=10, lr=0.5, batch=100):
    """ Example function to train two layer neural network """
    # X_tr.shape[1]: the number of features in the training data
    # y_tr.max()+1 : the number of classes
    n, k = X_tr.shape[1], y_tr.max() + 1
    # np.random.seed(0): This sets the random seed for reproducibility. By setting the seed, you ensure that every time you run the code, the same random numbers are generated, which means that the results will be the same.
    # W1: Initializes the weights for the first layer with random values, scaled by the square root of the hidden dimension.
    # W2: Initializes the weights for the second layer with random values, scaled by the square root of the number of classes.
    np.random.seed(0)
    W1 = np.random.randn(n, hidden_dim).astype(np.float32) / np.sqrt(hidden_dim)
    W2 = np.random.randn(hidden_dim, k).astype(np.float32) / np.sqrt(k)

    print("| Epoch | Train Loss | Train Err | Test Loss | Test Err |")
    for epoch in range(epochs):
        nn_epoch(X_tr, y_tr, W1, W2, lr=lr, batch=batch)
        train_loss, train_err = loss_err(np.maximum(X_tr@W1,0)@W2, y_tr)
        test_loss, test_err = loss_err(np.maximum(X_te@W1,0)@W2, y_te)
        print("|  {:>4} |    {:.5f} |   {:.5f} |   {:.5f} |  {:.5f} |"\
              .format(epoch, train_loss, train_err, test_loss, test_err))

if __name__ == "__main__":
    X_tr, y_tr = parse_mnist("data/train-images-idx3-ubyte.gz",
                             "data/train-labels-idx1-ubyte.gz")
    X_te, y_te = parse_mnist("data/t10k-images-idx3-ubyte.gz",
                             "data/t10k-labels-idx1-ubyte.gz")

    print("\nTraining two layer neural network w/ 100 hidden units")
    train_nn(X_tr, y_tr, X_te, y_te, hidden_dim=100, epochs=20, lr = 0.2)
```
`np.random.seed(0)` is a function call that sets the seed for NumPy's random number generator. Setting the seed ensures that the sequence of random numbers generated by NumPy's random functions (such as `np.random.randn`, `np.random.rand`, etc.) is deterministic and reproducible.

Code
```python
import numpy as np

# Without setting the seed
print("Without setting the seed:")
print(np.random.randn(5))

# Setting the seed
np.random.seed(0)
print("\nWith setting the seed to 0:")
print(np.random.randn(5))

# Resetting the seed to 0
np.random.seed(0)
print("\nWith setting the seed to 0 again:")
print(np.random.randn(5))
```
Output
```python
Without setting the seed:
[ 0.12667243 -0.86820841  2.6421595   0.28067869 -1.27068428]

With setting the seed to 0:
[1.76405235 0.40015721 0.97873798 2.2408932  1.86755799]

With setting the seed to 0 again:
[1.76405235 0.40015721 0.97873798 2.2408932  1.86755799]
```
> Uniform Distribution (np.random.rand): Suitable for generating values with equal probability, useful for tasks requiring random sampling from a specific range.
>
> Normal Distribution (np.random.randn): Suitable for tasks requiring values that follow a bell curve, especially beneficial for initializing neural network weights due to its properties that help in better training dynamics. Symmetry Breaking: If all weights are initialized to the same value (or from a uniform distribution that doesn't spread well), neurons in the same layer might learn the same features, making the network less expressive. Normally distributed weights help break symmetry and ensure that neurons learn diverse features.

❓❓❓❓❓❓❓❓❓❓❓❓❓❓❓❓❓❓❓❓❓❓❓❓❓❓❓❓❓❓:
Have confusion for why W1 scaled by hidden_dim not n, and why W2 scaled by k not hidden_dim?
```python
W1 = np.random.randn(n, hidden_dim).astype(np.float32) / np.sqrt(hidden_dim)
W2 = np.random.randn(hidden_dim, k).astype(np.float32) / np.sqrt(k)
```



