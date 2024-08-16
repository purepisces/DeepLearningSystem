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

---------------------------------------------------------

## `ReLU`: Rectified Linear Unit Activation Function

**Example**

**Forward Pass**

If you have the following `ndarray`:

- **Ndarray `a`**: `np.array([-2, -1, 0, 1, 2])`

The ReLU activation would result in:

- **Result**: `np.array([0, 0, 0, 1, 2])`

The ReLU function outputs the input directly if it is positive; otherwise, it outputs zero.

**Backward Pass**

During the backward pass, you want to calculate the gradient of the loss $\ell$ with respect to the input `a` of the `ReLU` operation.

- `out_grad` represents $\frac{\partial \ell}{\partial f}$, which is the gradient of the loss $\ell$ with respect to the output $f$ of the `ReLU` operation.
- The derivative of the ReLU function is defined as:
  $$\frac{\partial f}{\partial a} = \begin{cases} 
  1 & \text{if } a > 0 \\
  0 & \text{if } a \leq 0 
  \end{cases}$$

Combining these using the chain rule:

- $$\frac{\partial \ell}{\partial a} = \frac{\partial \ell}{\partial f} \cdot \frac{\partial f}{\partial a} = \text{outgrad} \cdot \text{ReLU}'(a)$$

Where `ReLU'(a)` is `1` where `a > 0` and `0` where `a <= 0`.

Code Implementation

```python
class ReLU(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.maximum(a, 0)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        # Get the input tensor's data as a NumPy array
        a = node.inputs[0].realize_cached_data()
    
        # Create a mask where elements are 1 if the corresponding element in 'a' is > 0, otherwise 0
        relu_grad = Tensor((a > 0).astype(array_api.float32))
    
        # Multiply the incoming gradient (Tensor) by the ReLU gradient (also a Tensor)
        return out_grad * relu_grad
        ### END YOUR SOLUTION


def relu(a):
    return ReLU()(a)
```
## nn_epoch
**Code Implementation**
```python3
def nn_epoch(X, y, W1, W2, lr=0.1, batch=100):
    """Run a single epoch of SGD for a two-layer neural network defined by the
    weights W1 and W2 (with no bias terms):
        logits = ReLU(X * W1) * W1
    The function should use the step size lr, and the specified batch size (and
    again, without randomizing the order of X).

    Args:
        X (np.ndarray[np.float32]): 2D input array of size
            (num_examples x input_dim).
        y (np.ndarray[np.uint8]): 1D class label array of size (num_examples,)
        W1 (ndl.Tensor[np.float32]): 2D array of first layer weights, of shape
            (input_dim, hidden_dim)
        W2 (ndl.Tensor[np.float32]): 2D array of second layer weights, of shape
            (hidden_dim, num_classes)
        lr (float): step size (learning rate) for SGD
        batch (int): size of SGD mini-batch

    Returns:
        Tuple: (W1, W2)
            W1: ndl.Tensor[np.float32]
            W2: ndl.Tensor[np.float32]
    """

    ### BEGIN YOUR SOLUTION
    num_examples = X.shape[0]
    num_classes = W2.shape[1]

    for start in range(0, num_examples, batch):
        end = min(start + batch, num_examples)
        X_batch = X[start:end]
        y_batch = y[start:end]

        # Convert numpy arrays to tensors
        X_batch_tensor = ndl.Tensor(X_batch)
        y_batch_tensor = ndl.Tensor(y_batch)

        # Forward pass: Compute logits
        Z1 = ndl.relu(ndl.matmul(X_batch_tensor, W1))
        Z2 = ndl.matmul(Z1, W2)

        # Create a one-hot encoded matrix of the true labels using NumPy
        I_y_np = np.zeros((len(y_batch), num_classes))
        I_y_np[np.arange(len(y_batch)), y_batch] = 1

        # Convert to ndl.Tensor
        I_y = ndl.Tensor(I_y_np)

        # Compute softmax loss
        loss = softmax_loss(Z2, I_y)

        # Backward pass: Compute gradients
        loss.backward()

        # Update the weights using gradient descent
        new_W1 = ndl.Tensor(W1.numpy() - lr * W1.grad.numpy())
        new_W2 = ndl.Tensor(W2.numpy() - lr * W2.grad.numpy())
        W1, W2 = new_W1, new_W2

    return (W1, W2)
    ### END YOUR SOLUTION
```
> No New Graph Node: The operation new_W1 = ndl.Tensor(W1.numpy() - lr * W1.grad.numpy()) does not create a new node in the computational graph. Instead, it creates a new tensor that is detached from the graph, ensuring that subsequent operations on this tensor do not include the history of past computations.
>
> Incorrect W1 = W1 - lr * W1.grad: Directly performing operations like W1 = W1 - lr * W1.grad may seem equivalent to W1 = ndl.Tensor(W1.numpy() - lr * W1.grad.numpy()), but in some frameworks, the former may not be allowed or could cause unintended side effects. Specifically, in-place operations could cause issues when gradients are being tracked for backpropagation. And in many deep learning frameworks, such as PyTorch, TensorFlow, or potentially ndl, tensors are immutable, meaning that once a tensor is created, its data cannot be modified in place. Instead, any operation that would change a tensor's data creates a new tensor with the updated values.

### Difference between hw0 and hw1

In hw0, the nn_epoch function cannot directly use the softmax_loss function, but in hw1, the nn_epoch function can directly use the softmax_loss function.

Code in hw0
```python3
def softmax_loss(Z, y):
    """ Return softmax loss.  Note that for the purposes of this assignment,
    you don't need to worry about "nicely" scaling the numerical properties
    of the log-sum-exp computation, but can just compute this directly.

    Args:
        Z (np.ndarray[np.float32]): 2D numpy array of shape
            (batch_size, num_classes), containing the logit predictions for
            each class.
        y (np.ndarray[np.uint8]): 1D numpy array of shape (batch_size, )
            containing the true label of each example.

    Returns:
        Average softmax loss over the sample.
    """
    ### BEGIN YOUR CODE
    # Formula for one training sample: \begin{equation} \ell_{\mathrm{softmax}}(z, y) = \log\sum_{i=1}^k \exp z_i - z_y. \end{equation}
    
    # Compute the log of the sum of exponentials of logits for each sample
    log_sum_exp = np.log(np.sum(np.exp(Z), axis = 1))
    # Extract the logits corresponding to the true class for each sample
    # np.arange(Z.shape[0]) generates array [0, 1, 2, ..., batch_size-1]
    # Z[np.arange(Z.shape[0]), y] = Z[[row_indices], [col_indices]]
    # This selects the logits Z[i, y[i]] for each i which is each row
    correct_class_logits = Z[np.arange(Z.shape[0]), y]
    losses = log_sum_exp - correct_class_logits
    return np.mean(losses)
    ### END YOUR CODE

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
Code in hw1
```python3
def softmax_loss(Z, y_one_hot):
    """Return softmax loss.  Note that for the purposes of this assignment,
    you don't need to worry about "nicely" scaling the numerical properties
    of the log-sum-exp computation, but can just compute this directly.

    Args:
        Z (ndl.Tensor[np.float32]): 2D Tensor of shape
            (batch_size, num_classes), containing the logit predictions for
            each class.
        y (ndl.Tensor[np.int8]): 2D Tensor of shape (batch_size, num_classes)
            containing a 1 at the index of the true label of each example and
            zeros elsewhere.

    Returns:
        Average softmax loss over the sample. (ndl.Tensor[np.float32])
    """
    ### BEGIN YOUR SOLUTION
    # Compute the log-sum-exp for each row in Z, this will be a 1D tensor of shape (batch_size,)
    log_sum_exp = ndl.log(ndl.summation(ndl.exp(Z), axes=1))
    # Note ndl.exp(Z).sum(axes=1) and ndl.summation(ndl.exp(Z), axes=1) are functionally equivalent, where sum defined in line346 in autograd.py, inside class Tensor.

    # Extract the logits corresponding to the true class by multiplying Z with y_one_hot, result in 1D tensor of shape (batch_size,)
    correct_class_logits = ndl.summation(Z * y_one_hot, axes=1)

    # Compute the loss for each sample, result in a 1D tensor of shape (batch_size,)
    losses = log_sum_exp - correct_class_logits

    # Return the average loss across the batch
    return ndl.summation(losses) / Z.shape[0]
 
    ### END YOUR SOLUTION


def nn_epoch(X, y, W1, W2, lr=0.1, batch=100):
    """Run a single epoch of SGD for a two-layer neural network defined by the
    weights W1 and W2 (with no bias terms):
        logits = ReLU(X * W1) * W1
    The function should use the step size lr, and the specified batch size (and
    again, without randomizing the order of X).

    Args:
        X (np.ndarray[np.float32]): 2D input array of size
            (num_examples x input_dim).
        y (np.ndarray[np.uint8]): 1D class label array of size (num_examples,)
        W1 (ndl.Tensor[np.float32]): 2D array of first layer weights, of shape
            (input_dim, hidden_dim)
        W2 (ndl.Tensor[np.float32]): 2D array of second layer weights, of shape
            (hidden_dim, num_classes)
        lr (float): step size (learning rate) for SGD
        batch (int): size of SGD mini-batch

    Returns:
        Tuple: (W1, W2)
            W1: ndl.Tensor[np.float32]
            W2: ndl.Tensor[np.float32]
    """

    ### BEGIN YOUR SOLUTION
    num_examples = X.shape[0]
    num_classes = W2.shape[1]

    for start in range(0, num_examples, batch):
        end = min(start + batch, num_examples)
        X_batch = X[start:end]
        y_batch = y[start:end]

        # Convert numpy arrays to tensors
        X_batch_tensor = ndl.Tensor(X_batch)
        y_batch_tensor = ndl.Tensor(y_batch)

        # Forward pass: Compute logits
        Z1 = ndl.relu(ndl.matmul(X_batch_tensor, W1))
        Z2 = ndl.matmul(Z1, W2)

        # Create a one-hot encoded matrix of the true labels using NumPy
        I_y_np = np.zeros((len(y_batch), num_classes))
        I_y_np[np.arange(len(y_batch)), y_batch] = 1

        # Convert to ndl.Tensor
        I_y = ndl.Tensor(I_y_np)

        # Compute softmax loss
        loss = softmax_loss(Z2, I_y)

        # Backward pass: Compute gradients
        loss.backward()

        # Update the weights using gradient descent
        new_W1 = ndl.Tensor(W1.numpy() - lr * W1.grad.numpy())
        new_W2 = ndl.Tensor(W2.numpy() - lr * W2.grad.numpy())
        W1, W2 = new_W1, new_W2

    return (W1, W2)
    ### END YOUR SOLUTION
```


#### First Implementation (with NumPy)

In the first implementation, the code uses NumPy, which is a powerful library for numerical computation but does not support automatic differentiation or backpropagation out of the box. Here's why the first implementation can't directly use the `softmax_loss` function inside the `nn_epoch` function:

- **No Automatic Differentiation**: NumPy doesn't inherently support automatic differentiation, which is required to calculate gradients needed for backpropagation. The `softmax_loss` function in this implementation only computes the loss value and does not return gradients. Without gradients, the neural network can't update its weights during training.

- **Manual Gradient Computation**: In the first implementation, you would need to manually compute the gradients of the loss with respect to the model's parameters (`W1` and `W2`). This requires additional code that calculates these gradients explicitly, which isn't done in this implementation. The loss calculation and gradient computation are separate concerns, and the `softmax_loss` function doesn't address the gradient aspect.

#### Second Implementation (with a Deep Learning Library like `ndl`)

In the second implementation, the code is using a library that presumably supports automatic differentiation (likely a custom deep learning framework or an alias for something like PyTorch, TensorFlow, etc.). Here's why the second implementation can use the `softmax_loss` function inside the `nn_epoch` function:

- **Support for Automatic Differentiation**: The `ndl` library (or whatever library is being used) supports automatic differentiation. When you perform operations on `ndl.Tensor` objects, the library automatically builds a computational graph. This graph tracks the operations performed on the tensors, allowing for gradients to be computed automatically when you call `loss.backward()`.

- **Gradient Computation with `backward()`**: After computing the loss using `softmax_loss(z, label)`, calling `loss.backward()` automatically computes the gradients of the loss with respect to all the model parameters involved in the computation. These gradients are stored in the `.grad` attribute of the tensors (`W1.grad`, `W2.grad`). This means you don't have to manually compute the gradients as the library handles it for you.

- **Seamless Integration**: Because the second implementation uses a deep learning framework that integrates forward computation, loss calculation, and backward computation (gradient calculation), it can directly use the `softmax_loss` function inside the training loop (`nn_epoch`). The loss function not only provides the loss value but also enables gradient computation, which is crucial for updating the weights during training.

