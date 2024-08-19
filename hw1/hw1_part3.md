## Question 5: Softmax loss [10 pts]

The following questions will be tested using the MNIST dataset, so we will use the `parse_mnist` function we wrote in the Homework 0. 

1. First, copy and paste your solution to Question 2 of Homework 0 to the `parse_mnist` function in the `apps/simple_ml.py` file.  

In this question, you will implement the softmax loss as defined in the `softmax_loss()` function in `apps/simple_ml.py`, which we defined in Question 3 of Homework 0, except this time, the softmax loss takes as input a `Tensor` of logits and a `Tensor` of one hot encodings of the true labels. As a reminder, for a multi-class output that can take on values $y \in \{1,\ldots,k\}$, the softmax loss takes as input a vector of logits $z \in \mathbb{R}^k$, the true class $y \in \{1,\ldots,k\}$ (which is encoded for this function as a one-hot-vector) returns a loss defined by

$$\begin{equation}
\ell_{\mathrm{softmax}}(z, y) = \log\sum_{i=1}^k \exp z_i - z_y.
\end{equation}$$

You will first need to implement the forward and backward passes of one additional operator: ``log``. 

2. Fill out the `compute()` function in the `Log` and `Exp` operator in `python/needle/ops/ops_mathematic.py`.
3. Fill out the `gradient()` function in the `Log` and `Exp` operator in `python/needle/ops/ops_mathematic.py`. 
 
Once those operators have been implemented, 

4. Implement the function `softmax_loss` in `apps/simple_ml.py`. 

You can start with your solution from Homework 0, and then modify it to be compatible with `needle` objects and operations. As with the previous homework, the function you implement should compute the _average_ softmax loss over a batch of size $m$, i.e. logits `Z` will be an $m \times k$ `Tensor` where each row represents one example, and `y_one_hot` will be an $m \times k$ `Tensor` that contains all zeros except for a 1 in the element corresponding to the true label for each row. Finally, note that the average softmax loss returned should also be a `Tensor`. 


---------------------------
## Parse_MNIST from previous homework

```python
def parse_mnist(image_filesname, label_filename):
    """Read an images and labels file in MNIST format.  See this page:
    http://yann.lecun.com/exdb/mnist/ for a description of the file format.

    Args:
        image_filename (str): name of gzipped images file in MNIST format
        label_filename (str): name of gzipped labels file in MNIST format

    Returns:
        Tuple (X,y):
            X (numpy.ndarray[np.float32]): 2D numpy array containing the loaded
                data.  The dimensionality of the data should be
                (num_examples x input_dim) where 'input_dim' is the full
                dimension of the data, e.g., since MNIST images are 28x28, it
                will be 784.  Values should be of type np.float32, and the data
                should be normalized to have a minimum value of 0.0 and a
                maximum value of 1.0.

            y (numpy.ndarray[dypte=np.int8]): 1D numpy array containing the
                labels of the examples.  Values should be of type np.int8 and
                for MNIST will contain the values 0-9.
    """
    ### BEGIN YOUR SOLUTION
    # Read the labels file
    with gzip.open(label_filename, 'rb') as lbl_f:
        magic, num_items = struct.unpack(">II", lbl_f.read(8))
        if magic != 2049:
            raise ValueError(f"Invalid magic number in label file: {magic}")
        labels = np.frombuffer(lbl_f.read(num_items), dtype=np.uint8)
    
    # Read the images file
    with gzip.open(image_filesname, 'rb') as img_f:
        magic, num_images, num_rows, num_cols = struct.unpack(">IIII", img_f.read(16))
        if magic != 2051:
            raise ValueError(f"Invalid magic number in image file: {magic}")
        images = np.frombuffer(img_f.read(num_images * num_rows * num_cols), dtype=np.uint8)
        images = images.reshape(num_images, num_rows * num_cols).astype(np.float32)
        images /= 255.0  # Normalize to range [0, 1]
    
    return images, labels
    ### END YOUR SOLUTION
```

## `Log`: Element-wise natural logarithm of the input

**Example**

**Forward Pass**

If you have the following `ndarray`:

- **Ndarray**: `np.array([1, 2, 4])`

The element-wise natural logarithm would result in:

- **Result**: `np.array([log(1), log(2), log(4)])` which is approximately `np.array([0, 0.693, 1.386])`

**Backward Pass**

During the backward pass, you want to calculate the gradient of the loss $\ell$ with respect to the input $a$ of the `Log` operation.

- `out_grad` represents $\frac{\partial \ell}{\partial f}$, which is the gradient of the loss $\ell$ with respect to the output $f$ of the `Log` operation.
- The chain rule states $\frac{\partial \ell}{\partial a} = \frac{\partial \ell}{\partial f} \cdot \frac{\partial f}{\partial a}$

For $f(a) = \log(a)$: $\frac{\partial f}{\partial a} = \frac{1}{a}$

Combining these using the chain rule:

$\frac{\partial \ell}{\partial a} = \frac{\partial \ell}{\partial f} \cdot \frac{\partial f}{\partial a} = \text{outgrad} \cdot \frac{1}{a}$

```python
class Log(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.log(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a = node.inputs[0]
        return out_grad / a
        ### END YOUR SOLUTION


def log(a):
    return Log()(a)
```

## `Exp`: Element-wise Exponential of the Input

**Example**

**Forward Pass**

If you have the following `ndarray`:

- **Ndarray**: `np.array([0, 1, 2])`

The element-wise exponential would result in:

- **Result**: `np.array([exp(0), exp(1), exp(2)])` which is approximately `np.array([1, 2.718, 7.389])`

**Backward Pass**

During the backward pass, you want to calculate the gradient of the loss $\ell$ with respect to the input $a$ of the `Exp` operation.

- `out_grad` represents $\frac{\partial \ell}{\partial f}$, which is the gradient of the loss $\ell$ with respect to the output $f$ of the `Exp` operation.
- The chain rule states $\frac{\partial \ell}{\partial a} = \frac{\partial \ell}{\partial f} \cdot \frac{\partial f}{\partial a}$

For $f(a) = \exp(a)$:

$$\frac{\partial f}{\partial a} = \exp(a)$$

Combining these using the chain rule:

$$\frac{\partial \ell}{\partial a} = \frac{\partial \ell}{\partial f} \cdot \frac{\partial f}{\partial a} = \text{outgrad} \cdot \exp(a)$$


```python
class Exp(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.exp(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a = node.inputs[0]
        return out_grad * exp(a)
        ### END YOUR SOLUTION


def exp(a):
    return Exp()(a)
```
## Implement the function softmax_loss in apps/simple_ml.py.

### From hw0

Implement the softmax (a.k.a. cross-entropy) loss as defined in `softmax_loss()` function in `src/simple_ml.py`.  Recall (hopefully this is review, but we'll also cover it in lecture on 9/1), that for a multi-class output that can take on values $y \in \{1,\ldots,k\}$, the softmax loss takes as input a vector of logits $z \in \mathbb{R}^k$, the true class $y \in \{1,\ldots,k\}$ returns a loss defined by

$$\begin{equation}
\ell_{\mathrm{softmax}}(z, y) = \log\sum_{i=1}^k \exp z_i - z_y.
\end{equation}$$

Note that as described in its docstring, `softmax_loss()` takes a _2D array_ of logits (i.e., the $k$ dimensional logits for a batch of different samples), plus a corresponding 1D array of true labels, and should output the _average_ softmax loss over the entire batch.  Note that to do this correctly, you should _not_ use any loops, but do all the computation natively with numpy vectorized operations (to set expectations here, we should note for instance that our reference solution consists of a single line of code).

Note that for "real" implementation of softmax loss you would want to scale the logits to prevent numerical overflow, but we won't worry about that here (the rest of the assignment will work fine even if you don't worry about this). 

```python
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
```
Example
```python
import numpy as np

# Logits for a batch of 3 samples and 4 classes
Z = np.array([[2.0, 1.0, 0.1, 0.5],
              [1.5, 2.1, 0.2, 0.7],
              [1.1, 1.8, 0.3, 0.4]])

# True labels for the 3 samples
y = np.array([0, 1, 2])

# np.arange(Z.shape[0]) creates an array [0, 1, 2]
row_indices = np.arange(Z.shape[0])
print("Row indices:", row_indices)  # Output: [0 1 2]

# y is [0, 1, 2]
print("True class labels:", y)  # Output: [0 1 2]

# Advanced indexing: Z[np.arange(Z.shape[0]), y] selects Z[0, 0], Z[1, 1], Z[2, 2]
correct_class_logits = Z[row_indices, y]

print("Correct class logits:", correct_class_logits)
# Output: [2.0, 2.1, 0.3]
```


### Math Prove CMU 10714 For Loss function #2: softmax / cross-entropy loss

Let's convert the hypothesis function to a "probability" by exponentiating and normalizing its entries (to make them all positive and sum to one)

$$z_i = p(\text{label} = i) = \frac{\exp(h_i(x))}{\sum_{j=1}^k \exp(h_j(x))} \Longleftrightarrow z \equiv \text{softmax}(h(x))$$

Then let's define a loss to be the (negative) log probability of the true class: this is called softmax or cross-entropy loss

$$\ell_{ce}(h(x), y) = - \log p(\text{label} = y) = - h_y(x) + \log \sum_{j=1}^k \exp(h_j(x))$$


### Math Prove By Myself

**Equation for All Training Examples**:

$$H(Y, P) = -\sum_{i=1}^k Y_i \log(P_i) = H(Y, \sigma(z)) = -\sum\limits_{i=1}^k Y_i \log(\sigma(z)_i)$$

**Equation for One Training Example**:

$$H(Y, \sigma(z)) = -\log(\sigma(z)y) = -\log\left( \frac{\exp(z_y)}{\sum\limits_{j=1}^k \exp(z_j)} \right)$$


**Simplified Equation for One Training Example**:

$$H(Y, \sigma(z)) = -z_y + \log\left( \sum\limits_{j=1}^k \exp(z_j) \right)$$

#### Softmax Function

The softmax function converts logits (raw scores) into probabilities. For a vector of logits $z$ of length $k$, the softmax function $\sigma(z)$ is defined as:

$$\sigma(z)i = \frac{\exp(z_i)}{\sum\limits_{j=1}^k \exp(z_j)}$$

for $i = 1, \ldots, k$.

#### Cross-Entropy Loss

The cross-entropy loss measures the difference between the true labels and the predicted probabilities. For a true label vector $Y$ (one-hot encoded) and a predicted probability vector $P$ (output of the softmax function), the cross-entropy loss $H(Y, P)$ is defined as:

$$H(Y, P) = -\sum_{i=1}^k Y_i \log(P_i)$$

#### Connection Between Softmax and Cross-Entropy

When using the softmax function as the final layer in a neural network for multi-class classification, the predicted probability vector $P$ is given by:

$$P_i = \sigma(z) i = \frac{\exp(z_i)}{\sum\limits_{j=1}^k \exp(z_j)}$$

The cross-entropy loss then becomes:

$$H(Y, \sigma(z)) = -\sum_{i=1}^k Y_i \log(\sigma(z)_i)$$

For a single training example where the true class is $y$, $Y$ is a one-hot encoded vector where $Y_y = 1$ and $Y_i = 0$ for $i \neq y$. Thus, the cross-entropy loss simplifies to:

$$H(Y, \sigma(z)) = -\log(\sigma(z)y) = -\log\left( \frac{\exp(z_y)}{\sum\limits_{j=1}^k \exp(z_j)} \right)$$

Using properties of logarithms, this can be rewritten as:

$$H(Y, \sigma(z)) = -\left( \log(\exp(z_y)) - \log\left( \sum\limits_{j=1}^k \exp(z_j) \right) \right)$$

$$H(Y, \sigma(z)) = -z_y + \log\left( \sum\limits_{j=1}^k \exp(z_j) \right)$$



### For hw1


Code Implementation:
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
```


Example
```python
import numpy as np

# Logits for a batch of 3 samples and 4 classes
Z = np.array([[2.0, 1.0, 0.1, 0.5],
              [1.5, 2.1, 0.2, 0.7],
              [1.1, 1.8, 0.3, 0.4]])

# True labels for the 3 samples (in one-hot encoded form)
y_one_hot = np.array([[1, 0, 0, 0],   # Corresponds to class 0
                      [0, 1, 0, 0],   # Corresponds to class 1
                      [0, 0, 1, 0]])  # Corresponds to class 2

print("Logits (Z):")
print(Z)
# Output:
# [[2.0 1.0 0.1 0.5]
#  [1.5 2.1 0.2 0.7]
#  [1.1 1.8 0.3 0.4]]

print("\nOne-hot encoded true labels (y_one_hot):")
print(y_one_hot)
# Output:
# [[1 0 0 0]
#  [0 1 0 0]
#  [0 0 1 0]]

# Element-wise multiplication of Z and y_one_hot
Z_times_y = Z * y_one_hot

print("\nElement-wise multiplication of Z and y_one_hot:")
print(Z_times_y)
# Output:
# [[2.0 0.0 0.0 0.0]
#  [0.0 2.1 0.0 0.0]
#  [0.0 0.0 0.3 0.0]]

# Summation along axis 1 to extract the correct class logits
correct_class_logits = np.sum(Z_times_y, axis=1)

print("\nCorrect class logits after summation:")
print(correct_class_logits)
# Output: [2.0 2.1 0.3]
```
