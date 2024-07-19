The assignment will require you to build a basic softmax regression algorithm, plus a simple two-layer neural network.  You will create these implementations both in native Python (using the numpy library), and (for softmax regression) in native C/C++. 

## Dataset:

The MNIST (Modified National Institute of Standards and Technology) dataset is a well-known dataset in the field of machine learning and computer vision. It consists of images of handwritten digits (0-9) and their corresponding labels. 

Structure of the MNIST Dataset
The MNIST dataset is split into two parts:

Training set: 60,000 examples
Test set: 10,000 examples
Each example in the dataset consists of:

Image: A 28x28 grayscale image of a handwritten digit.
Label: A label corresponding to the digit in the image (0-9).

```python
def parse_mnist(image_filename, label_filename):
    """ Read an images and labels file in MNIST format.  See this page:
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
                maximum value of 1.0 (i.e., scale original values of 0 to 0.0 
                and 255 to 1.0).

            y (numpy.ndarray[dtype=np.uint8]): 1D numpy array containing the
                labels of the examples.  Values should be of type np.uint8 and
                for MNIST will contain the values 0-9.
    """
    ### BEGIN YOUR CODE
    # Read the labels file
    with gzip.open(label_filename, 'rb') as lbl_f:
        magic, num_items = struct.unpack(">II", lbl_f.read(8))
        if magic != 2049:
            raise ValueError(f"Invalid magic number in label file: {magic}")
        labels = np.frombuffer(lbl_f.read(num_items), dtype=np.uint8)
    
    # Read the images file
    with gzip.open(image_filename, 'rb') as img_f:
        magic, num_images, num_rows, num_cols = struct.unpack(">IIII", img_f.read(16))
        if magic != 2051:
            raise ValueError(f"Invalid magic number in image file: {magic}")
        images = np.frombuffer(img_f.read(num_images * num_rows * num_cols), dtype=np.uint8)
        images = images.reshape(num_images, num_rows * num_cols).astype(np.float32)
        images /= 255.0  # Normalize to range [0, 1]
    
    return images, labels

    ### END YOUR CODE
```

## Softmax(a.k.a. cross-entropy) loss:

## Question 3: Softmax loss

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
## Stochastic gradient descent for softmax regression

In this question you will implement stochastic gradient descent (SGD) for (linear) softmax regression.  In other words, as discussed in lecture on 9/1, we will consider a hypothesis function that makes $n$-dimensional inputs to $k$-dimensional logits via the function

$$\begin{equation}
h(x) = \Theta^T x
\end{equation}$$

where $x \in \mathbb{R}^n$ is the input, and $\Theta \in \mathbb{R}^{n \times k}$ are the model parameters.  Given a dataset $\{(x^{(i)} \in \mathbb{R}^n, y^{(i)} \in \{1,\ldots,k\})\}$, for $i=1,\ldots,m$, the optimization problem associated with softmax regression is thus given by

$$\begin{equation}
\minimize_{\Theta} \; \frac{1}{m} \sum_{i=1}^m \ell_{\mathrm{softmax}}(\Theta^T x^{(i)}, y^{(i)}).
\end{equation}$$

Recall from class that the gradient of the linear softmax objective is given by

$$\begin{equation}
\nabla_\Theta \ell_{\mathrm{softmax}}(\Theta^T x, y) = x (z - e_y)^T
\end{equation}$$

where

$$\begin{equation}
z = \frac{\exp(\Theta^T x)}{1^T \exp(\Theta^T x)} \equiv \normalize(\exp(\Theta^T x))
\end{equation}$$

(i.e., $z$ is just the normalized softmax probabilities), and where $e_y$ denotes the $y$th unit basis, i.e., a vector of all zeros with a one in the $y$th position.

We can also write this in the more compact notation we discussed in class.  Namely, if we let $X \in \mathbb{R}^{m \times n}$ denote a design matrix of some $m$ inputs (either the entire dataset or a minibatch), $y \in \{1,\ldots,k\}^m$ a corresponding vector of labels, and overloading $\ell_{\mathrm{softmax}}$ to refer to the average softmax loss, then

$$\begin{equation}
\nabla_\Theta \ell_{\mathrm{softmax}}(X \Theta, y) = \frac{1}{m} X^T (Z - I_y)
\end{equation}$$

where

$$\begin{equation}
Z = \normalize(\exp(X \Theta)) \quad \mbox{(normalization applied row-wise)}
\end{equation}$$

denotes the matrix of logits, and $I_y \in \mathbb{R}^{m \times k}$ represents a concatenation of one-hot bases for the labels in $y$.

Using these gradients, implement the `softmax_regression_epoch()` function, which runs a single epoch of SGD (one pass over a data set) using the specified learning rate / step size `lr` and minibatch size `batch`.  As described in the docstring, your function should modify the `Theta` array in-place.  After implementation, run the tests.

### Math Prove


### Understanding the Derivative of Logarithm of the Softmax Denominator

To understand why
$$
\frac{\partial \log(A)}{\partial z_i} = \frac{1}{A} \frac{\partial A}{\partial z_i} = \frac{\exp(z_i)}{\sum_{j=1}^k \exp(z_j)} = \sigma(z_i)$$
let's break this down step-by-step.

1. **Logarithm Derivative Rule**
   The derivative of the logarithm function \(\log(A)\) with respect to \(z_i\) can be expressed using the chain rule. Specifically, if \(A = f(z)\), then:
$$
   \frac{\partial \log(A)}{\partial z_i} = \frac{1}{A} \frac{\partial A}{\partial z_i}
$$

2. **Softmax Function**
   The softmax function is defined as:
$$
   \sigma(z)_i = \frac{\exp(z_i)}{\sum_{j=1}^k \exp(z_j)}
$$
   Let's denote \(A\) as the denominator of the softmax function:
$$
   A = \sum_{j=1}^k \exp(z_j)
$$

3. **Logarithm of Softmax Denominator**
   Consider the logarithm of the softmax denominator:
$$
   \log(A) = \log\left(\sum_{j=1}^k \exp(z_j)\right)
$$
   We need to find the partial derivative of \(\log(A)\) with respect to \(z_i\):
$$
   \frac{\partial \log(A)}{\partial z_i}
$$

4. **Apply the Chain Rule**
   Using the chain rule for derivatives:
$$
   \frac{\partial \log(A)}{\partial z_i} = \frac{1}{A} \frac{\partial A}{\partial z_i}
$$

5. **Derivative of \(A\) with respect to \(z_i\)**
   Now, we need to compute the partial derivative of \(A\) with respect to \(z_i\):
$$
   A = \sum_{j=1}^k \exp(z_j)
$$
   So,
$$
   \frac{\partial A}{\partial z_i} = \frac{\partial}{\partial z_i} \left( \sum_{j=1}^k \exp(z_j) \right) = \exp(z_i)
$$

6. **Substitute Back**
   Substituting $\frac{\partial A}{\partial z_i} = \exp(z_i)\) and \(A = \sum_{j=1}^k \exp(z_j)$ into the chain rule expression:
$$
   \frac{\partial \log(A)}{\partial z_i} = \frac{1}{A} \frac{\partial A}{\partial z_i} = \frac{1}{\sum_{j=1}^k \exp(z_j)} \cdot \exp(z_i)
$$
   Simplifying this expression:
$$
   \frac{\partial \log(A)}{\partial z_i} = \frac{\exp(z_i)}{\sum_{j=1}^k \exp(z_j)}
$$

7. **Recognize the Softmax Output**
   Notice that this is exactly the softmax function \(\sigma(z)_i\):
$$
   \frac{\partial \log(A)}{\partial z_i} = \sigma(z)_i = \frac{\exp(z_i)}{\sum_{j=1}^k \exp(z_j)}
$$

### Gradient of the Softmax Loss

Let's derive the gradient $\nabla_\Theta \ell_{\mathrm{softmax}}(\Theta^T x, y) = x (z - e_y)^T$ step by step.

1. **Softmax Function**
   First, recall the softmax function for converting logits (raw scores) into probabilities:
$$\sigma(z_i) = \frac{\exp(z_i)}{\sum_{j=1}^k \exp(z_j)}$$
   where $z_i = \Theta_i^T x$are the logits for the $i$-th class.

2. **Cross-Entropy Loss**
   The cross-entropy loss for a single example \((x, y)\) is given by:
   \[
   \ell_{\mathrm{softmax}}(\Theta^T x, y) = -\log(\sigma(z_y))
   \]
   where \(z_y\) is the logit corresponding to the true class \(y\).

3. **Simplifying the Loss Function**
   Expressing \(\sigma(z_y)\) using the softmax function:
   \[
   \sigma(z_y) = \frac{\exp(z_y)}{\sum_{j=1}^k \exp(z_j)}
   \]
   Therefore, the loss can be written as:
   \[
   \ell_{\mathrm{softmax}}(\Theta^T x, y) = -\log\left(\frac{\exp(z_y)}{\sum_{j=1}^k \exp(z_j)}\right) = -\left(\log(\exp(z_y)) - \log\left(\sum_{j=1}^k \exp(z_j)\right)\right) = -z_y + \log\left(\sum_{j=1}^k \exp(z_j)\right)
   \]

4. **Gradient of the Loss with Respect to \(\Theta\)**
   To find the gradient of the loss with respect to \(\Theta\), we first need to find the partial derivatives of each term.

   4.1 **Gradient of \(-z_y**\):
   The term \(-z_y\) is linear in \(\Theta\):
   \[
   z_y = \Theta_y^T x
   \]
   So, the gradient of \(-z_y\) with respect to \(\Theta\) is:
   \[
   \frac{\partial (-z_y)}{\partial \Theta_i} = -x \cdot \delta_{iy}
   \]
   where \(\delta_{iy}\) is the Kronecker delta, which is 1 if \(i=y\) and 0 otherwise.

   4.2 **Gradient of \(\log\left(\sum_{j=1}^k \exp(z_j)\right)\)**:
   Let \(A = \sum_{j=1}^k \exp(z_j)\). Then,
   \[
   \log(A)
   \]
   The derivative of \(\log(A)\) with respect to \(z_i\) is:
   \[
   \frac{\partial \log(A)}{\partial z_i} = \frac{1}{A} \frac{\partial A}{\partial z_i} = \frac{\exp(z_i)}{\sum_{j=1}^k \exp(z_j)} = \sigma(z_i)
   \]
   Therefore, the gradient of the log term with respect to \(\Theta\) is:
   \[
   \frac{\partial \log\left(\sum_{j=1}^k \exp(z_j)\right)}{\partial \Theta_i} = \sigma(z_i) x
   \]

5. **Combining the Gradients**
   Combining the two parts, we get:
   \[
   \nabla_{\Theta_i} \ell_{\mathrm{softmax}}(\Theta^T x, y) = \sigma(z_i) x - x \cdot \delta_{iy}
   \]
   In vector form, this can be written as:
   \[
   \nabla_\Theta \ell_{\mathrm{softmax}}(\Theta^T x, y) = x (z - e_y)^T
   \]
   where \(\sigma(z)\) is the vector of softmax probabilities, and \(e_y\) is the one-hot encoded vector for the true class \(y\).

### Final Gradient Expression
Thus, we have the final gradient expression for the softmax loss:
\[
\nabla_\Theta \ell_{\mathrm{softmax}}(\Theta^T x, y) = x (z - e_y)^T
\]
where \(z = \sigma(\Theta^T x)\).

This derivation shows why the gradient of the softmax loss with respect to the model parameters \(\Theta\) is given by the expression \(x (z - e_y)^T\).



