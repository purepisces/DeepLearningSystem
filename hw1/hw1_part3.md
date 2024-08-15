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

```
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
    with gzip.open(image_filename, 'rb') as img_f:
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
    
    def compute(self, a: NDArray) -> NDArray:
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
    def compute(self, a: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        return array_api.exp(a)
        ### END YOUR SOLUTION
        
    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a = node.inputs[0]
        return out_grad * array_api.exp(a)
        ### END YOUR SOLUTION
        
def exp(a):
    return Exp()(a)
```




