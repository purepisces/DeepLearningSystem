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
### `Log`: Element-wise natural logarithm of the input

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

$\frac{\partial \ell}{\partial a} = \frac{\partial \ell}{\partial f} \cdot \frac{\partial f}{\partial a} = \text{out\_grad} \cdot \frac{1}{a}$

```python
class Log(TensorOp):
    """Op that applies the natural logarithm element-wise to a tensor."""
    
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
