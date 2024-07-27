## Question 1: Implementing forward computation

First, you will implement the forward computation for new operators.  To see how this works, consider the `EWiseAdd` operator in the `ops/ops_mathematic.py` file:

```python
class EWiseAdd(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a + b

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad, out_grad

def add(a, b):
    return EWiseAdd()(a, b)
```

The conventions for implementations of this class are the following.  The `compute()` function computes the "forward" pass, i.e., just computes the operation itself.  However, it is important to emphasize the inputs to compute are both `NDArray` objects (i.e., in this initial implementation, they are just `numpy.ndarray` objects, though in a later assignment you will implement your own NDArray).  That is, `compute()` computes the forward pass on the _raw data objects_ themselves, not on Tensor objects within the automatic differentiation.

We will discuss the `gradient()` call in the next section, but it is important to emphasize here that this call is different from forward in that it takes `Tensor` arguments.  This means that any call you make within this function _should_ be done via `TensorOp` operations themselves (so that you can take gradients of gradients).

Finally, note that we also define a helper `add()` function, to avoid the need to call `EWiseAdd()(a,b)` (which is a bit cumbersome) to add two `Tensor` objects.  These functions are all written for you, and should be self-explanation.


For this question, you will need to implement the `compute` call for each of the following classes.  These calls are very straightforward, and should be essentially one line that calls to the relevant numpy function.  Note that because in later homeworks you will use a backend other than numpy, we have imported numpy as `import numpy as array_api`, so that you'll need to call `array_api.add()` etc, if you want to use the typical `np.X()` calls.

- `PowerScalar`: raise input to an integer (scalar) power
- `EWiseDiv`: true division of the inputs, element-wise (2 inputs)
- `DivScalar`: true division of the input by a scalar, element-wise (1 input, `scalar` - number)
- `MatMul`: matrix multiplication of the inputs (2 inputs)
- `Summation`: sum of array elements over given axes (1 input, `axes` - tuple)
- `BroadcastTo`: broadcast an array to a new shape (1 input, `shape` - tuple)
- `Reshape`: gives a new shape to an array without changing its data (1 input, `shape` - tuple)
- `Negate`: numerical negative, element-wise (1 input)
- `Transpose`: reverses the order of two axes (axis1, axis2), defaults to the last two axes (1 input, `axes` - tuple)
------------------------------------------
```python
class EWiseAdd(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a + b

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad, out_grad

def add(a, b):
    return EWiseAdd()(a, b)
```

- **Using `NDArray` in `compute`**: Keeps the method focused on efficient numerical operations.
- **Using `Tensor` in the Rest of the Framework**: Manages the computational graph, tracks operations, and stores gradients for automatic differentiation.
- **Separation of Concerns**: Ensures that numerical computations and gradient tracking are handled separately, leveraging the strengths of numpy for numerical tasks and `Tensor` for managing the differentiation process.

## Question 2: Implementing backward computation 

Now that you have implemented the functions within our computation graph, in order to implement automatic differentiation using our computational graph, we need to be able to compute the backward pass, i.e., multiply the relevant derivatives of the function with the incoming backward gradients.

The easiest way to perform these computations is, again, via taking "fake" partial derivatives (assuming everything is a scalar), and then matching sizes: here the tests we provide will automatically check against numerical derivatives to ensure that your solution is correct.

The general goal of reverse mode autodifferentiation is to compute the gradient of some downstream function $\ell$ of $f(x,y)$ with respect to $x$ (or $y$). Written formally, we could write this as trying to compute

$$\begin{equation}
\frac{\partial \ell}{\partial x} = \frac{\partial \ell}{\partial f(x,y)} \frac{\partial f(x,y)}{\partial x}.
\end{equation}$$

The "incoming backward gradient" is precisely the term $\frac{\partial \ell}{\partial f(x,y)}$, so we want our `gradient()` function to ultimately compute the _product_ between this backward gradient the function's own derivative $\frac{\partial f(x,y)}{\partial x}$.

To see how this works a bit more concretely, consider the elementwise addition function we presented above

$$\begin{equation}
f(x,y) = x + y.
\end{equation}$$

Let's suppose that in this setting $x,y\in \mathbb{R}^n$, so that $f(x,y) \in \mathbb{R}^n$ as well. Then via simple differentiation

$$\begin{equation}
\frac{\partial f(x,y)}{\partial x} = 1
\end{equation}$$

so that

$$\begin{equation}
\frac{\partial \ell}{\partial x} = \frac{\partial \ell}{\partial f(x,y)} \frac{\partial f(x,y)}{\partial x} = \frac{\partial \ell}{\partial f(x,y)}
\end{equation}$$

i.e., the product of the function's derivative with respect to its first argument $x$ is just exactly the same as the backward incoming gradient. The same is true of the gradient with respect to the second argument $y$. This is precisely what is captured by the following method of the `EWiseAdd` operator.

```python
def gradient(self, out_grad: Tensor, node: Tensor):
	return out_grad, out_grad
```
i.e., the function just results the incoming backward gradient (which actually _is_ here the product between the backward incoming gradient and the derivative with respect to each argument of the function. And because the size of $f(x,y)$ is the same as the size of both $x$ and $y$, we don't even need to worry about dimensions here.

-   Here, `out_grad` represents $\frac{\partial \ell}{\partial f(x, y)}$.
-   Since $\frac{\partial f(x, y)}{\partial x} = 1$ and $\frac{\partial f(x, y)}{\partial y} = 1$, the gradients with respect to $x$ and $y$ are just `out_grad`.

Now consider another example, the (elementwise) multiplication function

$$\begin{equation}
f(x,y) = x \circ y
\end{equation}$$

where $\circ$ denotes elementwise multiplication between $x$ and $y$. The partial of this function is given by

$$\begin{equation}
\frac{\partial f(x,y)}{\partial x} = y
\end{equation}$$

and similarly

$$\begin{equation}
\frac{\partial f(x,y)}{\partial y} = x
\end{equation}$$

  

Thus to compute the product of the incoming gradient

$$\begin{equation}
\frac{\partial \ell}{\partial x} = \frac{\partial \ell}{\partial f(x,y)} \frac{\partial f(x,y)}{\partial x} = \frac{\partial \ell}{\partial f(x,y)} \cdot y
\end{equation}$$

If $x,y \in \mathbb{R}^n$ like in the previous example, then $f(x,y) \in \mathbb{R}^n$ as well so the first element returned back the graident function would just be the elementwise multiplication

$$\begin{equation}
\frac{\partial \ell}{\partial f(x,y)} \circ y
\end{equation}$$

This is captures in the `gradient()` call of the `EWiseMul` class.

```python
class EWiseMul(TensorOp):

	def compute(self, a: NDArray, b: NDArray):
		return a * b

	def gradient(self, out_grad: Tensor, node: Tensor):
		lhs, rhs = node.inputs
		return out_grad * rhs, out_grad * lhs
```
>**out_grad:**
>-   `out_grad` is the gradient of the loss function ℓ\ellℓ with respect to the output of the current operation.
>- It is the gradient that comes from the next layer or operation in the graph during the backward pass.
>
>**Node:**
>-   In automatic differentiation frameworks, a `node` typically represents a single operation (like addition, multiplication, etc.) in the computational graph.
>-   The `node` object contains:
>     -   `inputs`: A list of tensors that were the inputs to this operation during the forward pass. In this case, `node.inputs` would be a list containing the tensors `lhs` and `rhs` (i.e., the inputs to the elementwise multiplication).
    
### Implementing backward passes

Note that, unlike the forward pass functions, the arguments to the `gradient` function are `needle` objects. It is important to implement the backward passes using only `needle` operations (i.e. those defined in `python/needle/ops/ops_mathematic.py`), rather than using `numpy` operations on the underlying `numpy` data, so that we can construct the gradients themselves via a computation graph (one excpetion is for the `ReLU` operation defined below, where you could directly access data within the Tensor without risk because the gradient itself is non-differentiable, but this is a special case).


To complete this question, fill in the `gradient` function of the following classes:

- `PowerScalar`

- `EWiseDiv`

- `DivScalar`

- `MatMul`

- `Summation`

- `BroadcastTo`

- `Reshape`

- `Negate`

- `Transpose`

  

All of the `gradient` functions can be computed using just the operations defined in `python/needle/ops/ops_mathematic.py`, so there is no need to define any additional forward functions.


**Hint:** while gradients of multiplication, division, etc, may be relatively intuitive to compute it can seem a bit less intuitive to compute backward passes of items like `Broadcast` or `Summation`. To get a handle on these, you can check gradients numerically and print out their actual values, if you don't know where to start (see the `tests/test_autograd_hw.py`, specifically the `check_gradients()` function within that file to get a sense about how to do this). And remember that the side of `out_grad` will always be the size of the _output_ of the operation, whereas the sizes of the `Tensor` objects _returned_ by `graident()` have to always be the same as the original _inputs_ to the operator.


### Checking backward passes

To reiterate the above, remember that we can check that these backward passes are correct by doing numerical gradient checking as covered in lecture:

$$\begin{equation}
\delta^T \nabla_\theta f(\theta) = \frac{f(\theta + \epsilon \delta) - f(\theta - \epsilon \delta)}{2 \epsilon} + o(\epsilon^2)
\end{equation}$$

We provide the function `gradient_check` for doing this numerical checking in `tests/test_autograd_hw.py`.


----------------------------------


----------------------------------
### `PowerScalar`: raise input to an integer (scalar) power
** Example**
**Forward Pass**
If you have the following `ndarray` and scalar:

-   **Ndarray**: `np.array([2, 3, 4])`
-   **Scalar**: `3`

The element-wise power would result in:

-   **Result**: `np.array([2**3, 3**3, 4**3])` which is `np.array([8, 27, 64])`

**Backward Pass**
During the backward pass, you want to calculate the gradient of the loss $\ell$ with respect to the input $x$ of the `PowerScalar` operation.

- `out_grad` represents $\frac{\partial \ell}{\partial f}$, which is the gradient of the loss $\ell$ with respect to the output $f$ of the `PowerScalar` operation.
- The chain rule states $\frac{\partial \ell}{\partial x} = \frac{\partial \ell}{\partial f} \cdot \frac{\partial f}{\partial x}$

For $f(x) = x^c$: $\frac{\partial f}{\partial x} = c \cdot x^{c-1}$

Combining these using the chain rule:

$\frac{\partial \ell}{\partial x} = \frac{\partial \ell}{\partial f} \cdot \frac{\partial f}{\partial x} = \text{outgrad} \cdot c \cdot x^{c-1}$

```python
class PowerScalar(TensorOp):
    """Op raise a tensor to an (integer) power."""
    def __init__(self, scalar: int):
        self.scalar = scalar
        
    def compute(self, a: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        return array_api.power(a, self.scalar)
        ### END YOUR SOLUTION
        
    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a = node.inputs[0]
        return self.scalar * array_api.power(a, self.scalar - 1) * out_grad
        ### END YOUR SOLUTION
        
def power_scalar(a, scalar):
    return PowerScalar(scalar)(a)
```
> -   `PowerScalar(scalar)` initializes a `PowerScalar` object with the given scalar value, it creates an instance of the operation..
> -   `PowerScalar(scalar)(a)` calls the `__call__` method of the `PowerScalar` instance with `a` as the argument.
> -   Inside the `__call__` method, `Tensor.make_from_op(self, (a,))` is called:
>     -   `self` is the `PowerScalar` instance.
>     -   `(a,)` is a tuple containing the input tensor `a`.
>     
> This results in creating a new `Tensor` object that represents the result of applying the `PowerScalar` operation to `a`. The `make_from_op` method constructs and initializes a new `Tensor` object.

### `EWiseDiv`: true division of the inputs, element-wise (2 inputs)

** Example**
**Forward Pass**
If you have the following `ndarrays`:

- **Ndarray `a`**: `np.array([10, 20, 30])`
- **Ndarray `b`**: `np.array([2, 4, 6])`

The element-wise division would result in:

- **Result**: `np.array([10/2, 20/4, 30/6])` which is `np.array([5, 5, 5])`

**Backward Pass**

During the backward pass, you want to calculate the gradient of the loss $\ell$ with respect to the inputs $a$ and $b$ of the `EWiseDiv` operation.

- `out_grad` represents $\frac{\partial \ell}{\partial f}$, which is the gradient of the loss $\ell$ with respect to the output $f$ of the `EWiseDiv` operation.
- The chain rule states:
  $$\frac{\partial \ell}{\partial a} = \frac{\partial \ell}{\partial f} \cdot \frac{\partial f}{\partial a}$$
  and $$\frac{\partial \ell}{\partial b} = \frac{\partial \ell}{\partial f} \cdot \frac{\partial f}{\partial b}$$

For $f(a, b) = \frac{a}{b}$:

- $\frac{\partial f}{\partial a} = \frac{1}{b}$
- $\frac{\partial f}{\partial b} = -\frac{a}{b^2}$

Combining these using the chain rule:

- $$\frac{\partial \ell}{\partial a} = \frac{\partial \ell}{\partial f} \cdot \frac{1}{b} = \text{out\_grad} \cdot \frac{1}{b}$$

- $$\frac{\partial \ell}{\partial b} = \frac{\partial \ell}{\partial f} \cdot -\frac{a}{b^2} = \text{out\_grad} \cdot -\frac{a}{b^2}$$

```python
class EWiseDiv(TensorOp):
    """Op to element-wise divide two nodes."""

    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        return a / b
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a, b = node.inputs
        return out_grad / b, -a * out_grad / (b * b)
        ### END YOUR SOLUTION

def divide(a, b):
    return EWiseDiv()(a, b)
```

> In the `PowerScalar` class, writing `a = node.inputs` would be incorrect because `node.inputs` is a list containing one element, and you need to access that single element. In contrast, in the `EWiseDiv` class, `node.inputs` is a list containing two elements, so you unpack them into `a` and `b`.


### `DivScalar`: true division of the input by a scalar, element-wise (1 input, `scalar` - number)

**Example**
**Forward Pass**
If you have the following `ndarray` and scalar:

- **Ndarray**: `np.array([10, 20, 30])`
- **Scalar**: `2`

The element-wise division would result in:

- **Result**: `np.array([10/2, 20/2, 30/2])` which is `np.array([5, 10, 15])`

**Backward Pass**

During the backward pass, you want to calculate the gradient of the loss $\ell$ with respect to the input $a$ of the `DivScalar` operation.

-   `out_grad` represents $\frac{\partial \ell}{\partial f}$, which is the gradient of the loss $\ell$ with respect to the output $f$ of the `DivScalar` operation.
    
-   Using the chain rule:
	$\frac{\partial \ell}{\partial a} = \frac{\partial \ell}{\partial f} \cdot \frac{\partial f}{\partial a} = \text{out\_grad} \cdot \frac{1}{\text{scalar}}$
```python
class DivScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return a / self.scalar
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad /self.scalar
        ### END YOUR SOLUTION
        
def divide_scalar(a, scalar):
    return DivScalar(scalar)(a)
```
### `MatMul`: matrix multiplication of the inputs (2 inputs)

**Example**
**Forward Pass**
If you have the following `ndarrays`:

- **Ndarray `a`**: `np.array([[1, 2], [3, 4]])`
- **Ndarray `b`**: `np.array([[5, 6], [7, 8]])`

The matrix multiplication would result in:

- **Result**: `np.array([[1*5 + 2*7, 1*6 + 2*8], [3*5 + 4*7, 3*6 + 4*8]])` which is `np.array([[19, 22], [43, 50]])`

**Backward Pass**

During the backward pass, you want to calculate the gradients of the loss $\ell$ with respect to the inputs `a` and `b` of the `MatMul` operation.

- `out_grad` represents $\frac{\partial \ell}{\partial f}$, which is the gradient of the loss $\ell$ with respect to the output $f$ of the `MatMul` operation.
- The chain rule states $\frac{\partial \ell}{\partial a} = \frac{\partial \ell}{\partial f} \cdot \frac{\partial f}{\partial a}$ and $\frac{\partial \ell}{\partial b} = \frac{\partial \ell}{\partial f} \cdot \frac{\partial f}{\partial b}$
- Address any broadcasting issues to ensure the gradients match the original shapes of `a` and `b`.

For $f(a, b) = a \cdot b$:
- $\frac{\partial f}{\partial a}=b^\top$
- $\frac{\partial f}{\partial b}=a^\top$ 

Combining these using the chain rule:
- $\frac{\partial \ell}{\partial a} = \text{outgrad} \cdot b^T$
- $\frac{\partial \ell}{\partial b} = a^T \cdot \text{outgrad}$

```python
class MatMul(TensorOp):
    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        return array_api.matmul(a, b)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a, b = node.inputs
        # Compute gradients
        grad_a = matmul(out_grad, transpose(b))
        grad_b = matmul(transpose(a), out_grad)
        # Address broadcasting issues to align gradient shape with the original shape
        if grad_a.shape != a.shape:
            grad_a = summation(grad_a, tuple(range(len(grad_a.shape) - len(a.shape))))
        if grad_b.shape != b.shape:
            grad_b = summation(grad_b, tuple(range(len(grad_b.shape) - len(b.shape))))
        
        # Ensure the shapes match after summation
        assert grad_a.shape == a.shape
        assert grad_b.shape == b.shape
        return grad_a, grad_b
        ### END YOUR SOLUTION
        

def matmul(a, b):
    return MatMul()(a, b)
```
> If in forward pass, compute(self, a, b) return a*b, then it become np.array([[5, 12], [21, 32]]), incorrect result.
> 
> In forward pass, `numpy.matmul`  do handle broadcasting automatically during matrix multiplication.
> 

### `Summation`: sum of array elements over given axes (1 input, `axes` - tuple)

**Example**
**Forward Pass**

If you have the following `ndarray`:

- **Ndarray `a`**: `np.array([[1, 2, 3], [4, 5, 6]])`
- **Axes**: `(0,)`

The summation over the specified axes would result in:

- **Result**: `np.array([1+4, 2+5, 3+6])` which is `np.array([5, 7, 9])`

**Backward Pass**

During the backward pass, you want to calculate the gradient of the loss $\ell$ with respect to the input `a` of the `Summation` operation.

-   `out_grad` represents $\frac{\partial \ell}{\partial f}$, which is the gradient of the loss $\ell$ with respect to the output `f` of the `Summation` operation.
-   Using the chain rule: $\frac{\partial \ell}{\partial a} = \frac{\partial \ell}{\partial f} \cdot \frac{\partial f}{\partial a}$

For $f(a) = \sum(a \text{ over axes})$:

-   The gradient with respect to the input is `1` for all elements that were summed.
-   The gradient needs to be reshaped and broadcasted to match the shape of the input `a`.

```python
class Summation(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        return array_api.sum(a, axis=self.axes)
```

### `BroadcastTo`: broadcast an array to a new shape (1 input, `shape` - tuple)

**Example**
**Forward Pass**
If you have the following `ndarray`:

- **Ndarray `a`**: `np.array([1, 2, 3])`
- **Shape**: `(3, 3)`

The broadcasting to the specified shape would result in:

- **Result**: `np.array([[1, 2, 3], [1, 2, 3], [1, 2, 3]])`

**Backward Pass**
During the backward pass, you want to calculate the gradient of the loss $\ell$ with respect to the input `a` of the `BroadcastTo` operation.

-   `out_grad` represents $\frac{\partial \ell}{\partial f}$, which is the gradient of the loss $\ell$ with respect to the output `f` of the `BroadcastTo` operation.
-   Using the chain rule: $\frac{\partial \ell}{\partial a} = \frac{\partial \ell}{\partial f} \cdot \frac{\partial f}{\partial a}$

For broadcasting:

-   If broadcasting added extra dimensions, sum over those dimensions.
-   If broadcasting expanded dimensions of size 1, sum over those dimensions as well.

Combining these using the chain rule:

-   Calculate the sum over the dimensions added by broadcasting.
-   Calculate the sum over the dimensions where the input shape is 1.
```python
class BroadcastTo(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.broadcast_to(a, self.shape)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a = node.inputs[0]
        input_shape = a.shape
        output_shape = out_grad.shape
        grad = out_grad

        # Summing over the extra dimensions added by broadcasting
        for i in range(len(output_shape) - len(input_shape)):
            grad = summation(grad, axes=0)
        
        # Summing over the dimensions where the input shape is 1
        for i, dim in enumerate(input_shape):
            if dim == 1:
                grad = summation(grad, axes=i)
                
        return reshape(grad, input_shape)
        ### END YOUR SOLUTION


def broadcast_to(a, shape):
    return BroadcastTo(shape)(a)
```

### `Reshape`: Gives a new shape to an array without changing its data (1 input, `shape` - tuple)

**Example**
**Forward Pass**
If you have the following `ndarray`:

- **Ndarray `a`**: `np.array([[1, 2, 3], [4, 5, 6]])`
- **Shape**: `(3, 2)`

The reshaping to the specified shape would result in:

- **Result**: `np.array([[1, 2], [3, 4], [5, 6]])`

**Backward Pass**
During the backward pass, you want to calculate the gradient of the loss $\ell$ with respect to the input `a` of the `Reshape` operation.

-   `out_grad` represents $\frac{\partial \ell}{\partial f}$, which is the gradient of the loss $\ell$ with respect to the output `f` of the `Reshape` operation.
-   Using the chain rule: $\frac{\partial \ell}{\partial a} = \frac{\partial \ell}{\partial f} \cdot \frac{\partial f}{\partial a}$

For reshaping:

-   The gradient of reshaping is the reshaping of the gradient back to the original shape.

Combining these using the chain rule:

-   The gradient with respect to the input is simply the gradient reshaped to the original input shape.
```python
class Reshape(TensorOp):
class Reshape(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.reshape(a, self.shape)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a = node.inputs[0]
        return reshape(out_grad, a.shape)
        ### END YOUR SOLUTION


def reshape(a, shape):
    return Reshape(shape)(a)
```
>```python 
>import numpy as np
>a = np.array([[1, 2, 3], [4, 5, 6]])
>print(a.shape) #(2, 3)
>new_shape = (3,2)
> print(np.reshape(a,new_shape)) 
> # [[1 2],[3 4],[5 6]]
>```
>  **When the `np.reshape` function is used to change the shape of an array, it rearranges the elements of the array in a specific order. By default, `np.reshape` fills the elements of the new array in a row-major (C-style) order, which means that it reads and writes elements row by row.**

### `Negate`: Numerical negative, element-wise (1 input)

**Example**
**Forward Pass**

If you have the following `ndarray`:

- **Ndarray `a`**: `np.array([1, -2, 3])`

The negation would result in:

- **Result**: `np.array([-1, 2, -3])`

**Backward Pass**
During the backward pass, you want to calculate the gradient of the loss $\ell$ with respect to the input `a` of the `Negate` operation.

-   `out_grad` represents $\frac{\partial \ell}{\partial f}$, which is the gradient of the loss $\ell$ with respect to the output `f$ of the` Negate` operation.
-   Using the chain rule: $\frac{\partial \ell}{\partial a} = \frac{\partial \ell}{\partial f} \cdot \frac{\partial f}{\partial a}$

For negation:

-   The derivative of the negation function is -1.

Combining these using the chain rule:

-   The gradient with respect to the input is simply the gradient multiplied by -1.
```python
class Negate(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return -a
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return -out_grad
        ### END YOUR SOLUTION


def negate(a):
    return Negate()(a)
```

### `Transpose`: reverses the order of two axes (axis1, axis2), defaults to the last two axes (1 input, `axes` - tuple)

**Forward Pass**
**Example1**

If you have the following `ndarray`:

- **Ndarray `a`**: `np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])`
- **Axes**: `(0, 2)`

The transposition would result in:

- **Result**: `np.array([[[1, 5], [3, 7]], [[2, 6], [4, 8]]])`

**Example2**

If you have the following `ndarray`:

-   **Ndarray `a`**: `np.array([[1, 2, 3], [4, 5, 6]])`
-   **Axes**: `None` (defaults to the last two axes)

The transposition would result in:

-   **Result**: `np.array([[1, 4], [2, 5], [3, 6]])`
  
  **Backward Pass**
  During the backward pass, you want to calculate the gradient of the loss $\ell$ with respect to the input `a` of the `Transpose` operation.

-   `out_grad` represents $\frac{\partial \ell}{\partial f}$, which is the gradient of the loss $\ell$ with respect to the output $f$ of the` Transpose` operation.
-   Using the chain rule: $\frac{\partial \ell}{\partial a} = \frac{\partial \ell}{\partial f} \cdot \frac{\partial f}{\partial a}$

For transposition:

-   The gradient of the transposition is the transposition of the gradient using the same axes.

Combining these using the chain rule:

-   The gradient with respect to the input is simply the gradient transposed using the same axes.

```python
class Transpose(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        # reverses the order of two axes (axis1, axis2), defaults to the last two axes (1 input, `axes` - tuple)
        if self.axes is None:
            # Default to swapping the last two axes
            return array_api.swapaxes(a, -1, -2)
        else:
            # Swap the specified axes
            return array_api.swapaxes(a, self.axes[0], self.axes[1])
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        if self.axes is None:
            return transpose(out_grad, axes=(-1, -2))
        else:
            return transpose(out_grad, axes=self.axes)
        ### END YOUR SOLUTION


def transpose(a, axes=None):
    return Transpose(axes)(a)
```

> If we use `array_api.arange(a.ndim)`, it is creating an array of indices representing the axes of the input array `a`.
> -   If `a` is a 2D array (`a.ndim` is `2`), `self.axis` will be `array([0, 1])`.
> -   If `a` is a 3D array (`a.ndim` is `3`), `self.axis` will be `array([0, 1, 2])`.
>```python
> def compute(self, a):
> 	### BEGIN YOUR SOLUTION
> 	# reverses the order of two axes (axis1, axis2), defaults to the last two axes (1 input, `axes` - tuple)
>	self.axis = array_api.arange(a.ndim)
>	if self.axes is None:
>		self.axis[-1], self.axis[-2] = self.axis[-2], self.axis[-1]
>	else:
>		self.axis[self.axes[0]], self.axis[self.axes[1]] = self.axes[1], self.axes[0]
>	return array_api.transpose(a, self.axis)
>	### END YOUR SOLUTION
>```
> 
> Given the requirement that `Transpose` should reverse the order of two axes (axis1, axis2), defaulting to the last two axes, the implementation can be simplified to handle exactly this case. The `axes` parameter should either be `None` (default case) or a tuple of two integers specifying which axes to swap.
> 

>A shape of `(3, 2, 3)` represents a 3D array. The three numbers in the shape tuple indicate the dimensions of the array:
>
>-   The first dimension (axis 0) has a size of 3.
>-   The second dimension (axis 1) has a size of 2.
>-   The third dimension (axis 2) has a size of 3.

### Detailed Broadcasting Rules
1.  **Right Alignment of Shapes**:
    -   The shapes of the arrays are compared element-wise from the trailing (rightmost) dimension to the leading (leftmost) dimension.
2.  **Compatibility**:
    -   Two dimensions are compatible if they are equal or if one of them is 1.
3.  **Expansion**:
    -   If a dimension of one array is 1, it can be expanded to match the dimension of the other array.
  
-   **Trailing Dimensions**: The dimensions at the end of the shape tuple. These are compared first when determining broadcasting compatibility.
-   **Leading Dimensions**: The dimensions at the beginning of the shape tuple. These are compared after the trailing dimensions when determining broadcasting compatibility.

**Example1: Can Broadcasting**
```python
import numpy as np
c = np.array([1, 2, 3])
print(c.shape) #(3,)
new_shape = (2,3)
print(np.broadcast_to(c, new_shape))
```

1.  **Original Shape**: `(3,)`
    
    -   This shape has only one dimension: `3`. For broadcasting purposes, it can be treated as `(1, 3)`.
2.  **Target Shape**: `(2, 3)`

**Comparing the Trailing Dimensions**:

-   The rightmost dimensions (trailing dimensions) of both shapes are `3` and `3`, which are compatible because they are equal.

**Comparing the Leading Dimensions**:

-   The next dimensions to the left (leading dimensions) are `1` (from the original shape treated as `(1, 3)`) and `2` (from the target shape `(2, 3)`).
-   These dimensions are compatible because `1` can be broadcasted to `2`.

**Example2: Can't Broadcasting**
```python
import numpy as np
c = np.array([1, 2, 3])
print(c.shape) #(3,)
new_shape = (2,3)
print(np.broadcast_to(c, new_shape))
```
**Comparison**

-   **Trailing Dimensions**: `3` (original) vs. `2` (target) – these are not compatible because they are not equal and neither is 1.
-   **Leading Dimensions**: `1` (original) vs. `3` (target) – these are compatible because 1 can be expanded to 3.

Since the trailing dimensions do not match and are not compatible, broadcasting cannot proceed.


