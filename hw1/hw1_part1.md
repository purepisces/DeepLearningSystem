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

### `PowerScalar`: raise input to an integer (scalar) power
**Example**

If you have the following `ndarray` and scalar:

-   **Ndarray**: `np.array([2, 3, 4])`
-   **Scalar**: `3`

The element-wise power would result in:

-   **Result**: `np.array([2**3, 3**3, 4**3])` which is `np.array([8, 27, 64])`

```python
class PowerScalar(TensorOp):
    """Op raise a tensor to an (integer) power."""

    def __init__(self, scalar: int):
        self.scalar = scalar

    def compute(self, a: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        return array_api.power(a, self.scalar)
        ### END YOUR SOLUTION
```
### `EWiseDiv`: true division of the inputs, element-wise (2 inputs)

**Example**

If you have the following `ndarrays`:

- **Ndarray `a`**: `np.array([10, 20, 30])`
- **Ndarray `b`**: `np.array([2, 4, 6])`

The element-wise division would result in:

- **Result**: `np.array([10/2, 20/4, 30/6])` which is `np.array([5, 5, 5])`

```python
class EWiseDiv(TensorOp):
    """Op to element-wise divide two nodes."""

    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        return a / b
        ### END YOUR SOLUTION
```
### `DivScalar`: true division of the input by a scalar, element-wise (1 input, `scalar` - number)

**Example**

If you have the following `ndarray` and scalar:

- **Ndarray**: `np.array([10, 20, 30])`
- **Scalar**: `2`

The element-wise division would result in:

- **Result**: `np.array([10/2, 20/2, 30/2])` which is `np.array([5, 10, 15])`

```python
class DivScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return a / self.scalar
        ### END YOUR SOLUTION
```
### `MatMul`: matrix multiplication of the inputs (2 inputs)

**Example**

If you have the following `ndarrays`:

- **Ndarray `a`**: `np.array([[1, 2], [3, 4]])`
- **Ndarray `b`**: `np.array([[5, 6], [7, 8]])`

The matrix multiplication would result in:

- **Result**: `np.array([[1*5 + 2*7, 1*6 + 2*8], [3*5 + 4*7, 3*6 + 4*8]])` which is `np.array([[19, 22], [43, 50]])`

```python
class MatMul(TensorOp):
    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        return array_api.matmul(a, b)
        ### END YOUR SOLUTION
```
> If return a*b, then it become np.array([[5, 12], [21, 32]]), incorrect result.

### `Summation`: sum of array elements over given axes (1 input, `axes` - tuple)

**Example**

If you have the following `ndarray`:

- **Ndarray `a`**: `np.array([[1, 2, 3], [4, 5, 6]])`
- **Axes**: `(0,)`

The summation over the specified axes would result in:

- **Result**: `np.array([1+4, 2+5, 3+6])` which is `np.array([5, 7, 9])`

```python
class Summation(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        return array_api.sum(a, axis=self.axes)
```

### `BroadcastTo`: broadcast an array to a new shape (1 input, `shape` - tuple)

**Example**

If you have the following `ndarray`:

- **Ndarray `a`**: `np.array([1, 2, 3])`
- **Shape**: `(3, 3)`

The broadcasting to the specified shape would result in:

- **Result**: `np.array([[1, 2, 3], [1, 2, 3], [1, 2, 3]])`

```python
class BroadcastTo(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        return array_api.broadcast_to(a, self.shape)
```

### `Reshape`: Gives a new shape to an array without changing its data (1 input, `shape` - tuple)

**Example**
If you have the following `ndarray`:

- **Ndarray `a`**: `np.array([[1, 2, 3], [4, 5, 6]])`
- **Shape**: `(3, 2)`

The reshaping to the specified shape would result in:

- **Result**: `np.array([[1, 2], [3, 4], [5, 6]])`

```python
class Reshape(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        return array_api.reshape(a, self.shape)
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

If you have the following `ndarray`:

- **Ndarray `a`**: `np.array([1, -2, 3])`

The negation would result in:

- **Result**: `np.array([-1, 2, -3])`

```python
class Negate(TensorOp):
    def compute(self, a):
        return -a
```

### `Transpose`: reverses the order of two axes (axis1, axis2), defaults to the last two axes (1 input, `axes` - tuple)

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
```

> if we use
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
> 'array_api.arange(a.ndim)`, it is creating an array of indices representing the axes of the input array `a`.
> -   If `a` is a 2D array (`a.ndim` is `2`), `self.axis` will be `array([0, 1])`.
> -   If `a` is a 3D array (`a.ndim` is `3`), `self.axis` will be `array([0, 1, 2])`.
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

## Question 2: Implementing backward computation

