## Question 1: Implementing forward computation

## Question 1: Implementing forward computation [10 pts]


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

#### `PowerScalar`: raise input to an integer (scalar) power
```python

```

#### `EWiseDiv`: true division of the inputs, element-wise (2 inputs)

#### `DivScalar`: true division of the input by a scalar, element-wise (1 input, `scalar` - number)
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

#### `MatMul`: matrix multiplication of the inputs (2 inputs)

#### `Summation`: sum of array elements over given axes (1 input, `axes` - tuple)

#### `BroadcastTo`: broadcast an array to a new shape (1 input, `shape` - tuple)

#### `Reshape`: gives a new shape to an array without changing its data (1 input, `shape` - tuple)

#### `Negate`: numerical negative, element-wise (1 input)

#### `Transpose`: reverses the order of two axes (axis1, axis2), defaults to the last two axes (1 input, `axes` - tuple)
