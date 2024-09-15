Modified these function's code different from previous homework, also take a look at tanh's gradient, split's gradient, stack's gradient. **check if previous homework need to modify to this version**
```css
matmul
summation
transpose
LogSumExp
broadcast to in ndarray.py
reduce_view_out in ndarray.py:  if axis is None:
            view = self.compact().reshape((1,) * (self.ndim - 1) + (prod(self.shape),))
            out = NDArray.make((1,) * self.ndim if keepdims else (1,), device=self.device)
```
**Code Implementation**
ops.mathematics.py
```python
"""Operator implementations."""

from numbers import Number
from typing import Optional, List, Tuple, Union

from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp
import numpy

# NOTE: we will import numpy as the array_api
# as the backend for our computations, this line will change in later homeworks

from ..backend_selection import array_api, BACKEND 
from .ops_tuple import *

class EWiseAdd(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a + b

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad, out_grad


def add(a, b):
    return EWiseAdd()(a, b)


class AddScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a + self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad


def add_scalar(a, scalar):
    return AddScalar(scalar)(a)


class EWiseMul(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a * b

    def gradient(self, out_grad: Tensor, node: Tensor):
        lhs, rhs = node.inputs
        return out_grad * rhs, out_grad * lhs


def multiply(a, b):
    return EWiseMul()(a, b)


class MulScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a * self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return (out_grad * self.scalar,)


def mul_scalar(a, scalar):
    return MulScalar(scalar)(a)


class EWisePow(TensorOp):
    """Op to element-wise raise a tensor to a power."""

    def compute(self, a: NDArray, b: NDArray) -> NDArray:
        return a**b

    def gradient(self, out_grad, node):
        if not isinstance(node.inputs[0], NDArray) or not isinstance(
            node.inputs[1], NDArray
        ):
            raise ValueError("Both inputs must be tensors (NDArray).")

        a, b = node.inputs[0], node.inputs[1]
        grad_a = out_grad * b * (a ** (b - 1))
        grad_b = out_grad * (a**b) * log(a)
        return grad_a, grad_b

def power(a, b):
    return EWisePow()(a, b)


class PowerScalar(TensorOp):
    """Op raise a tensor to an (integer) power."""

    def __init__(self, scalar: int):
        self.scalar = scalar

    def compute(self, a: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        return a ** self.scalar
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a = node.inputs[0]
        return self.scalar * power_scalar(a, self.scalar - 1) * out_grad
        ### END YOUR SOLUTION


def power_scalar(a, scalar):
    return PowerScalar(scalar)(a)


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


class Transpose(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        # If no axes are provided, swap the last two axes
        if self.axes is None:
            # Default to swapping the last two axes
            new_axes = list(range(len(a.shape))) # Or write as list(range(a.ndim))
            new_axes[-1], new_axes[-2] = new_axes[-2], new_axes[-1]
        else:
            # Swap the specified axes
            new_axes = list(range(len(a.shape))) # Or write as list(range(a.ndim))
            new_axes[self.axes[0]], new_axes[self.axes[1]] = new_axes[self.axes[1]], new_axes[self.axes[0]]
        
        # Ensure the array is compact before permuting
        return a.compact().permute(tuple(new_axes))
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


class BroadcastTo(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.broadcast_to(a, self.shape)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        input_shape = node.inputs[0].shape
        output_shape = out_grad.shape
        # Align input_shape to output_shape by filling missing dimensions with 1s on the left
        aligned_shape = [1] * (len(output_shape) - len(input_shape)) + list(input_shape)
        grad = out_grad
        # Sum over expanded dimensions and reshape to input shape
        for i in range(len(output_shape)):
            if output_shape[i] != aligned_shape[i]:
                grad = summation(grad, axes=(i,))
        return reshape(grad, input_shape)
        ### END YOUR SOLUTION


def broadcast_to(a, shape):
    return BroadcastTo(shape)(a)


class Summation(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes
        # Ensure axes is always a tuple if provided as an int
        if isinstance(axes, int):
            self.axes = (axes,)

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.sum(a, axis=self.axes)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        input_shape = node.inputs[0].shape

        # Initialize the shape to the input shape
        grad_shape = list(input_shape)

        if self.axes is not None:
            for axis in self.axes:
                grad_shape[axis] = 1
        else:
            # If axes is None, summation was over all axes, set all dimensions to 1
            grad_shape = [1] * len(grad_shape)
        
        # Reshape out_grad to the calculated shape
        reshaped_grad = reshape(out_grad, grad_shape)
        
        # Broadcast the gradient to match the input shape
        return broadcast_to(reshaped_grad, input_shape)
        ### END YOUR SOLUTION


def summation(a, axes=None):
    return Summation(axes)(a)
    

class MatMul(TensorOp):
    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        return a @ b
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

class Tanh(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.tanh(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad * (1 + (-node ** 2)),  # Equivalent to 1 - node**2, Instead of using the minus operator (-), implement it with + and negation (-), as direct subtraction can cause issues in some implementations.
        ### END YOUR SOLUTION

def tanh(a):
    return Tanh()(a)


class Stack(TensorOp):
    def __init__(self, axis: int):
        """
        Concatenates a sequence of arrays along a new dimension.
        Parameters:
        axis - dimension to concatenate along
        All arrays need to be of the same size.
        """
        self.axis = axis

    def compute(self, args: TensorTuple) -> Tensor:
        ### BEGIN YOUR SOLUTION
        # Ensure there is at least one tensor and all have the same shape
        assert len(args) > 0, "The input tensor list cannot be empty."
        base_shape = args[0].shape
        for arg in args:
            assert arg.shape == base_shape, "All tensors must have the same shape."

        # Compute stacked shape by inserting the new axis
        stacked_shape = list(base_shape)  
        stacked_shape.insert(self.axis, len(args))  

        # Create an empty array for the stacked result
        result = array_api.empty(stacked_shape, dtype=args[0].dtype, device=args[0].device)

        # Insert each argument into the result along the new axis
        for i, arg in enumerate(args):
            # Create a tuple of slices that matches the stacked axis
            slices = [slice(None)] * len(stacked_shape)
            slices[self.axis] = i
            result[tuple(slices)] = arg # e.g. tuple(slices) == (slice(None), 0, slice(None))

        return result
        ### END YOUR SOLUTION


    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return split(out_grad, axis=self.axis)
        ### END YOUR SOLUTION


def stack(args, axis):
    return Stack(axis)(make_tuple(*args))


class Split(TensorTupleOp):
    def __init__(self, axis: int):
        """
        Splits a tensor along an axis into a tuple of tensors.
        (The "inverse" of Stack)
        Parameters:
        axis - dimension to split
        """
        self.axis = axis

    def compute(self, A):
        ### BEGIN YOUR SOLUTION
        axis_size = A.shape[self.axis]  
        split_tensors = []
        # Copy shape and remove the axis we are splitting on
        output_shape = list(A.shape) 
        output_shape.pop(self.axis) 
    
        # Iterate through each slice along the axis
        for i in range(axis_size): 
            # Create a list of slices, defaulting to slice(None) for all axes
            slices = [slice(None)] * len(A.shape) 
            # Set the slice for the axis we are splitting along
            slices[self.axis] = i 
            # Extract the slice and reshape it to the reduced shape, then append to result list
            split_tensors.append(A[tuple(slices)].compact().reshape(output_shape))
        return tuple(split_tensors)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return stack(out_grad, axis=self.axis)
        ### END YOUR SOLUTION


def split(a, axis):
    return Split(axis)(a)


class Flip(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


def flip(a, axes):
    return Flip(axes)(a)


class Dilate(TensorOp):
    def __init__(self, axes: tuple, dilation: int):
        self.axes = axes
        self.dilation = dilation

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


def dilate(a, axes, dilation):
    return Dilate(axes, dilation)(a)


class UnDilate(TensorOp):
    def __init__(self, axes: tuple, dilation: int):
        self.axes = axes
        self.dilation = dilation

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


def undilate(a, axes, dilation):
    return UnDilate(axes, dilation)(a)


class Conv(TensorOp):
    def __init__(self, stride: Optional[int] = 1, padding: Optional[int] = 0):
        self.stride = stride
        self.padding = padding

    def compute(self, A, B):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


def conv(a, b, stride=1, padding=1):
    return Conv(stride, padding)(a, b)
```
ops.logarithmic.py
```python
from typing import Optional
from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp

from .ops_mathematic import *

from ..backend_selection import array_api, BACKEND 

class LogSoftmax(TensorOp):
    def compute(self, Z):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


def logsoftmax(a):
    return LogSoftmax()(a)
    

class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes
        # Ensure axes is always a tuple if provided as an int
        if isinstance(axes, int):
            self.axes = (axes,)

    def compute(self, Z):
        ### BEGIN YOUR SOLUTION
        max_z_with_dim = Z.max(axis=self.axes, keepdims=True) 
        shifted_log_sum_exp = array_api.log(array_api.sum(array_api.exp(Z - max_z_with_dim.broadcast_to(Z.shape)), axis=self.axes, keepdims=False))
        max_z_no_dim = max_z_with_dim.reshape(shifted_log_sum_exp.shape)
        out = shifted_log_sum_exp + max_z_no_dim
        return out
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        if self.axes is None:
            self.axes = tuple(range(len(node.inputs[0].shape)))
        z = node.inputs[0]
        shape = [1 if i in self.axes else z.shape[i] for i in range(len(z.shape))]
        gradient = exp(z - node.reshape(shape).broadcast_to(z.shape))
        return out_grad.reshape(shape).broadcast_to(z.shape)*gradient
        ### END YOUR SOLUTION


def logsumexp(a, axes=None):
    return LogSumExp(axes=axes)(a)
```
ndarray.py
```
import operator
import math
from functools import reduce
import numpy as np
from . import ndarray_backend_numpy
from . import ndarray_backend_cpu


# math.prod not in Python 3.7
def prod(x):
    return reduce(operator.mul, x, 1)


class BackendDevice:
    """A backend device, wrapps the implementation module."""

    def __init__(self, name, mod):
        self.name = name
        self.mod = mod

    def __eq__(self, other):
        return self.name == other.name

    def __repr__(self):
        return self.name + "()"

    def __getattr__(self, name):
        return getattr(self.mod, name)

    def enabled(self):
        return self.mod is not None

    def randn(self, *shape, dtype="float32"):
        # note: numpy doesn't support types within standard random routines, and
        # .astype("float32") does work if we're generating a singleton
        return NDArray(np.random.randn(*shape).astype(dtype), device=self)

    def rand(self, *shape, dtype="float32"):
        # note: numpy doesn't support types within standard random routines, and
        # .astype("float32") does work if we're generating a singleton
        return NDArray(np.random.rand(*shape).astype(dtype), device=self)

    def one_hot(self, n, i, dtype="float32"):
        return NDArray(np.eye(n, dtype=dtype)[i], device=self)

    def empty(self, shape, dtype="float32"):
        dtype = "float32" if dtype is None else dtype
        assert dtype == "float32"
        return NDArray.make(shape, device=self)

    def full(self, shape, fill_value, dtype="float32"):
        dtype = "float32" if dtype is None else dtype
        assert dtype == "float32"
        arr = self.empty(shape, dtype)
        arr.fill(fill_value)
        return arr


def cuda():
    """Return cuda device"""
    try:
        from . import ndarray_backend_cuda

        return BackendDevice("cuda", ndarray_backend_cuda)
    except ImportError:
        return BackendDevice("cuda", None)


def cpu_numpy():
    """Return numpy device"""
    return BackendDevice("cpu_numpy", ndarray_backend_numpy)


def cpu():
    """Return cpu device"""
    return BackendDevice("cpu", ndarray_backend_cpu)


def default_device():
    return cpu_numpy()


def all_devices():
    """return a list of all available devices"""
    return [cpu(), cuda(), cpu_numpy()]


class NDArray:
    """A generic ND array class that may contain multipe different backends
    i.e., a Numpy backend, a native CPU backend, or a GPU backend.

    This class will only contains those functions that you need to implement
    to actually get the desired functionality for the programming examples
    in the homework, and no more.

    For now, for simplicity the class only supports float32 types, though
    this can be extended if desired.
    """

    def __init__(self, other, device=None):
        """Create by copying another NDArray, or from numpy"""
        if isinstance(other, NDArray):
            # create a copy of existing NDArray
            if device is None:
                device = other.device
            self._init(other.to(device) + 0.0)  # this creates a copy
        elif isinstance(other, np.ndarray):
            # create copy from numpy array
            device = device if device is not None else default_device()
            array = self.make(other.shape, device=device)
            array.device.from_numpy(np.ascontiguousarray(other), array._handle)
            self._init(array)
        else:
            # see if we can create a numpy array from input
            array = NDArray(np.array(other), device=device)
            self._init(array)

    def _init(self, other):
        self._shape = other._shape
        self._strides = other._strides
        self._offset = other._offset
        self._device = other._device
        self._handle = other._handle

    @staticmethod
    def compact_strides(shape):
        """Utility function to compute compact strides"""
        stride = 1
        res = []
        for i in range(1, len(shape) + 1):
            res.append(stride)
            stride *= shape[-i]
        return tuple(res[::-1])

    @staticmethod
    def make(shape, strides=None, device=None, handle=None, offset=0):
        """Create a new NDArray with the given properties.  This will allocation the
        memory if handle=None, otherwise it will use the handle of an existing
        array."""
        array = NDArray.__new__(NDArray)
        array._shape = tuple(shape)
        array._strides = NDArray.compact_strides(shape) if strides is None else strides
        array._offset = offset
        array._device = device if device is not None else default_device()
        if handle is None:
            array._handle = array.device.Array(prod(shape))
        else:
            array._handle = handle
        return array

    ### Properies and string representations
    @property
    def shape(self):
        return self._shape

    @property
    def strides(self):
        return self._strides

    @property
    def device(self):
        return self._device

    @property
    def dtype(self):
        # only support float32 for now
        return "float32"

    @property
    def ndim(self):
        """Return number of dimensions."""
        return len(self._shape)

    @property
    def size(self):
        return prod(self._shape)

    def __repr__(self):
        return "NDArray(" + self.numpy().__str__() + f", device={self.device})"

    def __str__(self):
        return self.numpy().__str__()

    ### Basic array manipulation
    def fill(self, value):
        """Fill (in place) with a constant value."""
        self._device.fill(self._handle, value)

    def to(self, device):
        """Convert between devices, using to/from numpy calls as the unifying bridge."""
        if device == self.device:
            return self
        else:
            return NDArray(self.numpy(), device=device)

    def numpy(self):
        """convert to a numpy array"""
        return self.device.to_numpy(
            self._handle, self.shape, self.strides, self._offset
        )

    def is_compact(self):
        """Return true if array is compact in memory and internal size equals product
        of the shape dimensions"""
        return (
            self._strides == self.compact_strides(self._shape)
            and prod(self.shape) == self._handle.size
        )

    def compact(self):
        """Convert a matrix to be compact"""
        if self.is_compact():
            return self
        else:
            out = NDArray.make(self.shape, device=self.device)
            self.device.compact(
                self._handle, out._handle, self.shape, self.strides, self._offset
            )
            return out

    def as_strided(self, shape, strides):
        """Restride the matrix without copying memory."""
        assert len(shape) == len(strides)
        return NDArray.make(
            shape, strides=strides, device=self.device, handle=self._handle
        )

    @property
    def flat(self):
        return self.reshape((self.size,))

    def reshape(self, new_shape):
        """
        Reshape the matrix without copying memory.  This will return a matrix
        that corresponds to a reshaped array but points to the same memory as
        the original array.

        Raises:
            ValueError if product of current shape is not equal to the product
            of the new shape, or if the matrix is not compact.

        Args:
            new_shape (tuple): new shape of the array

        Returns:
            NDArray : reshaped array; this will point to thep
        """

        ### BEGIN YOUR SOLUTION
        if prod(self.shape) != prod(new_shape):
            raise ValueError("Cannot reshape array of size {} into shape {}".format(self.shape, new_shape))
    
        if not self.is_compact(): 
    	    raise ValueError("Cannot reshape a non-compact array. Please compact the array first.")
        
        new_strides = NDArray.compact_strides(new_shape) 
        return self.as_strided(new_shape, new_strides)
        ### END YOUR SOLUTION

    def permute(self, new_axes):
        """
        Permute order of the dimensions.  new_axes describes a permuation of the
        existing axes, so e.g.:
          - If we have an array with dimension "BHWC" then .permute((0,3,1,2))
            would convert this to "BCHW" order.
          - For a 2D array, .permute((1,0)) would transpose the array.
        Like reshape, this operation should not copy memory, but achieves the
        permuting by just adjusting the shape/strides of the array.  That is,
        it returns a new array that has the dimensions permuted as desired, but
        which points to the same memroy as the original array.

        Args:
            new_axes (tuple): permuation order of the dimensions

        Returns:
            NDarray : new NDArray object with permuted dimensions, pointing
            to the same memory as the original NDArray (i.e., just shape and
            strides changed).
        """

        ### BEGIN YOUR SOLUTION
        new_shape = tuple(self._shape[i] for i in new_axes)
        new_strides = tuple(self._strides[i] for i in new_axes)
        return self.as_strided(new_shape, new_strides)
        ### END YOUR SOLUTION

    def broadcast_to(self, new_shape):
        """
        Broadcast an array to a new shape.  new_shape's elements must be the
        same as the original shape, except for dimensions in the self where
        the size = 1 (which can then be broadcast to any size).  As with the
        previous calls, this will not copy memory, and just achieves
        broadcasting by manipulating the strides.

        Raises:
            assertion error if new_shape[i] != shape[i] for all i where
            shape[i] != 1

        Args:
            new_shape (tuple): shape to broadcast to

        Returns:
            NDArray: the new NDArray object with the new broadcast shape; should
            point to the same memory as the original array.
        """
        ### BEGIN YOUR SOLUTION
        # Handle the case where the array is one-dimensional
        if self.ndim == 1:  # One-dimensional array
            # Ensure the last dimension in new_shape matches or is 1
            assert self._shape[0] == 1 or self._shape[0] == new_shape[-1], "Shapes cannot be broadcast together."
            # If broadcasting from (1,) to a higher dimension, strides should be all zeros
            new_strides = (0,) * len(new_shape)
            return self.as_strided(new_shape, new_strides)
            
        # Handle higher-dimensional arrays
        new_strides = []
        for i in range(len(self._shape)):
            if self._shape[i] == 1:
                # Dimension can be broadcast
                new_strides.append(0)
            else:
                # Dimension must match exactly
                assert new_shape[i] == self._shape[i], "Shapes cannot be broadcast together."
                new_strides.append(self._strides[i])
        return self.as_strided(new_shape, tuple(new_strides))
        ### END YOUR SOLUTION

    ### Get and set elements

    def process_slice(self, sl, dim):
        """Convert a slice to an explicit start/stop/step"""
        start, stop, step = sl.start, sl.stop, sl.step
        if start == None:
            start = 0
        if start < 0:
            start = self.shape[dim]
        if stop == None:
            stop = self.shape[dim]
        if stop < 0:
            stop = self.shape[dim] + stop
        if step == None:
            step = 1

        # we're not gonna handle negative strides and that kind of thing
        assert stop > start, "Start must be less than stop"
        assert step > 0, "No support for  negative increments"
        return slice(start, stop, step)

    def __getitem__(self, idxs):
        """
        The __getitem__ operator in Python allows us to access elements of our
        array.  When passed notation such as a[1:5,:-1:2,4,:] etc, Python will
        convert this to a tuple of slices and integers (for singletons like the
        '4' in this example).  Slices can be a bit odd to work with (they have
        three elements .start .stop .step), which can be None or have negative
        entries, so for simplicity we wrote the code for you to convert these
        to always be a tuple of slices, one of each dimension.

        For this tuple of slices, return an array that subsets the desired
        elements.  As before, this can be done entirely through compute a new
        shape, stride, and offset for the new "view" into the original array,
        pointing to the same memory

        Raises:
            AssertionError if a slice has negative size or step, or if number
            of slices is not equal to the number of dimension (the stub code
            already raises all these errors.

        Args:
            idxs tuple: (after stub code processes), a tuple of slice elements
            coresponding to the subset of the matrix to get

        Returns:
            NDArray: a new NDArray object corresponding to the selected
            subset of elements.  As before, this should not copy memroy but just
            manipulate the shape/strides/offset of the new array, referecing
            the same array as the original one.
        """

        # handle singleton as tuple, everything as slices
        if not isinstance(idxs, tuple):
            idxs = (idxs,)
        idxs = tuple(
            [
                self.process_slice(s, i) if isinstance(s, slice) else slice(s, s + 1, 1)
                for i, s in enumerate(idxs)
            ]
        )
        assert len(idxs) == self.ndim, "Need indexes equal to number of dimensions"

        ### BEGIN YOUR SOLUTION
        new_shape = []  # List to store the new shape of the array view.
        new_strides = []  # List to store the new strides for the array view.
        new_offset = self._offset  # Start with the original offset in memory.
        # Note that idxs now is a tuple of slices
        # Loop through each dimension and its corresponding slice
        for i, sl in enumerate(idxs):
            # Calculate the size(number of elements) of this dimension in the new array view.
            new_shape.append((sl.stop - sl.start + (sl.step - 1)) // sl.step)
            # Calculate the new stride for this dimension by scaling the original stride by the step size from the slice.
            new_strides.append(self._strides[i] * sl.step)
            # Adjust the offset to account for the starting position of this slice in the original array's memory.
            new_offset += self._strides[i] * sl.start
    
        # Convert the shape and strides lists to tuples.
        new_shape = tuple(new_shape)
        new_strides = tuple(new_strides)
        
        # Create and return a new NDArray with the calculated shape, strides, and offset.
        return self.make(shape=new_shape, strides=new_strides, device=self.device, handle=self._handle, offset=new_offset)
        ### END YOUR SOLUTION

    def __setitem__(self, idxs, other):
        """Set the values of a view into an array, using the same semantics
        as __getitem__()."""
        view = self.__getitem__(idxs)
        if isinstance(other, NDArray):
            assert prod(view.shape) == prod(other.shape)
            self.device.ewise_setitem(
                other.compact()._handle,
                view._handle,
                view.shape,
                view.strides,
                view._offset,
            )
        else:
            self.device.scalar_setitem(
                prod(view.shape),
                other,
                view._handle,
                view.shape,
                view.strides,
                view._offset,
            )

    ### Collection of elementwise and scalar function: add, multiply, boolean, etc

    def ewise_or_scalar(self, other, ewise_func, scalar_func):
        """Run either an elementwise or scalar version of a function,
        depending on whether "other" is an NDArray or scalar
        """
        out = NDArray.make(self.shape, device=self.device)
        if isinstance(other, NDArray):
            assert self.shape == other.shape, "operation needs two equal-sized arrays"
            ewise_func(self.compact()._handle, other.compact()._handle, out._handle)
        else:
            scalar_func(self.compact()._handle, other, out._handle)
        return out

    def __add__(self, other):
        return self.ewise_or_scalar(
            other, self.device.ewise_add, self.device.scalar_add
        )

    __radd__ = __add__

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return other + (-self)

    def __mul__(self, other):
        return self.ewise_or_scalar(
            other, self.device.ewise_mul, self.device.scalar_mul
        )

    __rmul__ = __mul__

    def __truediv__(self, other):
        return self.ewise_or_scalar(
            other, self.device.ewise_div, self.device.scalar_div
        )

    def __neg__(self):
        return self * (-1)

    def __pow__(self, other):
        out = NDArray.make(self.shape, device=self.device)
        self.device.scalar_power(self.compact()._handle, other, out._handle)
        return out

    def maximum(self, other):
        return self.ewise_or_scalar(
            other, self.device.ewise_maximum, self.device.scalar_maximum
        )

    ### Binary operators all return (0.0, 1.0) floating point values, could of course be optimized
    def __eq__(self, other):
        return self.ewise_or_scalar(other, self.device.ewise_eq, self.device.scalar_eq)

    def __ge__(self, other):
        return self.ewise_or_scalar(other, self.device.ewise_ge, self.device.scalar_ge)

    def __ne__(self, other):
        return 1 - (self == other)

    def __gt__(self, other):
        return (self >= other) * (self != other)

    def __lt__(self, other):
        return 1 - (self >= other)

    def __le__(self, other):
        return 1 - (self > other)

    ### Elementwise functions

    def log(self):
        out = NDArray.make(self.shape, device=self.device)
        self.device.ewise_log(self.compact()._handle, out._handle)
        return out

    def exp(self):
        out = NDArray.make(self.shape, device=self.device)
        self.device.ewise_exp(self.compact()._handle, out._handle)
        return out

    def tanh(self):
        out = NDArray.make(self.shape, device=self.device)
        self.device.ewise_tanh(self.compact()._handle, out._handle)
        return out

    ### Matrix multiplication
    def __matmul__(self, other):
        """Matrix multplication of two arrays.  This requires that both arrays
        be 2D (i.e., we don't handle batch matrix multiplication), and that the
        sizes match up properly for matrix multiplication.

        In the case of the CPU backend, you will implement an efficient "tiled"
        version of matrix multiplication for the case when all dimensions of
        the array are divisible by self.device.__tile_size__.  In this case,
        the code below will restride and compact the matrix into tiled form,
        and then pass to the relevant CPU backend.  For the CPU version we will
        just fall back to the naive CPU implementation if the array shape is not
        a multiple of the tile size

        The GPU (and numpy) versions don't have any tiled version (or rather,
        the GPU version will just work natively by tiling any input size).
        """

        assert self.ndim == 2 and other.ndim == 2
        assert self.shape[1] == other.shape[0]

        m, n, p = self.shape[0], self.shape[1], other.shape[1]

        # if the matrix is aligned, use tiled matrix multiplication
        if hasattr(self.device, "matmul_tiled") and all(
            d % self.device.__tile_size__ == 0 for d in (m, n, p)
        ):

            def tile(a, tile):
                return a.as_strided(
                    (a.shape[0] // tile, a.shape[1] // tile, tile, tile),
                    (a.shape[1] * tile, tile, a.shape[1], 1),
                )

            t = self.device.__tile_size__
            a = tile(self.compact(), t).compact()
            b = tile(other.compact(), t).compact()
            out = NDArray.make((a.shape[0], b.shape[1], t, t), device=self.device)
            self.device.matmul_tiled(a._handle, b._handle, out._handle, m, n, p)

            return (
                out.permute((0, 2, 1, 3))
                .compact()
                .reshape((self.shape[0], other.shape[1]))
            )

        else:
            out = NDArray.make((m, p), device=self.device)
            self.device.matmul(
                self.compact()._handle, other.compact()._handle, out._handle, m, n, p
            )
            return out

    ### Reductions, i.e., sum/max over all element or over given axis
    def reduce_view_out(self, axis, keepdims=False):
        """ Return a view to the array set up for reduction functions and output array. """
        if isinstance(axis, tuple) and not axis:
            raise ValueError("Empty axis in reduce")

        if axis is None:
            view = self.compact().reshape((1,) * (self.ndim - 1) + (prod(self.shape),))
            out = NDArray.make((1,) * self.ndim if keepdims else (1,), device=self.device)

        else:
            if isinstance(axis, (tuple, list)):
                assert len(axis) == 1, "Only support reduction over a single axis"
                axis = axis[0]

            view = self.permute(
                tuple([a for a in range(self.ndim) if a != axis]) + (axis,)
            )
            out = NDArray.make(
                tuple([1 if i == axis else s for i, s in enumerate(self.shape)])
                if keepdims else
                tuple([s for i, s in enumerate(self.shape) if i != axis]),
                device=self.device,
            )
        return view, out

    def sum(self, axis=None, keepdims=False):
        view, out = self.reduce_view_out(axis, keepdims=keepdims)
        self.device.reduce_sum(view.compact()._handle, out._handle, view.shape[-1])
        return out

    def max(self, axis=None, keepdims=False):
        view, out = self.reduce_view_out(axis, keepdims=keepdims)
        self.device.reduce_max(view.compact()._handle, out._handle, view.shape[-1])
        return out

    def flip(self, axes):
        """
        Flip this ndarray along the specified axes.
        Note: compact() before returning.
        """
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION

    def pad(self, axes):
        """
        Pad this ndarray by zeros by the specified amount in `axes`,
        which lists for _all_ axes the left and right padding amount, e.g.,
        axes = ( (0, 0), (1, 1), (0, 0)) pads the middle axis with a 0 on the left and right side.
        """
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION

def array(a, dtype="float32", device=None):
    """Convenience methods to match numpy a bit more closely."""
    dtype = "float32" if dtype is None else dtype
    assert dtype == "float32"
    return NDArray(a, device=device)


def empty(shape, dtype="float32", device=None):
    device = device if device is not None else default_device()
    return device.empty(shape, dtype)


def full(shape, fill_value, dtype="float32", device=None):
    device = device if device is not None else default_device()
    return device.full(shape, fill_value, dtype)


def broadcast_to(array, new_shape):
    return array.broadcast_to(new_shape)


def reshape(array, new_shape):
    return array.reshape(new_shape)


def maximum(a, b):
    return a.maximum(b)


def log(a):
    return a.log()


def exp(a):
    return a.exp()


def tanh(a):
    return a.tanh()


def sum(a, axis=None, keepdims=False):
    return a.sum(axis=axis, keepdims=keepdims)


def flip(a, axes):
    return a.flip(axes)
```
# 10-714 Homework 4

In this homework, you will leverage all of the components built in the last three homeworks to solve some modern problems with high performing network structures. We will start by adding a few new ops leveraging our new CPU/CUDA backends. Then, you will implement convolution, and a convolutional neural network to train a classifier on the CIFAR-10 image classification dataset. Then, you will implement recurrent and long-short term memory (LSTM) neural networks, and do word-level prediction language modeling on the Penn Treebank dataset.

As always, we will start by copying this notebook and getting the starting code.

Reminder: __you must save a copy in drive__.


## Part 1: ND Backend [10 pts]

Recall that in homework 2, the `array_api` was imported as `numpy`. In this part, the goal is to write the necessary operations with `array_api` imported from the needle backend `NDArray` in `python/needle/backend_ndarray/ndarray.py`. Make sure to copy the solutions for `reshape`, `permute`, `broadcast_to` and `__getitem__` from homework 3.

Fill in the following classes in `python/needle/ops_logarithmic.py` and `python/needle/ops_mathematic.py`:

- `PowerScalar`

- `EWiseDiv`

- `DivScalar`

- `Transpose`

- `Reshape`

- `BroadcastTo`

- `Summation`

- `MatMul`

- `Negate`

- `Log`

- `Exp`

- `ReLU`

- `LogSumExp`

- `Tanh` (new)

- `Stack` (new)

- `Split` (new)

  
Note that for most of these, you already wrote the solutions in the previous homework and you should not change most part of your previous solution, if issues arise, please check if the `array_api` function used is supported in the needle backend.

`TanhOp`, `Stack`, and `Split` are newly added. `Stack` concatenates same-sized tensors along a new axis, and `Split` undoes this operation. The gradients of the two operations can be written in terms of each other. We do not directly test `Split`, and only test the backward pass of `Stack` (for which we assume you used `Split`).


**Note:** You may want to make your Summation op support sums over multiple axes; you will likely need it for the backward pass of the BroadcastTo op if yours supports broadcasting over multiple axes at a time. However, this is more about ease of use than necessity, and we leave this decision up to you (there are no corresponding tests).

**Note:** Depending on your implementations, you may want to ensure that you call `.compact()` before reshaping arrays. (If this is necessary, you will run into corresponding error messages later in the assignment.)



___
## Explain `stack`

Let’s go through a more detailed example where the tensor `A` has the shape $4 \times 3$ (4 rows and 3 columns) and we'll stack it with another tensor `B` of the same shape. We’ll explore different values of `axis` to show how the `compute` method works.

### Explain With Example:
```python
A = [[ 1,  2,  3],   # shape (4, 3)
     [ 4,  5,  6],
     [ 7,  8,  9],
     [10, 11, 12]]

B = [[13, 14, 15],   # shape (4, 3)
     [16, 17, 18],
     [19, 20, 21],
     [22, 23, 24]]
```
We are now going to stack `A` and `B` along different axes and explain how the shape evolves.

1. **Stack Along `axis=0`**:

When `axis=0`, we are adding a new first dimension, and `A` and `B` will be stacked along that dimension. This means that `A` will be placed at index `0` in the first axis, and `B` at index `1`.

#### Step-by-Step:

-   The shape of `A` and `B` is initially `(4, 3)`.
-   We insert `len(args) = 2` at position `0` (axis=0).
    -   New shape: `(2, 4, 3)`.

#### Result:
```python
stack([A, B], axis=0)

# The result is:
[[[ 1,  2,  3],
  [ 4,  5,  6],
  [ 7,  8,  9],
  [10, 11, 12]],   # This is tensor A

 [[13, 14, 15],
  [16, 17, 18],
  [19, 20, 21],
  [22, 23, 24]]]   # This is tensor B
```
-   The first axis (axis 0) has size 2 because we stacked 2 tensors.
-   Each "slice" along axis 0 is a $4 \times 3$ matrix corresponding to one of the input tensors (`A` or `B`).

2. **Stack Along `axis=1`**:

When `axis=1`, we are adding a new second dimension (between rows and columns). This means we are stacking corresponding rows from `A` and `B` together.

#### Step-by-Step:

-   The shape of `A` and `B` is initially `(4, 3)`.
-   We insert `len(args) = 2` at position `1` (axis=1).
    -   New shape: `(4, 2, 3)`.

#### Result:
```python
stack([A, B], axis=1)

# The result is:
[[[ 1,  2,  3],  [13, 14, 15]],   # Stacking the first row of A and B
 [[ 4,  5,  6],  [16, 17, 18]],   # Stacking the second row of A and B
 [[ 7,  8,  9],  [19, 20, 21]],   # Stacking the third row of A and B
 [[10, 11, 12],  [22, 23, 24]]]   # Stacking the fourth row of A and B
```
-   The first axis (axis 0) still represents the rows (4 rows).
-   The second axis (axis 1) has size 2, representing the new dimension created by stacking `A` and `B` row-wise.
-   The third axis (axis 2) still represents the columns (3 columns).

3. **Stack Along `axis=2`**:

When `axis=2`, we are adding a new third dimension, meaning that the elements of `A` and `B` will be stacked within each row and column.

#### Step-by-Step:

-   The shape of `A` and `B` is initially `(4, 3)`.
-   We insert `len(args) = 2` at position `2` (axis=2).
    -   New shape: `(4, 3, 2)`.

#### Result:
```python
stack([A, B], axis=2)

# The result is:
[[[ 1, 13],  [ 2, 14],  [ 3, 15]],   # Stacking corresponding elements from A and B in each column
 [[ 4, 16],  [ 5, 17],  [ 6, 18]],   # Stacking corresponding elements from A and B in each column
 [[ 7, 19],  [ 8, 20],  [ 9, 21]],   # Stacking corresponding elements from A and B in each column
 [[10, 22],  [11, 23],  [12, 24]]]   # Stacking corresponding elements from A and B in each column
```
-   The first axis (axis 0) still represents the rows (4 rows).
-   The second axis (axis 1) still represents the columns (3 columns).
-   The third axis (axis 2) has size 2 because you’re stacking the corresponding elements from `A` and `B` within each row and column.

### Explain `ret[:, 3, :]`

-   **`:` along axis 0**: Select all rows (so both the first and second row will be included).
-   **`3` along axis 1**: Select the 3rd slice along axis 1 (the fourth sub-array, as Python uses 0-based indexing).
-   **`:` along axis 2**: Select all columns for each selected slice.

So, `ret[:, 3, :]` selects the **3rd slice** (sub-array) from each row, including all columns in that slice.

Example Setup:
```python
ret = [[[ 1,  2,  3],  [ 4,  5,  6],  [ 7,  8,  9],  [10, 11, 12]],  # First row (axis 0, index 0)
       [[13, 14, 15],  [16, 17, 18],  [19, 20, 21],  [22, 23, 24]]]  # Second row (axis 0, index 1)
```
This `ret` tensor has:

-   2 rows (axis 0),
-   4 slices per row (axis 1),
-   3 columns per slice (axis 2).

Final Output of `ret[:, 3, :]`:
```python
[[10, 11, 12],   # 3rd slice from the first row
 [22, 23, 24]]   # 3rd slice from the second row
```

### Explain why the result is NDArray not Tensor
The `result = array_api.empty()` line creates an `NDArray` (not a `Tensor`) because:

-   **NDArray** is responsible for numerical storage and computation.
-   **Tensor** is a higher-level structure that wraps around `NDArray` to add additional functionality like gradients and computational graph management.

___
## Explain Split

### Understanding Axes in Tensor with Examples

```python
A = Tensor([[1, 2, 3],
            [4, 5, 6]])
```
-   The **first axis (axis 0)** refers to the **rows** of the tensor.
-   The **second axis (axis 1)** refers to the **columns** of the tensor.

Let's break it down:

#### **Axis 0 (First Axis)**: Rows

-   The elements along axis 0 are the **rows** of the tensor. Each row is treated as a distinct element along axis 0.
    -   The first element (along axis 0) is the row `[1, 2, 3]`.
    -   The second element (along axis 0) is the row `[4, 5, 6]`.

So, the elements in axis 0 are:
```css
[1, 2, 3]  # First row
[4, 5, 6]  # Second row
```
#### **Axis 1 (Second Axis)**: Columns

-   The elements along axis 1 are the **columns** of the tensor. Each column is treated as a distinct element along axis 1.
    -   The first element (along axis 1) is the column `[1, 4]`.
    -   The second element (along axis 1) is the column `[2, 5]`.
    -   The third element (along axis 1) is the column `[3, 6]`.

So, the elements in axis 1 are:
```css
[1, 4]  # First column
[2, 5]  # Second column
[3, 6]  # Third column
```
#### Conclusion:

-   **Axis 0 (rows)**: `[1, 2, 3]`, `[4, 5, 6]`
-   **Axis 1 (columns)**: `[1, 4]`, `[2, 5]`, `[3, 6]`

Each axis refers to a different way of slicing through the tensor: axis 0 slices through rows, and axis 1 slices through columns.

### Explain split `compute` code with example

**Example:** Let `A = Tensor([[1, 2, 3], [4, 5, 6]])`.
#### Case 1: `axis = 0`

In this case, we are splitting along the rows (axis 0), so we expect the output to be two separate rows.

-   **Initial state:**
```python
A.shape = (2, 3)  # 2 rows, 3 columns
axis = 0
```
- **Step-by-step Explanation:**
```python
axis_size = A.shape[self.axis]
# axis_size = 2, since A.shape[0] = 2
```
-   We are splitting along axis 0, which has 2 elements (rows).
```python
split_tensors = []
```
-   We initialize an empty list `split_tensors` to store the split tensors.
```python
output_shape = list(A.shape)
output_shape.pop(self.axis)
# output_shape = [3], since we removed the axis 0 dimension
```
- `output_shape` becomes `[3]` because we are removing axis 0 (which had size 2), leaving us with 3 columns.
```python
for i in range(axis_size):  # Loop over the two rows
```
-   We loop through the two elements along axis 0.

**First Iteration (`i=0`):**
```python
slices = [slice(None)] * len(A.shape)
slices[self.axis] = i
# slices = [0, slice(None)], since axis = 0 and i = 0
```
- `slices` becomes `[0, slice(None)]`, meaning we take the 0th row and all columns.
```python
split_tensors.append(A[tuple(slices)].compact().reshape(output_shape))
# A[tuple(slices)] gives Tensor([1, 2, 3])
# reshape(output_shape) keeps it as [1, 2, 3]
```
-   This extracts the first row `[1, 2, 3]` from the tensor `A` and appends it to `split_tensors`.

**Second Iteration (`i=1`):**
```python
slices = [slice(None)] * len(A.shape)
slices[self.axis] = i
# slices = [1, slice(None)], since axis = 0 and i = 1
```
- `slices` becomes `[1, slice(None)]`, meaning we take the 1st row and all columns.
```python
split_tensors.append(A[tuple(slices)].compact().reshape(output_shape))
# A[tuple(slices)] gives Tensor([4, 5, 6])
# reshape(output_shape) keeps it as [4, 5, 6]
```
- This extracts the second row `[4, 5, 6]` and appends it to `split_tensors`.
```python
return tuple(split_tensors)
```
- The final result is
```python
(Tensor([1, 2, 3]), Tensor([4, 5, 6]))
```
#### Case 2: `axis = 1`

In this case, we are splitting along the columns (axis 1), so we expect the output to be three separate columns.

-   **Initial state:**
```python
A.shape = (2, 3)  # 2 rows, 3 columns
axis = 1
```
- **Step-by-step Explanation:**
```python
axis_size = A.shape[self.axis]
# axis_size = 3, since A.shape[1] = 3
```
-   We are splitting along axis 1, which has 3 elements (columns).
```python
split_tensors = []
```
-   We initialize an empty list `split_tensors` to store the split tensors.
```python
output_shape = list(A.shape)
output_shape.pop(self.axis)
# output_shape = [2], since we removed the axis 1 dimension
```
- `output_shape` becomes `[2]` because we are removing axis 1 (which had size 3), leaving us with 2 rows.
```python
for i in range(axis_size):  # Loop over the three columns
```
-   We loop through the three elements along axis 1.

**First Iteration (`i=0`):**
```python
slices = [slice(None)] * len(A.shape)
slices[self.axis] = i
# slices = [slice(None), 0], since axis = 1 and i = 0
```
- `slices` becomes `[slice(None), 0]`, meaning we take all rows and the 0th column.

```python
split_tensors.append(A[tuple(slices)].compact().reshape(output_shape))
# A[tuple(slices)] gives Tensor([1, 4])
# reshape(output_shape) keeps it as [1, 4]
```
-   This extracts the first column `[1, 4]` and appends it to `split_tensors`.

**Second Iteration (`i=1`):**
```python
slices = [slice(None)] * len(A.shape)
slices[self.axis] = i
# slices = [slice(None), 1], since axis = 1 and i = 1
```
- `slices` becomes `[slice(None), 1]`, meaning we take all rows and the 1st column.
```python
split_tensors.append(A[tuple(slices)].compact().reshape(output_shape))
# A[tuple(slices)] gives Tensor([2, 5])
# reshape(output_shape) keeps it as [2, 5]
```
-   This extracts the second column `[2, 5]` and appends it to `split_tensors`.

**Third Iteration (`i=2`):**
```python
slices = [slice(None)] * len(A.shape)
slices[self.axis] = i
# slices = [slice(None), 2], since axis = 1 and i = 2
```
- `slices` becomes `[slice(None), 2]`, meaning we take all rows and the 2nd column.
```python
split_tensors.append(A[tuple(slices)].compact().reshape(output_shape))
# A[tuple(slices)] gives Tensor([3, 6])
# reshape(output_shape) keeps it as [3, 6]
```
- This extracts the third column `[3, 6]` and appends it to `split_tensors`.
```python
return tuple(split_tensors)
```
- The final result is:
```python
(Tensor([1, 4]), Tensor([2, 5]), Tensor([3, 6]))
```
#### Summary:

-   **Splitting along axis 0 (rows):**
```css
(Tensor([1, 2, 3]), Tensor([4, 5, 6]))
```
- **Splitting along axis 1 (columns):**
```css
(Tensor([1, 4]), Tensor([2, 5]), Tensor([3, 6]))
```
