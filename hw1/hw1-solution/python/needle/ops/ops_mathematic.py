"""Operator implementations."""

from numbers import Number
from typing import Optional, List, Tuple, Union

from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp
import numpy

# NOTE: we will import numpy as the array_api
# as the backend for our computations, this line will change in later homeworks

import numpy as array_api


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
        reduced_power = power_scalar(node.inputs[0].data, self.scalar - 1)
        scalar_times_reduced_power = mul_scalar(reduced_power, self.scalar)
        grad = multiply(out_grad, scalar_times_reduced_power)
        return grad,
        ### END YOUR SOLUTION


def power_scalar(a, scalar):
    return PowerScalar(scalar)(a)


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
        grad_b = out_grad * (a**b) * array_api.log(a.data)
        return grad_a, grad_b

def power(a, b):
    return EWisePow()(a, b)


class EWiseDiv(TensorOp):
    """Op to element-wise divide two nodes."""

    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        return array_api.true_divide(a, b)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a, b = node.inputs
        # Gradient with respect to 'a'
        grad_a = array_api.true_divide(out_grad, b.data)
        # Gradient with respect to 'b'
        grad_b = array_api.true_divide(-a.data * out_grad, array_api.power(b.data, 2))
        return grad_a, grad_b
        ### END YOUR SOLUTION
     
        

def divide(a, b):
    return EWiseDiv()(a, b)


class DivScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.divide(a, self.scalar) 
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        # Gradient with respect to 'a' (tensor)
        grad_a = array_api.true_divide(1, self.scalar)

        # Multiply by the incoming backward gradient
        return out_grad * grad_a,
        ### END YOUR SOLUTION


def divide_scalar(a, scalar):
    return DivScalar(scalar)(a)
    


class Transpose(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        if self.axes is None:
            return array_api.swapaxes(a, -1, -2)
        else:
            return array_api.swapaxes(a, self.axes[0], self.axes[1])
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        if self.axes is None:
            return transpose(out_grad, (-1, -2)),
        else:
            # Reverse the axes for the transpose of the gradient.
            reversed_axes = (self.axes[1], self.axes[0])
            return transpose(out_grad, reversed_axes),

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
        original_shape = node.inputs[0].shape  
        reshaped_grad = reshape(out_grad, original_shape)
        return reshaped_grad,
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
        input_shape = node.inputs[0].shape

        def get_reduce_axes():
            input_length = len(input_shape) - 1
            for idx in range(len(self.shape) - 1, -1, -1):
                if input_length < 0 or input_shape[input_length] != self.shape[idx]:
                    yield idx
                input_length -= 1
        reduce_axes = tuple(get_reduce_axes())
        reduced_grad = summation(out_grad, reduce_axes)
        reduced_grad = reshape(reduced_grad, input_shape)
        return reduced_grad

def broadcast_to(a, shape):
    return BroadcastTo(shape)(a)
    
class Summation(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes
        if isinstance(self.axes, int): 
            self.axes = (self.axes, )

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.sum(a, axis=self.axes)

        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):

        ## BEGIN YOUR SOLUTION
        input_tensor = node.inputs[0]
        input_shape = input_tensor.shape

        if self.axes is None:
            return broadcast_to(out_grad, input_shape),
        else:
            target_shape = list(input_shape)
            for axis in self.axes:
                target_shape[axis] = 1
            return broadcast_to(reshape(out_grad, target_shape), input_shape),
        ## END YOUR SOLUTION

def summation(a, axes=None):
    return Summation(axes)(a)


class MatMul(TensorOp):
    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        return array_api.matmul(a, b)
        ### END YOUR SOLUTION
    def gradient(self, out_grad, node):
        A, B = node.inputs  
        grad_A = matmul(out_grad, transpose(B))
        grad_B = matmul(transpose(A), out_grad)

        shape_diff_A = len(grad_A.shape) - len(A.shape)
        if shape_diff_A > 0:
            axes_to_sum_A = tuple(range(shape_diff_A))
            grad_A = summation(grad_A, axes=axes_to_sum_A)
    
        shape_diff_B = len(grad_B.shape) - len(B.shape)
        if shape_diff_B > 0:
            axes_to_sum_B = tuple(range(shape_diff_B))
            grad_B = summation(grad_B, axes=axes_to_sum_B)
        return grad_A, grad_B
        ### END YOUR SOLUTION

def matmul(a, b):
    return MatMul()(a, b)


class Negate(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.negative(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        ones_tensor = add_scalar(node.data * 0, 1)  
        neg_ones_tensor = negate(ones_tensor)  
        grad = multiply(out_grad, neg_ones_tensor)  
        return grad,
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
        return out_grad / node.inputs[0]
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
        input_tensor = node.inputs[0]
        gradient = out_grad * exp(input_tensor)
        return gradient
        ### END YOUR SOLUTION

def exp(a):
    return Exp()(a)


class ReLU(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()
        return array_api.maximum(a, 0)  
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        # raise NotImplementedError()  
        relu_mask = (node.inputs[0].cached_data > 0).astype(float)
        return out_grad * relu_mask
        ### END YOUR SOLUTION


def relu(a):
    return ReLU()(a)
