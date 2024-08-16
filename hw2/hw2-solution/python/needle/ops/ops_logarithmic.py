from typing import Optional
from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp

from .ops_mathematic import *

import numpy as array_api

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

    def compute(self, Z):
        ### BEGIN YOUR SOLUTION
        maxZ = array_api.amax(Z, axis=self.axes, keepdims=True) # Compute max value along the specified axes
        self.exp_term = exp(Tensor(Z - maxZ, dtype=(Z-maxZ).dtype ))
        out = summation(self.exp_term, self.axes)
        out = log(out) + reshape(Tensor(maxZ, dtype =maxZ.dtype ),out.shape)
        return out.numpy()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        reduce_axes = []
        if self.axes is None:
          pass
        else:
          for i in range(len(self.exp_term.shape)):
            if i in self.axes:
              reduce_axes.append(1)
            else:
              reduce_axes.append(self.exp_term.shape[i])
        sum_exp = reshape(summation(self.exp_term, self.axes), tuple(reduce_axes))

        grad_Z = divide(self.exp_term, sum_exp)
        rs_out_grad = reshape(out_grad,tuple(reduce_axes))
        return multiply(rs_out_grad, grad_Z)


def logsumexp(a, axes=None):
    return LogSumExp(axes=axes)(a)

