"""Operator implementations."""

from numbers import Number
from re import I
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
        a = node.inputs[0]
        return a ** (self.scalar - 1) * self.scalar * out_grad
        ### END YOUR SOLUTION


def power_scalar(a, scalar):
    return PowerScalar(scalar)(a)


class EWiseDiv(TensorOp):
    """Op to element-wise divide two nodes."""

    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        return array_api.divide(a, b)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        lhs, rhs = node.inputs
        return (out_grad / rhs, - lhs * out_grad / rhs ** 2)
        ### END YOUR SOLUTION


def divide(a, b):
    return EWiseDiv()(a, b)


class DivScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.divide(a, self.scalar, dtype=a.dtype)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad / self.scalar
        ### END YOUR SOLUTION


def divide_scalar(a, scalar):
    return DivScalar(scalar)(a)


class Transpose(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        if self.axes:
            return array_api.swapaxes(a, self.axes[0], self.axes[1])
        else:
            return array_api.swapaxes(a, a.ndim - 2, a.ndim - 1)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad.transpose(self.axes)
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
        ori_shape = node.inputs[0].shape
        return out_grad.reshape(ori_shape)
        ### END YOUR SOLUTION


def reshape(a, shape):
    return Reshape(shape)(a)


class BroadcastTo(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        return array_api.broadcast_to(a, self.shape)

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        ori_shape = node.inputs[0].shape
        shrink_dims = [i for i in range(len(self.shape))]
        for i, (ori, cur) in enumerate(zip(reversed(ori_shape), reversed(self.shape))):
            if ori == cur:
                shrink_dims[len(self.shape) - i - 1] = -1
        shrink_dims = tuple(filter(lambda x: x >= 0, shrink_dims))
        return out_grad.sum(shrink_dims).reshape(ori_shape)
        ### END YOUR SOLUTION


def broadcast_to(a, shape):
    return BroadcastTo(shape)(a)


class Summation(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.sum(a, self.axes)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        new_shape = list(node.inputs[0].shape)
        axes = range(len(new_shape)) if self.axes is None else self.axes
        for axis in axes:
            new_shape[axis] = 1
        return out_grad.reshape(new_shape).broadcast_to(node.inputs[0].shape)
        ### END YOUR SOLUTION


def summation(a, axes=None):
    return Summation(axes)(a)


class MatMul(TensorOp):
    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        return array_api.matmul(a, b)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        lhs, rhs = node.inputs
        lgrad, rgrad = matmul(out_grad, rhs.transpose()), matmul(lhs.transpose(), out_grad)

        # X(B, M, K) matmul W(K, N)
        # out(B, M, N)
        # dX = out matmul W_transpose
        # dW = X_transpose matmul out -> B, K, N，然后在B维度上reduce-> K, N

        if len(lhs.shape) < len(lgrad.shape):
            lgrad = lgrad.sum(tuple([i for i in range(len(lgrad.shape) - len(lhs.shape))]))
        if len(rhs.shape) < len(rgrad.shape):
            rgrad = rgrad.sum(tuple([i for i in range(len(rgrad.shape) - len(rhs.shape))]))
        return (lgrad, rgrad)
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
        return - out_grad
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
        a = node.inputs[0].realize_cached_data()
        return out_grad * Tensor(a > 0)
        ### END YOUR SOLUTION


def relu(a):
    return ReLU()(a)
# ============================================

class Dilate(TensorOp):
    def __init__(self, axes: tuple, dilation: int):
        self.axes = axes
        self.dilation = dilation

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        dilated_shape = list(a.shape)
        if type(self.axes) == int:
          self.axes = (self.axes,)
        for axis in self.axes:
          # ignore the invalid axis
  
          if axis < len(dilated_shape):
            dilated_shape[axis] = a.shape[axis] + (a.shape[axis]) * self.dilation
        dilated = array_api.full(dilated_shape, 0, dtype=a.dtype)
        insert_slices = tuple(slice(None, None, self.dilation + 1) if axis in self.axes else slice(None) for axis in range(len(a.shape)))
        dilated[insert_slices] = a
        return dilated
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return UnDilate(self.axes, self.dilation)(out_grad),
        ### END YOUR SOLUTION


def dilate(a, axes, dilation):
    return Dilate(axes, dilation)(a)


class UnDilate(TensorOp):
    def __init__(self, axes: tuple, dilation: int):
        self.axes = axes
        self.dilation = dilation

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        if type(self.axes) == int:
          self.axes = (self.axes,)
        undilated_shape = list(a.shape)
        for axis in list(self.axes):
          # ignore the invalid axis
          if axis < len(a.shape):
            undilated_shape[axis] = (a.shape[axis]) // (self.dilation + 1) + 1
        undilated_slices = tuple(slice(None, None, self.dilation + 1) if axis in self.axes else slice(None) for axis in range(len(a.shape)))
        return a[undilated_slices]
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


def undilate(a, axes, dilation):
    return UnDilate(axes, dilation)(a)



# ============================================

pi = 3.141592653589793
def DFT_slow(x):
    """Compute the discrete Fourier Transform of the 1D array x"""
    # x = np.asarray(x, dtype=float)
    if isinstance(x, Tensor):
      x = x.numpy()

    N = x.shape[0]
    n = array_api.array([x for x in range(N)])
    k = n.reshape((N, 1))
    constant = -2j * pi / N
    array_tmp = k * n
    M = array_api.exp(array_tmp * constant)
    # return array_api.multiply(M, x)
    # print("type: ", type(M @ x))
    return M @ x

def FFT_normal(x):
    """A recursive implementation of the 1D Cooley-Tukey FFT"""
    N = x.shape[0]
    if isinstance(x, Tensor):
      x = x.numpy()
    if N % 2 > 0:
        if N > 32:
          print("N:",N)
          raise ValueError("size of x must be a power of 2")
        else:
          return DFT_slow(x)
    elif N <= 32:  # this cutoff should be optimized
        return DFT_slow(x)
    else:
        X_even = FFT_normal(x[::2])
        X_odd = FFT_normal(x[1::2])
        tmp = array_api.array([x for x in range(N)]) 
        factor = array_api.exp(-2j * pi * tmp / N)
        return array_api.concatenate([X_even + factor[:N // 2] * X_odd,
                               X_even + factor[N // 2:] * X_odd])
import math

def log2(N):
  return math.log(N) / math.log(2)

def FFT_vectorized(x):
    """A vectorized, non-recursive version of the Cooley-Tukey FFT"""
    # x = np.asarray(x, dtype=float)
    if isinstance(x, Tensor):
      x = x.numpy()
    N = x.shape[0]

    if log2(N) % 1 > 0:
      raise ValueError("size of x must be a power of 2")

    # N_min here is equivalent to the stopping condition above,
    # and should be a power of 2
    N_min = min(N, 32)

    # Perform an O[N^2] DFT on all length-N_min sub-problems at once
    n = array_api.array([x for x in range(N_min)])
    k = n[:, None]
    M = array_api.exp(-2j * pi * n * k / N_min)
    X = M @ (x.reshape((N_min, -1)))

    # build-up each level of the recursive calculation all at once
    while X.shape[0] < N:
        X_even = X[:, :X.shape[1] // 2]
        X_odd = X[:, X.shape[1] // 2:]
        tmp  = array_api.array([x for x in range(X.shape[0])])
        factor = array_api.exp(-1j * pi * tmp
                        / X.shape[0])[:, None]
        X = array_api.vstack([X_even + factor * X_odd,
                       X_even - factor * X_odd])

    return X.ravel()

def fft_1d_handle(x):
  if isinstance(x, Tensor):
    x = x.numpy()
  N = x.shape[0]
  if log2(N) % 1 > 0:
    # print("size of x is not power of 2")
    # print("try to use normal fft")
    return FFT_normal(x)
  else:
    # print("using vect_version")
    return FFT_vectorized(x)

def my_2d_fft(input_2d):
  # 第一步：对每行应用一维FFT
  step1_result = [fft_1d_handle(row) for row in input_2d]
  return array_api.array(step1_result)

def my_high_fft(input, dim = None):
  """
  input: any dimention
  dim: only handle the positive & < len(input_shape), default = last dim
  """
  input_shape = input.shape
  if dim == None:
    dim = len(input_shape) -1
  assert dim < len(input_shape)
  if len(input_shape) == 1:
    return fft_1d_handle(input)
  if dim != len(input_shape) -1:
    trans_shape = list(range(input.ndim))
    trans_shape[len(input_shape) -1],trans_shape[dim] = dim, len(input_shape) -1
    # print("tuple: ", trans_shape)
    input = input.transpose(trans_shape)
  # flatten ==> (B,S,V) -> (-1, dim)
  transpose_shape = input.shape
  input_flat = input.reshape(-1,input_shape[dim])

  result = my_2d_fft(input_flat).reshape(transpose_shape)
  if dim != len(input_shape) -1:
    result = result.transpose(trans_shape)
  return result

class FFTOp(TensorOp):
  def __init__(self, dim = None):
        self.dim = dim

  def compute(self, a: NDArray):
    return my_high_fft(a, self.dim)

  def gradient(self, out_grad: Tensor, node: Tensor):

    # FFT的反向传播通常是IFFT（逆FFT）
    return IFFTOp(self.dim)(out_grad)



def fft(a, dim = None):
  return FFTOp(dim)(a)


def ifft_1d_handle(input):
  N = len(input)
  x_conjugate = array_api.conj(input)  # 取共轭
  fft_result = fft_1d_handle(x_conjugate)
  fft_conjugate = array_api.conj(fft_result)
  return fft_conjugate / N

def my_2d_ifft(input_2d):
    step1_result = [ifft_1d_handle(row) for row in input_2d]
    return array_api.array(step1_result)


def my_high_ifft(input, dim=None):
    """
    Handle IFFT for input of any dimension.
    dim: The dimension to apply IFFT, defaults to the last dimension.
    """
    input_shape = input.shape
    if dim is None:
      dim = len(input_shape) - 1
    assert dim < len(input_shape)

    if len(input_shape) == 1:
      return ifft_1d_handle(input)

    if dim != len(input_shape) - 1:
      trans_shape = list(range(input.ndim))
      trans_shape[len(input_shape) -1],trans_shape[dim] = dim, len(input_shape) -1
      input = input.transpose(trans_shape)

    # Flatten to 2D for processing
    transpose_shape = input.shape
    input_flat = input.reshape(-1, input_shape[dim])
    result = my_2d_ifft(input_flat).reshape(transpose_shape)

    if dim != len(input_shape) - 1:
        result = result.transpose(trans_shape)

    return result

class IFFTOp(TensorOp):
  def __init__(self, dim = None):
        self.dim = dim

  def compute(self, a: NDArray):
    return my_high_ifft(a, self.dim)

  def gradient(self, out_grad: Tensor, node: Tensor):

    # FFT的反向传播通常是IFFT（逆FFT）
    # 需要根据您的自动微分系统来实现
    raise NotImplementedError("Gradient not implemented for FFT")


def ifft(a, dim = None):
  return IFFTOp(dim)(a)


def rfft_1d_handle(input):
  N = len(input)
  N_half = N // 2 + 1
  fft_result = fft_1d_handle(input)
  return fft_result[:N_half]



def apply_n_rfft(input, axis, len_change):
  input_shape = input.shape
  if axis != len(input.shape) -1:
    trans_axes = list(range(len(input_shape)))
    trans_axes[len(input.shape) -1],trans_axes[axis] = axis, len(input.shape) -1
    input = input.transpose(tuple(trans_axes))
  original_shape = input.shape
  input_flat = input.reshape(-1, input_shape[axis])
  if len_change == True:
    step1_result = [rfft_1d_handle(row) for row in input_flat]
    result = array_api.array(step1_result)
    new_last_dim = original_shape[-1] // 2 + 1
    original_shape = original_shape[:-1] + (new_last_dim,)
  else:
    result = my_2d_fft(input_flat)
  input_back = result.reshape(original_shape)
  if axis != len(input.shape) -1:
    input_back = input_back.transpose(tuple(trans_axes))
  return input_back

def my_high_rfft(input, dim=None):
    """
    Handle IFFT for input of any dimension.
    dim: The dimension to apply IFFT, defaults to the last dimension.
    """
    input_shape = input.shape
    if dim is None:
      dim = tuple(range(len(input_shape)))
  

    if len(input_shape) == 1:
      return rfft_1d_handle(input)

    len_change = True
    for d in reversed(dim):
      input = apply_n_rfft(input, d, len_change)
      len_change = False
    return input

class RFFTOp(TensorOp):
  def __init__(self, dim = None):
        self.dim = dim

  def compute(self, a: NDArray):
    self.original_shape = a.shape
    return my_high_rfft(a, self.dim)

  def gradient(self, out_grad: Tensor, node: Tensor):
    return my_high_irfft(out_grad, self.original_shape, self.dim)

# rfft dim should be (1,2,)
def rfft(a, dim = None):
  return RFFTOp(dim)(a)


def apply_n_irfft(input, axis, len_change, original_shape):
  input_shape = input.shape
  if axis != len(input.shape) -1:
    trans_axes = list(range(len(input_shape)))
    trans_axes[len(input.shape) -1],trans_axes[axis] = axis, len(input.shape) -1
    input = input.transpose(tuple(trans_axes))
  old_shape = input.shape
  input_flat = input.reshape(-1, input_shape[axis])
  if len_change == True:
    assert (input_shape[axis] -1)*2 == original_shape[axis]
    conjugate_part = array_api.conj(input_flat[..., 1:input_shape[axis]])[..., ::-1]
    input_flat = array_api.concatenate([input_flat, conjugate_part[..., 1:]], axis=-1)
    step1_result = [ifft_1d_handle(row) for row in input_flat]
    result = array_api.array(step1_result)
    new_last_dim = (old_shape[-1]-1) *2
    old_shape = old_shape[:-1] + (new_last_dim,)
  else:
    result = my_2d_ifft(input_flat)
  input_back = result.reshape(old_shape)
  if axis != len(input.shape) -1:
    input_back = input_back.transpose(tuple(trans_axes))
  return input_back



def my_high_irfft(input, original_shape, dim=None):
    """
    Handle IFFT for input of any dimension.
    dim: The dimension to apply IFFT, defaults to the last dimension.
    """
    input_shape = input.shape
    if dim is None:
      dim = tuple(range(len(input_shape)))
  

    # if len(input_shape) == 1:
    #   return ifft_1d_handle(input)

    len_change = False
    for d in dim:
      if d == dim[-1]:
        len_change = True
      input = apply_n_irfft(input, d, len_change, original_shape)

    return input




class IRFFTOp(TensorOp):
  def __init__(self, original_shape, dim = None):
        self.dim = dim
        self.original_shape = original_shape

  def compute(self, a: NDArray):
    return my_high_irfft(a, self.original_shape, self.dim)

  def gradient(self, out_grad: Tensor, node: Tensor):

    # FFT的反向传播通常是IFFT（逆FFT）
    # 需要根据您的自动微分系统来实现
    raise NotImplementedError("Gradient not implemented for FFT")

# rfft dim should be (1,2,)
def irfft(a, original_shape, dim = None):
  return IRFFTOp(original_shape, dim)(a)



def complex_matmul(a, b, groups=1):
    a = a.reshape((a.shape[0], groups, -1) + a.shape[2:])
    b = b.reshape((groups, -1) + b.shape[1:])

    a = array_api.moveaxis(a, 2, -1)
    a = array_api.expand_dims(a, -2)
    b = array_api.moveaxis(b, (1, 2), (-1, -2))

    # 复数矩阵乘法
    real = a.real @ b.real + a.imag @ b.imag
    imag = a.imag @ b.real - a.real @ b.imag
    c = real + 1j * imag

    # 重新调整维度顺序并改变形状
    c = array_api.moveaxis(c, -1, 2).squeeze(-1)
    return c.reshape((c.shape[0], -1, *c.shape[3:]))



class FFTconv(TensorOp):
    def __init__(self, stride: Optional[int] = 1, padding: Optional[int] = 0, dilation = 1, groups = 1):
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

    def compute(self, A, kernel):
        ### BEGIN YOUR SOLUTION
        # Apply padding to the input tensor

        # A is the input tensor
        # kernel is the weight tensor (kernel_size, kernel_size, input_channels, output_channels)

        # print("init_kernal:",kernel.shape)
        # print("init_A:",A.shape)
        n = 2
        # dilation
        if self.dilation > 1:
          my_dilation = self.dilation -1
          dilation_kernel = Dilate((2, 3), my_dilation).compute(kernel)
          original_shape = (kernel.shape[-2], kernel.shape[-1])
          trimmed_size = [k + (k - 1) * my_dilation for k in original_shape]
          clip_dilation_kernel = dilation_kernel[..., :trimmed_size[0], :trimmed_size[1]]
        else:
          clip_dilation_kernel = kernel
        
        

        signal_size = A.shape # original signal size without padding to even
        kernal_size = kernel.shape
        if signal_size[-1] %2 != 0:
          signal_final = array_api.pad(A, pad_width=[(0, 0), (0, 0), (0, 0), (0, 1)])
        else:
          signal_final = A
        pad_width_four = signal_final.shape[-1] - clip_dilation_kernel.shape[-1]
        pad_width_thr = signal_final.shape[-2] - clip_dilation_kernel.shape[-2]
        kernal_final = array_api.pad(clip_dilation_kernel, pad_width=[(0, 0), (0, 0), (0, pad_width_thr), (0, pad_width_four)])
        fft_dim = (2,3)

        signal_fft = RFFTOp(fft_dim).compute(signal_final)
        kernel_fft = RFFTOp(fft_dim).compute(kernal_final)
        # print("signal_fft", signal_fft.shape)
        # print("kernel_fft", kernel_fft.shape)
        output_fft = complex_matmul(signal_fft, kernel_fft, groups=self.groups)
        out = IRFFTOp(signal_final.shape, fft_dim).compute(output_fft)

        indices_dim3 = list(range(0,signal_size[2] - clip_dilation_kernel.shape[2] + 1, self.stride))
        indices_dim4 = list(range(0,signal_size[3] - clip_dilation_kernel.shape[3] + 1, self.stride))



        index_array = array_api.ix_(array_api.arange(out.shape[0]), array_api.arange(out.shape[1]), indices_dim3, indices_dim4)
        out = out[index_array]

        return out



    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        # A, B = node.inputs  # A is input, B is weights
        
        # B_grad = conv(trans_A, out_grad_permuted, stride=1, padding=self.padding).transpose((1,2,0,3))
        # return A_grad, B_grad
        pass
        ### END YOUR SOLUTION
def fftconv(a, b, stride=1, padding=1, dilation = 1, groups = 1):
    return FFTconv(stride, padding, dilation, groups)(a, b)