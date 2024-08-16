"""The module.
"""
from typing import List, Callable, Any
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np


class Parameter(Tensor):
    """A special kind of tensor that represents parameters."""


def _unpack_params(value: object) -> List[Tensor]:
    if isinstance(value, Parameter):
        return [value]
    elif isinstance(value, Module):
        return value.parameters()
    elif isinstance(value, dict):
        params = []
        for k, v in value.items():
            params += _unpack_params(v)
        return params
    elif isinstance(value, (list, tuple)):
        params = []
        for v in value:
            params += _unpack_params(v)
        return params
    else:
        return []


def _child_modules(value: object) -> List["Module"]:
    if isinstance(value, Module):
        modules = [value]
        modules.extend(_child_modules(value.__dict__))
        return modules
    if isinstance(value, dict):
        modules = []
        for k, v in value.items():
            modules += _child_modules(v)
        return modules
    elif isinstance(value, (list, tuple)):
        modules = []
        for v in value:
            modules += _child_modules(v)
        return modules
    else:
        return []


class Module:
    def __init__(self):
        self.training = True

    def parameters(self) -> List[Tensor]:
        """Return the list of parameters in the module."""
        return _unpack_params(self.__dict__)

    def _children(self) -> List["Module"]:
        return _child_modules(self.__dict__)

    def eval(self):
        self.training = False
        for m in self._children():
            m.training = False

    def train(self):
        self.training = True
        for m in self._children():
            m.training = True

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Identity(Module):
    def forward(self, x):
        return x


class Linear(Module):
    def __init__(
        self, in_features, out_features, bias=True, device=None, dtype="float32"
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        self.device = device
        self.dtype = dtype
        self.bias = bias
        
        self.weight = Parameter(init.kaiming_uniform(in_features, out_features, device=self.device, dtype=self.dtype, requires_grad=True))

        if self.bias:
          self.bias = Parameter(ops.reshape(init.kaiming_uniform(out_features, 1, device=self.device, dtype=self.dtype, requires_grad=True), (1, out_features)))
        else:
          self.bias = None
        ### END YOUR SOLUTION

    def forward(self, X: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        result = X @ self.weight
        if self.bias is not None:
          # YOU Have to broadcast to shape!!
          # bias = ops.broadcast_to(self.bias,(X.shape[0],self.out_features))
          result += self.bias.broadcast_to(result.shape)
        return result
        ### END YOUR SOLUTION


class Flatten(Module):
    def forward(self, X):
      original_shape = X.shape
      feature_num = 1
      for dim in original_shape[1:]:
        feature_num *= dim
      return ops.reshape(X,(original_shape[0], feature_num))
        


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return ops.relu(x)
        ### END YOUR SOLUTION


class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        self.modules = modules

    def forward(self, x: Tensor) -> Tensor:
        for module in self.modules:
            x = module(x)
        return x


class SoftmaxLoss(Module):
    def forward(self, logits: Tensor, y: Tensor):
        ### BEGIN YOUR SOLUTION
        # m, n = logits.shape
        # expZ = ops.exp(logits)
        # sumZ = ops.summation(expZ, axes=(1,))
        # sumZ = ops.reshape(sumZ,(-1,1))
        # sumZ = ops.broadcast_to(sumZ,(m, n))
        # logit = expZ / sumZ # (batch_size, k)

        # label_y = init.one_hot(n, y)
        # loss = ops.multiply(label_y,ops.log(logit))
        # print("logit",ops.log(logit))
        # print("loss",loss)
        # loss = ops.summation(loss)
        # loss = ops.negate(loss)
        # loss = ops.divide_scalar(loss,np.float32(m))
        # return label_y
        
        m, n = logits.shape
        label_y = init.one_hot(n, y, device = logits.device)
        zy = ops.multiply(label_y,logits)
        zy = ops.summation(zy,axes = (1,))

        logexpsum = ops.logsumexp(logits,axes = (1,))
        loss = logexpsum - zy
        loss = ops.summation(loss) / m

        # print("loss: ", loss)
        return loss
        ### END YOUR SOLUTION


class BatchNorm1d(Module):
    def __init__(self, dim, eps=1e-5, momentum=0.1, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        self.device = device
        self.dtype = dtype


        self.weight = Parameter(init.ones(dim, device = self.device, dtype = self.dtype, requires_grad=True))
        self.bias = Parameter(init.zeros(dim, device = self.device, dtype = self.dtype, requires_grad=True))
        self.training = True

        self.running_var = init.ones(dim, device = self.device, dtype = self.dtype, requires_grad=True)
        self.running_mean = init.zeros(dim, device = self.device, dtype = self.dtype, requires_grad=True)
        
        

    def forward(self, x: Tensor) -> Tensor:
        B, D = x.shape
        if self.training is True:
          # train part
          mean_x = ops.summation(x, axes=(0,))
          mean_x = ops.divide_scalar(mean_x, B)
          self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean_x
          mean_x = ops.reshape(mean_x, (1,D))
          mean_x = ops.broadcast_to(mean_x,(B,D))
          tmp = ops.add(x,ops.negate(mean_x))

          var_x = ops.multiply(tmp, tmp)
          var_x = ops.summation(var_x, axes=(0,))
          var_x = ops.divide_scalar(var_x, B)
          self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var_x
          var_x = ops.reshape(var_x, (1,D))
          var_x = ops.broadcast_to(var_x,(B,D))
          self.running_mean = self.running_mean.detach()
          self.running_var = self.running_var.detach()
        else:
          # eval part
          mean_x = ops.reshape(self.running_mean, (1,D))
          mean_x = ops.broadcast_to(mean_x,(B,D))
          tmp = ops.add(x,ops.negate(mean_x))
          var_x = ops.reshape(self.running_var, (1,D))
          var_x = ops.broadcast_to(var_x,(B,D))

        bott = ops.add_scalar(var_x,self.eps)
        bott =  ops.power_scalar(bott,0.5)
        norm_X = ops.divide(tmp,bott)

        tmp_weight = ops.reshape(self.weight, (1,D))
        tmp_weight = ops.broadcast_to(tmp_weight, norm_X.shape)
        tmp_bias = ops.reshape(self.bias, (1,D))
        tmp_bias = ops.broadcast_to(tmp_bias, norm_X.shape)

        out = ops.add(ops.multiply(tmp_weight, norm_X), tmp_bias)
        return out
        ### END YOUR SOLUTION


class LayerNorm1d(Module):
    def __init__(self, dim, eps=1e-5, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.dtype = dtype
        self.device = device

        self.weight = Parameter(init.ones(dim, device = self.device, dtype = self.dtype, requires_grad=True))
        self.bias = Parameter(init.zeros(dim, device = self.device, dtype = self.dtype, requires_grad=True))
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        B, D = x.shape
        mean_x = ops.summation(x, axes=(1,))
        mean_x = ops.divide_scalar(mean_x, D)
        mean_x = ops.reshape(mean_x, (B,1))
        mean_x = ops.broadcast_to(mean_x,(B,D))
        tmp = ops.add(x,ops.negate(mean_x))
        
        var_x = ops.multiply(tmp, tmp)
        var_x = ops.summation(var_x, axes=(1,))
        var_x = ops.divide_scalar(var_x, D)
        var_x = ops.reshape(var_x, (B,1))
        var_x = ops.broadcast_to(var_x,(B,D))
        bott = ops.add_scalar(var_x,self.eps)
        bott =  ops.power_scalar(bott,0.5)
        norm_X = ops.divide(tmp,bott)


        tmp_weight = ops.reshape(self.weight, (1,D))
        tmp_weight = ops.broadcast_to(tmp_weight, norm_X.shape)
        tmp_bias = ops.reshape(self.bias, (1,D))
        tmp_bias = ops.broadcast_to(tmp_bias, norm_X.shape)

        out = ops.add(ops.multiply(tmp_weight, norm_X), tmp_bias)
        return out
        ### END YOUR SOLUTION


class Dropout(Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p
        self.training = True

    def forward(self, x: Tensor) -> Tensor:
        if self.training is True:
          mask = init.randb(*x.shape, p = 1-self.p, dtype = x.dtype)
          return ops.mul_scalar(ops.multiply(x, mask),(1.0 / (1.0 - self.p)))
        else:
          return x


class Residual(Module):
    def __init__(self, fn: Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return self.fn(x)+x
        ### END YOUR SOLUTION


class Fftconv2d(Module):
    """
    Multi-channel 2D convolutional layer
    IMPORTANT: Accepts inputs in NCHW format, outputs also in NCHW format
    Only supports padding=same
    No grouped convolution or dilation
    Only supports square kernels
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, groups = 1, dilation = 1, padding = 1, bias=True, device=None, dtype="float32"):
        super().__init__()
        if isinstance(kernel_size, tuple):
            kernel_size = kernel_size[0]
        if isinstance(stride, tuple):
            stride = stride[0]
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.groups = groups
        self.dilation = dilation

        self.padding = padding

        # Initialize the weights using Kaiming uniform initialization
        self.weight = Parameter(init.kaiming_uniform((in_channels // groups) * kernel_size ** 2,
                                              out_channels,
                                              shape=(out_channels, in_channels // groups, kernel_size, kernel_size),
                                              device=device,
                                              dtype=dtype,
                                              requires_grad=True))

        # Initialize the bias term
        if bias:
            bias_bound = 1.0 / (in_channels * kernel_size ** 2) ** 0.5
            shape = (out_channels,)
            self.bias = Parameter(init.rand(out_channels, low = -bias_bound, high = bias_bound,
                                               device=device,
                                               dtype=dtype,
                                               requires_grad=True))
        else:
            self.bias = None
    
    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        conv_out = ops.fftconv(x, self.weight, stride=self.stride, padding=self.padding, dilation = self.dilation, groups = self.groups)
        if self.bias is not None:
          # Broadcasting bias
          tmp = self.bias.reshape((1, -1, 1, 1)).broadcast_to(conv_out.shape)
          conv_out = conv_out + tmp
        
        return conv_out
        ### END YOUR SOLUTION
