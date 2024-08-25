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

        ### BEGIN YOUR SOLUTION
        # Initialize weights with Kaiming Uniform initialization (fan_in = in_features)
        self.weight = Parameter(
            init.kaiming_uniform(in_features, out_features, nonlinearity="relu", device=device, dtype=dtype)
        )
        # Initialize bias if applicable (shape = (1, out_features))
        if bias:
            # Initialize the bias, and then reshape it to (1, out_features)
            self.bias = Parameter(
                ops.reshape(
                    init.kaiming_uniform(out_features, 1, nonlinearity="relu", device=device, dtype=dtype),
                    (1, out_features)
                )
            )
        else:
            self.bias = None
        ### END YOUR SOLUTION

    def forward(self, X: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        output = ops.matmul(X, self.weight)
        if self.bias is not None:
            # Create a broadcasted version of bias without modifying the original bias tensor
            broadcasted_bias = ops.broadcast_to(self.bias, output.shape)
            output = ops.add(output, broadcasted_bias)
        return output
        ### END YOUR SOLUTION


class Flatten(Module):
    def forward(self, X):
        ### BEGIN YOUR SOLUTION
        batch_size = X.shape[0]
        flattened_shape = (batch_size, -1)
        return X.reshape(flattened_shape)
        ### END YOUR SOLUTION


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
        ### BEGIN YOUR SOLUTION
        for module in self.modules:
            x = module(x)
        return x
        ### END YOUR SOLUTION


class SoftmaxLoss(Module):
    def forward(self, logits: Tensor, y: Tensor):
        ### BEGIN YOUR SOLUTION
        batch_size = logits.shape[0]
        num_class = logits.shape[1]
        # Step 1: Compute the log-sum-exp for each row in logits, this will be a 1D tensor of shape (batch_size,)
        log_sum_exp = ops.logsumexp(logits, axes=(1,))
        
        # Step 2: Extract the logits corresponding to the true class labels
        # Convert y (true labels) into a one-hot encoded matrix, the shape of `y_one_hot` is (batch_size, num_class)
        y_one_hot = init.one_hot(num_class, y)
        # Compute the correct class logits by multiplying logits with y_one_hot and summing over the class dimension. logits shape is (batch_size, num_class), y_one_hot shape is (batch_size, num_class), they multiplied element-wise. The shape of correct_class_logits is (batch_size,)
        correct_class_logits = ops.summation(logits * y_one_hot, axes=(1,))
        
        # Step 3: Compute the loss for each sample.
        # The 'losses' tensor has shape (batch_size,), containing individual loss values for each example in the batch.
        losses = log_sum_exp - correct_class_logits
        
        # Step 4: Return the average loss across the batch
        # The result is a 0-dimensional tensor (a scalar tensor)
        return ops.summation(losses) / batch_size
        ### END YOUR SOLUTION


class BatchNorm1d(Module):
    def __init__(self, dim, eps=1e-5, momentum=0.1, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        ### BEGIN YOUR SOLUTION
        # Learnable parameters
        # Both self.weight and self.bias have shape (dim,) = (features,)
        self.weight = Parameter(init.ones(self.dim, device=device, dtype=dtype))
        self.bias = Parameter(init.zeros(self.dim, device=device, dtype=dtype))
        # Running mean and variance (not learnable)
        # Both self.running_mean and self.running_var have shape (dim,) = (features,)
        self.running_mean = init.zeros(self.dim, device=device, dtype=dtype)
        self.running_var = init.ones(self.dim, device=device, dtype=dtype)
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        if self.training:
            # Compute mean and variance across the batch
            # The shape of x is (batch_size, features)
            batch_size = x.shape[0]
            # The shape of batch_mean is (features, )
            batch_mean = ops.divide_scalar(ops.summation(x, axes=(0,)),batch_size)
            # The batch_mean has the shape (features,). It is first reshaped to (1, features)
            # and then broadcasted to (batch_size, features).
            broadcast_batch_mean = ops.broadcast_to(ops.reshape(batch_mean, (1, -1)), x.shape)
            
            # The shape of batch_var is (features, )
            batch_var =ops.divide_scalar(ops.summation(ops.power_scalar((x - broadcast_batch_mean),2), axes=(0,)), batch_size)
            # The batch_var has the shape (features,). It is first reshaped to (1, features)
            # and then broadcasted to (batch_size, features).
            broadcast_batch_var = ops.broadcast_to(ops.reshape(batch_var, (1, -1)), x.shape)
            
            # Update running mean and variance
            # Both self.running_mean and self.running_var have shape (dim,) = (features,)
            # We must use the detached `batch_mean` and `batch_var` (i.e., using `.data`), 
            # otherwise the `requires_grad` attribute of `self.running_mean` and `self.running_var` 
            # will become `True`.
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * batch_mean.data
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * batch_var.data

            # Normalize the input
            # The shape of x_hat = (batch_size, features)
            x_hat = (x - broadcast_batch_mean) / ops.power_scalar(broadcast_batch_var + self.eps, 0.5)
        else:
            # Use running mean and variance during evaluation
            # Both self.running_mean and self.running_var have the shape (features,). 
            # They are first reshaped to (1, features) and then broadcasted to (batch_size, features).
            broadcast_running_mean = ops.broadcast_to(ops.reshape(self.running_mean, (1, -1)), x.shape)
            broadcast_running_var = ops.broadcast_to(ops.reshape(self.running_var, (1, -1)), x.shape)
            
            # The shape of x_hat = (batch_size, features)
            # self.eps is a scalar value, when added to broadcast_running_var, 
            # it is automatically broadcasted to match the shape of broadcast_running_var, 
            # which is (batch_size, features).
            x_hat = (x - broadcast_running_mean) / ops.power_scalar(broadcast_running_var + self.eps, 0.5)
        
        # Both self.weight and self.bias have the shape (features,). 
        # They are first reshaped to (1, features) and then broadcasted to (batch_size, features).
        broadcast_weight = ops.broadcast_to(ops.reshape(self.weight, (1, -1)), x.shape)
        broadcast_bias = ops.broadcast_to(ops.reshape(self.bias, (1, -1)), x.shape)
        
        # Apply learnable scale (weight) and shift (bias)
        # Element-wise multiplication of broadcast_weight and x_hat (batch_size, features)
        return broadcast_weight * x_hat + broadcast_bias
        ### END YOUR SOLUTION


class LayerNorm1d(Module):
    def __init__(self, dim, eps=1e-5, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        ### BEGIN YOUR SOLUTION
        # Both self.weight and self.bias have the shape (dim,) = (features,)
        self.weight = Parameter(init.ones(self.dim, device=device, dtype=dtype))
        self.bias = Parameter(init.zeros(self.dim, device=device, dtype=dtype))
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        # The shape of X is (batch_size, features)
        batch_size = x.shape[0]
        features = x.shape[1]
        # The shape of mean is (batch_size,)
        mean = ops.divide_scalar(ops.summation(x, axes=(1,)),features)
        # The shape of mean is (batch_size,)
        mean = ops.reshape(mean, (batch_size, 1))
        # The shape of broadcast_mean is (batch_size, features)
        broadcast_mean = ops.broadcast_to(mean, x.shape)
        # Subtract the mean from the input, the shape of x_minus_mean is (batch_size, features)
        x_minus_mean = x - broadcast_mean
        # Compute the variance of each feature across the batch
        # The shape of var is (batch_size,)
        var = ops.divide_scalar(ops.summation(x_minus_mean ** 2, axes=(1,)), features)
        # The shape of var is (batch_size, 1)
        var = ops.reshape(var, (batch_size, 1))
        # The shape of broadcast_var is (batch_size, features)
        broadcast_var = ops.broadcast_to(var, x.shape)
        # The shapes of broadcast_weight and broadcast_bias are both (batch_size, features).
        # Both self.weight and self.bias have the shape (features,). They are first reshaped to (1, features) 
        # and then broadcasted to (batch_size, features).
        broadcast_weight = ops.broadcast_to(ops.reshape(self.weight, (1, features)), x.shape)
        broadcast_bias = ops.broadcast_to(ops.reshape(self.bias, (1, features)), x.shape)
        # Element-wise multiplication of broadcast_weight and x_minus_mean (batch_size, features)
        # self.eps is a scalar value, when add self.eps to broadcast_var, it is automatically broadcasted to match the shape of broadcast_var, which is (batch_size, features).
        return broadcast_weight * x_minus_mean / ops.power_scalar(broadcast_var + self.eps, 0.5) +  broadcast_bias
        ### END YOUR SOLUTION


class Dropout(Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        if self.training:
           # Create a mask with the same shape as x, where each element is 1 with probability 1-p, and 0 with probability p
           mask = init.randb(*x.shape, p=1 - self.p)
           # Return the input tensor scaled by 1/(1 - p) and then multiplied by the mask
           return x / (1 - self.p) * mask
        else:
           # During evaluation, dropout does nothing; just return the input as is.
           return x
        ### END YOUR SOLUTION


class Residual(Module):
    def __init__(self, fn: Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        # Apply the function/module to the input x, then add the original input x to create the residual connection.
        return self.fn(x) + x
        ### END YOUR SOLUTION
