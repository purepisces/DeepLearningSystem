### Convolution forward

Implement the forward pass of 2D multi-channel convolution in `ops.py`. You should probably refer to [this notebook](https://github.com/dlsyscourse/public_notebooks/blob/main/convolution_implementation.ipynb) from lecture, which implements 2D multi-channel convolution using im2col in numpy.

**Note:** Your convolution op should accept tensors in the NHWC format, as in the example above, and weights in the format (kernel_size, kernel_size, input_channels, output_channels).

However, you will need to add two additional features. Your convolution function should accept arguments for `padding` (default 0) and `stride` (default 1). For `padding`, you should simply apply your padding function to the spatial dimensions (i.e., axes 1 and 2).

Implementing strided convolution should consist of a relatively small set of changes to your plain convolution implementation.

We recommend implementing convolution without stride first, ensuring you pass some of the tests below, and then adding in stride.


### Convolution backward

Finding the gradients of 2D multi-channel convolution can be technically quite challenging (especially "rigorously"). We will try to provide some useful hints here. Basically, we encourage you to make use of the surprising fact that _whatever makes the dimensions work out is typically right_.


Ultimately, the backward pass of convolution can be done in terms of the convolution operator itself, with some clever manipulations using `flip`, `dilate`, and multiple applications of `transpose` to both the arguments and the results.


In the last section, we essentially implemented convolution as a matrix product: ignoring the various restride and reshape operations, we basically have something like `X @ W`, where `X` is the input and `W` is the weight. We also have `out_grad`, which is the same shape as `X @ W`. Now, you have already implemented the backward pass of matrix multiplication in a previous assignment, and we can use this knowledge to get some insight into the backward pass of convolution. In particular, referencing your matmul backward implementation, you may notice (heuristically speaking here):

  

`X.grad = out_grad @ W.transpose` \

`W.grad = X.transpose @ out_grad`

  

Surprisingly enough, things work out if we just assume that these are also convolutions (and now assuming that `out_grad`, `W`, and `X` are tensors amenable to 2D multi-channel convolution instead of matrices):


`X.grad = ≈conv(≈out_grad, ≈W)` \

`W.grad = ≈conv(≈X, ≈out_grad)`


In which the "≈" indicates that you need to apply some additional operators to these terms in order to get the dimensions to work out, such as permuting/transposing axes, dilating, changing the `padding=` argument to the convolution function, or permuting/transposing axes of the resulting convolution.

  

As we saw on the [last few slides here](https://dlsyscourse.org/slides/conv_nets.pdf) in class, the transpose of a convolution can be found by simply flipping the kernel. Since we're working in 2D instead of 1D, this means flipping the kernel both vertically and horizontally (thus why we implemented `flip`).

Summarizing some hints for both `X.grad` and `W.grad`:

`X.grad`

- The convolution of `out_grad` and `W`, with some operations applied to those

- `W` should be flipped over both the kernel dimensions

- If the convolution is strided, increase the size of `out_grad` with a corresponding dilation

- Do an example to analyze dimensions: note the shape you want for `X.grad`, and think about how you must permute/transpose the arguments and add padding to the convolution to achieve this shape

- This padding depends on both the kernel size and the `padding` argument to the convolution

  

`W.grad`

- The convolution of `X` and `out_grad`, with some operations applied to those

- The gradients of `W` must be accumulated over the batches; how can you make the conv operator itself do this accumulation?

- Consider turning batches into channels via transpose/permute

- Analyze dimensions: how can you modify `X` and `out_grad` so that the shape of their convolution matches the shape of `W`? You may need to transpose/permute the result.

- Remember to account for the `padding` argument passed to convolution

General tips

- Deal with strided convolutions last (you should be able to just drop in `dilate` when you've passed most of the tests)

- Start with the case where `padding=0`, then consider changing `padding` arguments

- You can "permute" axes with multiple calls to `transpose`


It might also be useful to skip ahead to nn.Conv, pass the forward tests, and then use both the tests below and the nn.Conv backward tests to debug your implementation.

**Code Implementation**
```python
class Conv(TensorOp):
    def __init__(self, stride: Optional[int] = 1, padding: Optional[int] = 0):
        self.stride = stride
        self.padding = padding

    def compute(self, A, B):
        stride = 1 if self.stride is None else self.stride
        if self.padding > 0:
            A = A.pad(((0, 0), (self.padding, self.padding), 
                  (self.padding, self.padding), (0, 0)))
        N, H, W, C_in = A.shape
        K, _, _, C_out = B.shape
        N_stride, H_stride, W_stride, C_in_stride = A.strides
        H_out = ((H - K) // stride) + 1
        W_out = ((W - K) // stride ) + 1
        strided_shape = (N, H_out, W_out, K, K, C_in)
        strided_strides = (N_stride, H_stride * stride, W_stride * stride, H_stride, W_stride, C_in_stride)
        A_strided = A.as_strided(strided_shape, strided_strides).compact()
        A_strided_compact = A_strided.compact() if not A_strided.is_compact() else A_strided
        reshaped_dim = K * K * C_in
        A_reshaped = A_strided_compact.reshape((N * H_out * W_out, reshaped_dim))
        B_compact = B.compact() if not B.is_compact() else B
        B_reshaped = B_compact.reshape((reshaped_dim, C_out))
        output = (A_reshaped @ B_reshaped).reshape((N, H_out, W_out, C_out))
        return output

    def gradient(self, out_grad, node):
        x, weight = node.inputs
        N, H, W, C_in = x.shape
        K, _, _, C_out = weight.shape
        p = self.padding
        s = self.stride

        # Flip and transpose the weight for gradient computation
        weight = flip(weight, axes=(0, 1)).transpose((2, 3))  # K, K, C_out, C_in

        # Dilate out_grad if the convolution is strided
        if s > 1:
            out_grad = dilate(out_grad, axes=(1, 2), dilation=s - 1)
        # Compute the gradient with respect to the input
        grad_x = conv(out_grad, weight, padding=K - 1 - p)
        # Transpose x and out_grad for weight gradient computation
        x_transposed = x.transpose(axes=(0, 3))
        out_grad_transposed = out_grad.transpose(axes=(0, 1)).transpose(axes=(1, 2))
        # Compute the gradient with respect to the weight
        grad_weight = conv(x_transposed, out_grad_transposed, padding=p)
        grad_weight = grad_weight.transpose(axes=(0, 1)).transpose(axes=(1, 2))
        return grad_x, grad_weight

def conv(a, b, stride=1, padding=1):
    return Conv(stride, padding)(a, b)
```


### nn.Conv

#### Fixing init._calculate_fans for convolution

Previously, we have implemented Kaiming uniform/normal initializations, where we essentially assigned `fan_in = input_size` and `fan_out = output_size`.

For convolution, this becomes somewhat more detailed, in that you should multiply both of these by the "receptive field size", which is in this case just the product of the kernel sizes -- which in our case are always going to be the same, i.e., $k\times k$ kernels.

  

**You will need to edit your `kaiming_uniform` in `python/needle/init/init_initializers.py`, etc. init functions to support multidimensional arrays.** In particular, it should support a new `shape` argument which is then passed to, e.g., the underlying `rand` function. Specifically, if the argument `shape` is not None, then ignore `fan_in` and `fan_out` but use the value of `shape` for initializations.
  

You can test this below; though it is not _directly_ graded, it must match ours to pass the nn.Conv mugrade tests.

**Code Implementation**
```python
def kaiming_uniform(fan_in, fan_out, shape=None, nonlinearity="relu", **kwargs):
    assert nonlinearity == "relu", "Only relu supported currently"
    ### BEGIN YOUR SOLUTION
    # Use the recommended gain value for ReLU: gain = sqrt(2)
    gain = math.sqrt(2.0)
    
    # Calculate the bound for the uniform distribution
    bound = gain * math.sqrt(3.0 / fan_in)
    # Generate and return a tensor with values uniformly distributed between -bound and bound
    if shape is None:
        return rand(fan_in, fan_out, low=-bound, high=bound, **kwargs)
    else:
        return rand(*shape, low=-bound, high=bound, **kwargs)
    ### END YOUR SOLUTION
```

#### Implementing nn.Conv

Essentially, nn.Conv is just a wrapper of the convolution operator we previously implemented

which adds a bias term, initializes the weight and bias, and ensures that the padding is set so that the input and output dimensions are the same (in the `stride=1` case, anyways).

Importantly, nn.Conv should support NCHW format instead of NHWC format. In particular, we think this makes more sense given our current BatchNorm implementation. You can implement this by applying `transpose` twice to both the input and output.

- Ensure nn.Conv works for (N, C, H, W) tensors even though we implemented the conv op for (N, H, W, C) tensors

- Initialize the (k, k, i, o) weight tensor using Kaiming uniform initialization with default settings

- Initialize the (o,) bias tensor using uniform initialization on the interval $\pm$`1.0/(in_channels * kernel_size**2)**0.5`

- Calculate the appropriate padding to ensure input and output dimensions are the same

- Calculate the convolution, then add the properly-broadcasted bias term if present

You can now test your nn.Conv against PyTorch's nn.Conv2d with the two PyTest calls below.

**Code Implementation**
In nn_conv.py
```python
class Conv(Module):
    """
    Multi-channel 2D convolutional layer
    IMPORTANT: Accepts inputs in NCHW format, outputs also in NCHW format
    Only supports padding=same
    No grouped convolution or dilation
    Only supports square kernels
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, bias=True, device=None, dtype="float32"):
        super().__init__()
        if isinstance(kernel_size, tuple):
            kernel_size = kernel_size[0]
        if isinstance(stride, tuple):
            stride = stride[0]
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride

        ### BEGIN YOUR SOLUTION
        self.padding = (kernel_size - 1) // 2
        fan_in = kernel_size * kernel_size * in_channels
        fan_out =kernel_size * kernel_size * out_channels
        weight_shape = (kernel_size, kernel_size, in_channels, out_channels)
        self.weight = Parameter(init.kaiming_uniform(fan_in, fan_out,shape=weight_shape,device=device,dtype=dtype))
        if bias:
            stdv = 1.0 / ((in_channels * kernel_size ** 2) ** 0.5)
            self.bias = Parameter(init.rand(out_channels, low=-stdv, high=stdv,device=device,dtype=dtype))
        else:
            self.bias = None
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        conv_output_nhwc = ops.conv(x.transpose((1, 2)).transpose((2, 3)), self.weight, stride=self.stride, padding=self.padding)
        if self.bias is not None:
            bias_nhwc = self.bias.reshape((1, 1, 1, self.out_channels))
            conv_output_nhwc += ops.broadcast_to(bias_nhwc, conv_output_nhwc.shape)
        return conv_output_nhwc.transpose((2, 3)).transpose((1, 2))
        ### END YOUR SOLUTION
```
Also modify summation to handle case when self.axes == ()
```python
class Summation(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes
        # Ensure axes is always a tuple if provided as an int
        if isinstance(axes, int):
            self.axes = (axes,)

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        if self.axes == ():
            print(f"Axes is empty, returning input tensor with shape {a.shape}")
            return a
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
```
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
        input_shape = node.inputs[0].shape
        output_shape = out_grad.shape
        # Step 1: Summing over the extra dimensions (broadcasted dimensions)
        grad = summation(out_grad, tuple(range(len(output_shape) - len(input_shape))))

        # Step 2: Summing over axes where input_shape has size 1 and output_shape has a different size
        for i in range(len(input_shape)):
            if input_shape[-1 - i] == 1 and self.shape[-1 - i] != 1:
                grad = summation(grad, axes=(len(input_shape) - 1 - i,))
        # Step 3: Reshape the final result back to input_shape
        return reshape(grad, input_shape)
        ### END YOUR SOLUTION

def broadcast_to(a, shape):
    return BroadcastTo(shape)(a)
```




### Implementing "ResNet9"

You will now use your convolutional layer to implement a model similar to _ResNet9_, which is known to be a reasonable model for getting good accuracy on CIFAR-10 quickly (see [here](https://github.com/davidcpage/cifar10-fast)). Our main change is that we used striding instead of pooling and divided all of the channels by 4 for the sake of performance (as our framework is not as well-optimized as industry-grade frameworks).

  

In the figure below, before the linear layer, you should "flatten" the tensor. You can use the module `Flatten` in `nn_basic.py`, or you can simply use `.reshape` in the `forward()` method of your ResNet9.

  

Make sure that you pass the device to all modules in your model; otherwise, you will get errors about mismatched devices when trying to run with CUDA.

  

<center><img  src="https://github.com/dlsyscourse/hw4/blob/main/ResNet9.png?raw=true"  alt="ResNet9"  style="width: 400px;"  /></center>

We have tried to make it easier to pass the tests here than for previous assignments where you have implemented models. In particular, we are just going to make sure it has the right number of parameters and similar accuracy and loss after 1 or 2 batches of CIFAR-10.

```python
class Reshape(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.reshape(a.compact(), self.shape)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a = node.inputs[0]
        return reshape(out_grad, a.shape)
        ### END YOUR SOLUTION


def reshape(a, shape):
    return Reshape(shape)(a)
```
modified reshape method from (other elements, -1) to manual calculation
```python
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
        batch_size, features = x.shape
        if self.training:
            # Compute mean and variance across the batch
            # The shape of x is (batch_size, features)
            # The shape of batch_mean is (features, )
            batch_mean = ops.divide_scalar(ops.summation(x, axes=(0,)),batch_size)
            # The batch_mean has the shape (features,). It is first reshaped to (1, features)
            # and then broadcasted to (batch_size, features).
            broadcast_batch_mean = ops.broadcast_to(ops.reshape(batch_mean, (1, features)), x.shape)
            
            # The shape of batch_var is (features, )
            batch_var =ops.divide_scalar(ops.summation(ops.power_scalar((x - broadcast_batch_mean),2), axes=(0,)), batch_size)
            # The batch_var has the shape (features,). It is first reshaped to (1, features)
            # and then broadcasted to (batch_size, features).
            broadcast_batch_var = ops.broadcast_to(ops.reshape(batch_var, (1, features)), x.shape)
            
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
            broadcast_running_mean = ops.broadcast_to(ops.reshape(self.running_mean, (1, features)), x.shape)
            broadcast_running_var = ops.broadcast_to(ops.reshape(self.running_var, (1, features)), x.shape)
            
            # The shape of x_hat = (batch_size, features)
            # self.eps is a scalar value, when added to broadcast_running_var, 
            # it is automatically broadcasted to match the shape of broadcast_running_var, 
            # which is (batch_size, features).
            x_hat = (x - broadcast_running_mean) / ops.power_scalar(broadcast_running_var + self.eps, 0.5)
        
        # Both self.weight and self.bias have the shape (features,). 
        # They are first reshaped to (1, features) and then broadcasted to (batch_size, features).
        broadcast_weight = ops.broadcast_to(ops.reshape(self.weight, (1, features)), x.shape)
        broadcast_bias = ops.broadcast_to(ops.reshape(self.bias, (1, features)), x.shape)
        
        # Apply learnable scale (weight) and shift (bias)
        # Element-wise multiplication of broadcast_weight and x_hat (batch_size, features)
        return broadcast_weight * x_hat + broadcast_bias
        ### END YOUR SOLUTION
```

```python
class Flatten(Module):
    def forward(self, X):
        ### BEGIN YOUR SOLUTION
        batch_size = X.shape[0]
        flattened_size = 1
        for dim in X.shape[1:]:
            flattened_size *= dim
        flattened_shape = (batch_size, flattened_size)
        return X.reshape(flattened_shape)
        ### END YOUR SOLUTION
```
```python
class ConvBN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, bias=True, device=None, dtype="float32"):
        super().__init__()
        self.conv = nn.Conv(in_channels = in_channels, out_channels = out_channels, kernel_size = kernel_size, stride = stride, bias = bias, device = device, dtype = dtype)
        self.batch_norm = ndl.nn.BatchNorm2d(dim=out_channels, device=device, dtype=dtype)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        ### BEGIN YOUR SOLUTION
        return self.relu(self.batch_norm(self.conv(x)))
        ### END YOUR SOLUTION
        
class ResNet9(ndl.nn.Module):
    def __init__(self, device=None, dtype="float32"):
        super().__init__()
        bias = True
        ### BEGIN YOUR SOLUTION ###
        self.conv1 = ConvBN(3, 16, 7, 4, bias=bias, device=device, dtype=dtype)
        self.conv2 = ConvBN(16, 32, 3, 2, bias=bias, device=device, dtype=dtype)
        self.res = ndl.nn.Residual(
            ndl.nn.Sequential(
                ConvBN(32, 32, 3, 1, bias=bias, device=device, dtype=dtype),
                ConvBN(32, 32, 3, 1, bias=bias, device=device, dtype=dtype)
            )
        )
        self.conv3 = ConvBN(32, 64, 3, 2, bias=bias, device=device, dtype=dtype)
        self.conv4 = ConvBN(64, 128, 3, 2, bias=bias, device=device, dtype=dtype)
        self.res2 = ndl.nn.Residual(
            ndl.nn.Sequential(
                ConvBN(128, 128, 3, 1, bias=bias, device=device, dtype=dtype),
                ConvBN(128, 128, 3, 1, bias=bias, device=device, dtype=dtype)
            )
        )
        self.flatten = ndl.nn.Flatten()
        self.linear = ndl.nn.Linear(128, 128, bias=bias, device=device, dtype=dtype)
        self.relu = ndl.nn.ReLU()
        self.linear2 = ndl.nn.Linear(128, 10, bias=bias, device=device, dtype=dtype)
        ### END YOUR SOLUTION

    def forward(self, x):
        ### BEGIN YOUR SOLUTION
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.res(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.res2(x)
        x = self.flatten(x)
        x = self.linear(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x
        ### END YOUR SOLUTION
```
___
Now we can train a ResNet on CIFAR10: (remember to copy the solutions in `python/needle/optim.py` from previous homeworks)

**Code Implementation**

Modified relu and softmax loss to specify the device
```python
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
        relu_grad = Tensor((a > 0), device=node.inputs[0].device)
        
        # Multiply the incoming gradient (Tensor) by the ReLU gradient (also a Tensor)
        return out_grad * relu_grad
        ### END YOUR SOLUTION
```

```python
class SoftmaxLoss(Module):
    def forward(self, logits: Tensor, y: Tensor):
        ### BEGIN YOUR SOLUTION
        batch_size = logits.shape[0]
        num_class = logits.shape[1]
        # Step 1: Compute the log-sum-exp for each row in logits, this will be a 1D tensor of shape (batch_size,)
        log_sum_exp = ops.logsumexp(logits, axes=(1,))
        
        # Step 2: Extract the logits corresponding to the true class labels
        # Convert y (true labels) into a one-hot encoded matrix, the shape of `y_one_hot` is (batch_size, num_class)
        y_one_hot = init.one_hot(num_class, y, device=logits.device)
        # Compute the correct class logits by multiplying logits with y_one_hot and summing over the class dimension. logits shape is (batch_size, num_class), y_one_hot shape is (batch_size, num_class), they multiplied element-wise. The shape of correct_class_logits is (batch_size,)
        correct_class_logits = ops.summation(logits * y_one_hot, axes=(1,))
        
        # Step 3: Compute the loss for each sample.
        # The 'losses' tensor has shape (batch_size,), containing individual loss values for each example in the batch.
        losses = log_sum_exp - correct_class_logits
        
        # Step 4: Return the average loss across the batch
        # The result is a 0-dimensional tensor (a scalar tensor)
        return ops.summation(losses) / batch_size
        ### END YOUR SOLUTION
```
