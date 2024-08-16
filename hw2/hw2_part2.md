## Question 2

In this question, you will implement additional modules in `python/needle/nn/nn_basic.py`. Specifically, for the following modules described below, initialize any variables of the module in the constructor, and fill out the `forward` method. **Note:** Be sure that you are using the `init` functions that you just implemented to initialize the parameters, and don't forget to pass the `dtype` argument.
___

### Linear
`needle.nn.Linear(in_features, out_features, bias=True, device=None, dtype="float32")`

Applies a linear transformation to the incoming data: $y = xA^T + b$. The input shape is $(N, H_{in})$ where $H_{in}=\text{infeatures}$. The output shape is $(N, H_{out})$ where $H_{out}=\text{outfeatures}$.

Be careful to explicitly broadcast the bias term to the correct shape -- Needle does not support implicit broadcasting.

**Note:** for all layers including this one, you should initialize the weight Tensor before the bias Tensor, and should initialize all Parameters using only functions from `init`.

##### Parameters
- `in_features` - size of each input sample
- `out_features` - size of each output sample
- `bias` - If set to `False`, the layer will not learn an additive bias.

##### Variables
- `weight` - the learnable weights of shape (`in_features`, `out_features`). The values should be initialized with the Kaiming Uniform initialization with `fan_in = in_features`
- `bias` - the learnable bias of shape (1, `out_features`). The values should be initialized with the Kaiming Uniform initialize with `fan_in = out_features`. **Note the difference in fan_in choice, due to their relative sizes**.

**Code implementation**
```python
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
            self.bias = ops.broadcast_to(self.bias, output.shape)
            output = ops.add(output, self.bias)
        return output
        ### END YOUR SOLUTION
```

### My explanation

The `Linear` layer performs a linear transformation on the input data, where the input tensor `X` has a shape of $(N, H_{in})$ (with $H_{in}$ representing `in_features`). This tensor is multiplied by a weight matrix `W`, initialized with the Kaiming Uniform method, and has a shape of $(H_{in}, H_{out})$ (with $H_{out}$ representing `out_features`), producing an output tensor of shape $(N, H_{out})$. If a bias term is included, it is also initialized with the Kaiming Uniform method, starting with a shape of $(H_{out}, 1)$ and then reshaped to $(1, H_{out})$. This reshaped bias is explicitly broadcasted to match the output tensor's shape of $(N, H_{out})$, allowing it to be added element-wise to the output. Both the weight and bias are encapsulated in `Parameter` objects, which the framework recognizes as learnable during the training process. This meticulous handling of shapes, initialization, and broadcasting ensures the layer accurately performs the linear transformation, with the bias being correctly applied across all batch samples.

___
### ReLU
`needle.nn.ReLU()`

Applies the rectified linear unit function element-wise:
$ReLU(x) = max(0, x)$.

If you have previously implemented ReLU's backwards pass in terms of itself, note that this is numerically unstable and will likely cause problems down the line.
Instead, consider that we could write the derivative of ReLU as $I\{x>0\}$, where we arbitrarily decide that the derivative at $x=0$ is 0.
(This is a _subdifferentiable_ function.)

**Code Implementation**
```python
class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return ops.relu(x)
        ### END YOUR SOLUTION
```

#### My explanation

**Subdifferentiable Function:** The note mentions that ReLU is a subdifferentiable function. This means that at the point $x = 0$, the function is not strictly differentiable because it has a "kink" or a sharp corner. However, in practice, we often define the derivative at this point to be 0 arbitrarily, which simplifies the computation and avoids instability.

```python
class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return ops.relu(x)
        ### END YOUR SOLUTION
```
**How the `compute` Method is Triggered**

1.  **Operation Definition (`Op` and `TensorOp` Classes)**:
    
    -   The `Op` class defines a general interface for operations (like addition, multiplication, etc.).
    -   The `TensorOp` class is a subclass of `Op` that is specialized for operations that output tensors.
2.  **Calling an Operation (`__call__` Method in `TensorOp`)**:
    
    -   When you call an operation, such as `ops.relu(x)`, it uses the `__call__` method of the `TensorOp` class.
    -   This method is responsible for creating a `Tensor` (or `TensorTuple`) that represents the result of the operation.
    -   Inside `__call__`, the method `Tensor.make_from_op(self, args)` is called, where `self` is the `ReLU` operation instance, and `args` contains the input tensor(s).
3.  **Creating a Tensor (`make_from_op` Method)**:
    
    -   The `make_from_op` method in the `Tensor` class creates a new `Tensor` object that represents the output of the operation.
    -   The operation (`op`) and the input tensor(s) (`inputs`) are stored in this `Tensor`.
    -   If the `LAZY_MODE` flag is `False`, the `Tensor` will immediately realize its data by calling the `realize_cached_data` method.
4.  **Realizing Cached Data (`realize_cached_data` Method)**:
    
    -   The `realize_cached_data` method checks if the data for the tensor has already been computed (cached). If not, it triggers the computation by calling the `compute` method of the operation.
    -   Specifically, it calls `self.op.compute(*[x.realize_cached_data() for x in self.inputs])`.
    -   This line of code triggers the `compute` method of the `ReLU` operation, which applies the ReLU function to the input tensor(s).


___
