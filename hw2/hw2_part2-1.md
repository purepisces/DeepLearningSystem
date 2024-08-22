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
            # Create a broadcasted version of bias without modifying the original bias tensor
            bias_broadcasted = ops.broadcast_to(self.bias, output.shape)
            output = ops.add(output, bias_broadcasted)
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


### Sequential

`needle.nn.Sequential(*modules)`

Applies a sequence of modules to the input (in the order that they were passed to the constructor) and returns the output of the last module.

These should be kept in a `.module` property: you should _not_ redefine any magic methods like `__getitem__`, as this may not be compatible with our tests.

##### Parameters

- `*modules` - any number of modules of type `needle.nn.Module`


**Code implementation**
```python
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
```

#### My explanation

**Understanding Module**
Based on the code, a `Module` is an abstract representation of a neural network component. It could be a layer (like a linear layer, convolutional layer), a function (like ReLU), or a composite structure that combines several layers (like `Sequential`). The `Module` class is designed to be subclassed so that specific layers and models can be created by extending it.

```python
 def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
```
The `__call__` method makes an instance of `Module` callable, meaning you can use it like a function. When you call an instance of a `Module`, it automatically invokes the `forward` method.

**Understanding Sequential**
The `Sequential` class is a specific type of `Module` that is designed to chain together multiple `Module` objects and apply them in sequence to an input tensor. The `Sequential` class makes it easy to define a neural network by simply listing out the layers or operations in the order they should be applied.

```python
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
```

**Initialization (`__init__` method)**:

-   The `__init__` method in the `Sequential` class takes any number of `Module` objects as arguments, using the `*modules` syntax. This allows you to pass in a flexible number of modules (layers, activations, etc.).
-   These modules are stored in the `self.modules` attribute as a tuple.

>### Understanding `*modules` Syntax
>**`*args` in Python**:
> -   When you define a function or method with an argument like `*args`, it means that the function can accept any number of positional arguments. All these arguments are then collected into a single tuple named `args`.
> - For example:
```python
def example_function(*args):
    print(args)
example_function(1, 2, 3)
# This will print: `(1, 2, 3)`.
```
Example usage:
```python
model = Sequential(Linear(10, 20), ReLU(), Linear(20, 10))
output = model(input_tensor)
```
___




### LogSumExp

  

`needle.ops.LogSumExp(axes)`


Here you will need to implement one additional operatior in the `python/needle/ops/ops_logarithmic.py` file, as you did in HW1. Applies a numerically stable log-sum-exp function to the input by subtracting off the maximum elements.


$$\text{LogSumExp}(z) = \log (\sum_{i} \exp (z_i - \max{z})) + \max{z}$$

#### Parameters

- `axes` - Tuple of axes to sum and take the maximum element over. This uses the same conventions as `needle.ops.Summation()`


**Code implementation**
```python
class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, Z):
        ### BEGIN YOUR SOLUTION
        max_z_with_dim = array_api.max(Z, axis=self.axes, keepdims=True)
        max_z_no_dim = array_api.max(Z, axis=self.axes, keepdims=False)
        out = array_api.log(array_api.sum(array_api.exp(Z - max_z_with_dim), axis=self.axes, keepdims=False)) + max_z_no_dim
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

### My explanation for  def compute(self, Z)
```python
class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, Z):
        ### BEGIN YOUR SOLUTION
        max_z_with_dim = array_api.max(Z, axis=self.axes, keepdims=True)
        max_z_no_dim = array_api.max(Z, axis=self.axes, keepdims=False)
        out = array_api.log(array_api.sum(array_api.exp(Z - max_z_with_dim), axis=self.axes, keepdims=False)) + max_z_no_dim
        return out
        ### END YOUR SOLUTION
```

A **reduction operation** in the context of data processing and computational frameworks like NumPy, TensorFlow, or PyTorch refers to an operation that reduces the number of elements in an array or tensor by performing some form of aggregation. This aggregation is typically done along specific dimensions (axes) of the data.

Here are some common examples of reduction operations:

1.  **Sum (`np.sum`)**:
    
    -   Adds up all the elements in an array or along a specified axis.
    -   Example: Summing the elements in a matrix to get a single number (scalar) or a reduced matrix.

2.  **Maximum (`np.max`)**:
	-   Finds the maximum value among all the elements in an array or along a specified axis.
	-   Example: Finding the maximum value in each row or column of a matrix.

#### Explain `axis` in NumPy

The `axis` parameter in NumPy specifies the dimension along which an operation is performed. The key is to remember that:

-   **`axis=0`**: Operations are performed **down the rows** (i.e., along columns).
-   **`axis=1`**: Operations are performed **across the columns** (i.e., along rows).

**Example with `np.max`**
Given the array `Z`:
```python
Z = np.array([[1, 2, 3],
              [4, 5, 6]])  # Shape: (2, 3)
```
This is a 2D array (matrix) with 2 rows and 3 columns.

#### Case 1: `np.max(Z, axis=0)`

-   **Operation**: Find the maximum along `axis=0` (which means across rows, down each column).
-   **Result**: You get the maximum value in each column.
```python
max_axis0 = np.max(Z, axis=0)
# max_axis0 = [4, 5, 6]
# Shape: (3,)
```
Here, `axis=0` corresponds to reducing the rows, so the operation is applied across rows, resulting in one value per column.

#### Case 2: `np.max(Z, axis=1)`

-   **Operation**: Find the maximum along `axis=1` (which means across columns, for each row).
-   **Result**: You get the maximum value in each row.
```python
max_axis1 = np.max(Z, axis=1)
# max_axis1 = [3, 6]
# Shape: (2,)
```
Here, `axis=1` corresponds to reducing the columns, so the operation is applied across columns within each row, resulting in one value per row.

**Understanding whole code**
```python
class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, Z):
        ### BEGIN YOUR SOLUTION
        max_z_with_dim = array_api.max(Z, axis=self.axes, keepdims=True)
        max_z_no_dim = array_api.max(Z, axis=self.axes, keepdims=False)
        out = array_api.log(array_api.sum(array_api.exp(Z - max_z_with_dim), axis=self.axes, keepdims=False)) + max_z_no_dim
        return out
        ### END YOUR SOLUTION
  ```
 Let's go through an example using the `compute` method step by step, showing the shapes and the intermediate results for each line of the code.
#### Example Setup
Consider the following 2D array `Z`:
```python
import numpy as np

Z = np.array([[1, 2, 3],
              [4, 5, 6]])  # Shape: (2, 3)
```
We'll assume that `self.axes = 1`, meaning we are going to apply the `LogSumExp` operation across the columns (axis 1) of the array `Z`.

### Step-by-Step Execution

#### 1. Compute `max_z_with_dim`
```python
max_z_with_dim = np.max(Z, axis=1, keepdims=True)
```
-   **Operation**: Compute the maximum value along the specified axis (`axis=1`), while keeping the dimensions.
-   **Result**:
	```python
	max_z_with_dim = np.array([[3],   # Max of [1, 2, 3] is 3
                               [6]])  # Max of [4, 5, 6] is 6
	```
- **Shape**: `(2, 1)`

	-   The result keeps the reduced dimension as a size-1 dimension, which is crucial for broadcasting in the next steps.
	
#### 2. Compute `max_z_no_dim`
```python
max_z_no_dim = np.max(Z, axis=1, keepdims=False)
```
-   **Operation**: Compute the maximum value along the specified axis (`axis=1`), without keeping the dimensions.
-   **Result**
	```python
	max_z_no_dim = np.array([3, 6])  # Max of [1, 2, 3] is 3, 		Max of [4, 5, 6] is 6
	```
- **Shape**: `(2,)`
	-   The result is a 1D array where the dimension over which the operation was performed has been collapsed.

#### 3. Compute the LogSumExp
```python
shifted = Z - max_z_with_dim
exp_shifted = np.exp(shifted)
sum_exp_shifted = np.sum(exp_shifted, axis=1, keepdims=False)
log_sum_exp_shifted = np.log(sum_exp_shifted)
```
Let's break it down further:

**a. Subtract `max_z_with_dim` from `Z`**:
```python
shifted = Z - max_z_with_dim
# shifted = [[1-3, 2-3, 3-3],   # Subtract 3 from each element in the first row
#            [4-6, 5-6, 6-6]]   # Subtract 6 from each element in the second row
#         = [[-2, -1, 0],
#            [-2, -1, 0]]
# Shape: (2, 3)
```
-   **Shape**: `(2, 3)` (same as `Z`), because `max_z_with_dim` was broadcast to match the shape of `Z`.

**b. Exponentiate the shifted result**:
```python
exp_shifted = np.exp(shifted)
# exp_shifted = [[exp(-2), exp(-1), exp(0)],
#                [exp(-2), exp(-1), exp(0)]]
#             = [[0.1353, 0.3679, 1.0000],
#                [0.1353, 0.3679, 1.0000]]
# Shape: (2, 3)
```
-   **Shape**: `(2, 3)`, since exponentiation is applied element-wise.

**c. Sum the exponentiated values along `axis=1`**:
```python
sum_exp_shifted = np.sum(exp_shifted, axis=1, keepdims=False)
# sum_exp_shifted = [0.1353 + 0.3679 + 1.0000,  # Sum across the columns for the first row
#                    0.1353 + 0.3679 + 1.0000]  # Sum across the columns for the second row
#                = [1.5032, 1.5032]
# Shape: (2,)
```
-   **Shape**: `(2,)`, because we summed along `axis=1`, collapsing that dimension.

**d. Take the logarithm of the summed values**:
```python
log_sum_exp_shifted = np.log(sum_exp_shifted)
# log_sum_exp_shifted = [log(1.5032), log(1.5032)]
#                    = [0.4076, 0.4076]
# Shape: (2,)
```
-   **Shape**: `(2,)`, since the logarithm is applied element-wise to the summed result.

#### 4. Add `max_z_no_dim` to `log_sum_exp_shifted`
```python
out = log_sum_exp_shifted + max_z_no_dim
# out = [0.4076 + 3, 0.4076 + 6]
#     = [3.4076, 6.4076]
# Shape: (2,)
```
-   **Operation**: Add `max_z_no_dim` to `log_sum_exp_shifted`, element-wise.
-   **Result**:
	```python
	out = np.array([3.4076, 6.4076])
	```
- **Shape**: `(2,)`, matching the shape of `log_sum_exp_shifted` and `max_z_no_dim`.

### Final Output
The final output of the `compute` method is:
```python
out = np.array([3.4076, 6.4076])  # Shape: (2,)
```
### Summary of Shapes

1.  **`max_z_with_dim`**: Shape `(2, 1)` — Maximum values along the specified axis, with dimensions retained.
2.  **`max_z_no_dim`**: Shape `(2,)` — Maximum values along the specified axis, without retaining dimensions.
3.  **`shifted`**: Shape `(2, 3)` — Result of subtracting `max_z_with_dim` from `Z`.
4.  **`exp_shifted`**: Shape `(2, 3)` — Exponentiated values after the shift.
5.  **`sum_exp_shifted`**: Shape `(2,)` — Summed exponentiated values along the specified axis.
6.  **`log_sum_exp_shifted`**: Shape `(2,)` — Logarithm of the summed exponentiated values.
7.  **`out`**: Shape `(2,)` — Final result after adding back the max values without dimensions.


### First Solution: My explanation for  def gradient(self, out_grad, node):
```python
def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        if self.axes is None:
            self.axes = tuple(range(len(node.inputs[0].shape)))
        z = node.inputs[0]
        shape = [1 if i in self.axes else z.shape[i] for i in range(len(z.shape))]
        gradient = exp(z - node.reshape(shape).broadcast_to(z.shape))
        return out_grad.reshape(shape).broadcast_to(z.shape)*gradient
        ### END YOUR SOLUTION
```

### Derivation Process 
#### Symbol Explanation 
The symbols used in the derivation process are explained as follows:

$$z \in \mathbb{R}^n$$

$$z_k = \max z$$

$$\hat{z} = z - \max z$$

$$f = \log \sum_{i=1}^{n} \exp(z_i - \max z) + \max z$$

$$= \log \sum_{i=1}^{n} \exp \hat{z}_i + z_k$$


### Non-Maximum Case Derivation

When $z_j \neq z_k$, the derivation of $\frac{\partial f}{\partial z_j}$ is as follows:


$$\begin{equation}\begin{aligned}
\frac{\partial f}{\partial z_j} &= \frac{\partial \left( \log \sum_{i=1}^{n} \exp \hat{z}_i \right)}{\partial z_j} + \frac{\partial z_k}{\partial z_j} \\
&= \frac{\partial \left( \log \sum_{i=1}^{n} \exp \hat{z}_i \right)}{\partial z_j} \cdot \frac{\partial \sum_{i=1}^{n} \exp \hat{z}_i}{\partial z_j} + 0 \\
&= \frac{1}{\sum_{i=1}^{n} \exp \hat{z}_i} \cdot \left(\sum_{i \neq j} \frac{\partial \exp \hat{z}_i}{\partial z_j} + \frac{\partial \exp \hat{z}_j}{\partial z_j}\right) \\
&= \frac{1}{\sum_{i=1}^{n} \exp \hat{z}_i} \cdot \left(0 + \exp \hat{z}_j\right) \\
&= \frac{\exp \hat{z}_j}{\sum_{i=1}^{n} \exp \hat{z}_i}\end{aligned}\end{equation}$$


### Maximum Case Derivation

When $z_j = z_k$, the derivation of $\frac{\partial f}{\partial z_j}$ is as follows:

$$\begin{equation}
\begin{aligned}
\frac{\partial f}{\partial z_j} &= \frac{\partial \left( \log \sum_{i=1}^{n} \exp \hat{z}_i \right)}{\partial z_j} + \frac{\partial z_k}{\partial z_j} \\
&= \frac{\partial \left( \log \sum_{i=1}^{n} \exp \hat{z}_i \right)}{\partial z_j} \cdot \frac{\partial \sum_{i=1}^{n} \exp \hat{z}_i}{\partial z_j} + 1 \\
&= \frac{1}{\sum_{i=1}^{n} \exp \hat{z}_i} \cdot \left[ \sum_{z_i \neq z_k} \frac{\partial \exp(z_i - z_k)}{\partial z_j} + \sum_{z_i = z_k} \frac{\partial \exp(z_i - z_k)}{\partial z_j} \right] + 1 \\
&= \frac{1}{\sum_{i=1}^{n} \exp \hat{z}_i} \cdot \left[ \sum_{z_i \neq z_k} - \exp(z_i - z_k) + 0 \right] + 1
\end{aligned}\end{equation}$$

Note, in the above equation, $z_j = z_k$

$$\begin{equation}
\begin{aligned} &= 1 - \frac{\sum_{z_i \neq z_k} \exp(z_i - z_k)}{\sum_{i=1}^{n} \exp \hat{z}_i}\\
&= \frac{\exp \hat{z}_j}{\sum_{i=1}^{n} \exp \hat{z}_i}
\end{aligned}\end{equation}$$

### General Case

Note that whether $z_j$ is the maximum value or not, the following holds:

$$\frac{\partial f}{\partial z_j} = \frac{\exp \hat{z}_j}{\sum_{i=1}^{n} \exp \hat{z}_i} = \exp(z_j - \text{LogSumExp}(z) )$$

### Prove $\frac{\exp \hat{z}_j}{\sum_{i=1}^{n} \exp \hat{z}_i} = \exp(z_j - \text{LogSumExp}(z) )$
We need to prove that:

$$\frac{\exp(Z_i - \max(Z))}{\sum_{j} \exp(Z_j - \max(Z))} = \exp\left(Z_i - \text{LogSumExp}(Z)\right)$$

#### Step 1: Start with the Left Side
Start with the left side:

$$\frac{\exp(Z_i - \max(Z))}{\sum_{j} \exp(Z_j - \max(Z))}$$

#### Step 2: Express the Denominator Using Logarithm and Exponential
Recognize that the denominator can be rewritten using the logarithm-exponential identity:

$$\sum_{j} \exp(Z_j - \max(Z)) = \exp\left(\log\left(\sum_{j} \exp(Z_j - \max(Z))\right)\right)$$

Thus, the expression becomes:

$$\frac{\exp(Z_i - \max(Z))}{\exp\left(\log\left(\sum_{j} \exp(Z_j - \max(Z))\right)\right)}$$

#### Step 3: Simplify the Fraction
Now, recall the identity:

$$\frac{\exp(A)}{\exp(B)} = \exp(A - B)$$

Applying this identity to our expression:

$$\frac{\exp(Z_i - \max(Z))}{\exp\left(\log\left(\sum_{j} \exp(Z_j - \max(Z))\right)\right)} = \exp\left((Z_i - \max(Z)) - \log\left(\sum_{j} \exp(Z_j - \max(Z))\right)\right)$$

#### Step 4: Combine the Terms
Simplify the expression by combining terms:

$$\exp\left(Z_i - \max(Z) - \log\left(\sum_{j} \exp(Z_j - \max(Z))\right)\right)$$

#### Step 5: Recognize the Expression for LogSumExp
Recall that:

$$\text{LogSumExp}(Z) = \log\left(\sum_{j} \exp(Z_j - \max(Z))\right) + \max(Z)$$

So:

$$\exp\left(Z_i - \max(Z) - \log\left(\sum_{j} \exp(Z_j - \max(Z))\right)\right) = \exp\left(Z_i - \left[\log\left(\sum_{j} \exp(Z_j - \max(Z))\right) + \max(Z)\right]\right)$$


#### Step 6: Simplify to the Final Form
Finally, recognize that the expression we've derived on the right is exactly the right side of the original equation:

$$\exp\left(Z_i - \text{LogSumExp}(Z)\right)$$

#### Conclusion
We have successfully shown that starting from the left side of the equation and using basic logarithm and exponential identities, we can derive the right side. This proves that:

$$\frac{\exp(Z_i - \max(Z))}{\sum_{j} \exp(Z_j - \max(Z))} = \exp\left(Z_i - \text{LogSumExp}(Z)\right)$$

### Understanding `if self.axes is None: self.axes = tuple(range(len(node.inputs[0].shape)))`

**`self.axes`**: These are the axes along which the `LogSumExp` was computed. If `self.axes` is `None`, it means the `LogSumExp` was computed across all axes, resulting in a scalar output.


### Understanding `shape = [1 if i in self.axes else z.shape[i] for i in range(len(z.shape))]`

### Example

Suppose:

-   `z.shape` is `(2, 3, 4)`
-   `self.axes` is `(1, 2)`

The list comprehension would generate `shape` as follows:

-   **For `i = 0`**: `0` is **not** in `self.axes`, so `shape[0]` will be `z.shape[0] = 2`.
-   **For `i = 1`**: `1` **is** in `self.axes`, so `shape[1]` will be `1`.
-   **For `i = 2`**: `2` **is** in `self.axes`, so `shape[2]` will be `1`.

Thus, `shape` will be `[2, 1, 1]`.



### Understanding `node.reshape(shape).broadcast_to(z.shape)`


### Example

Let’s say `Z` has a shape `(3, 4, 5)` and `self.axes = (1, 2)`.

1.  **Original `Z` Shape**: `(3, 4, 5)`
2.  **Output `node` Shape**: After the `LogSumExp`, if computed along axes `(1, 2)`, the shape of `node` would be `(3,)`, as the `LogSumExp` reduces the second and third dimensions.
3.  **`shape` Computation**:
    -   `self.axes` is `(1, 2)`, so we want to reduce these dimensions in `shape`.
    -   `shape` becomes `[3, 1, 1]`, which aligns with `z.shape` but reduces the dimensions where `LogSumExp` was applied.
4.  **Reshape and Broadcast**:
    -   `node.reshape(shape)` results in a tensor with shape `(3, 1, 1)`.
    -   Broadcasting this to `z.shape` results in a tensor of shape `(3, 4, 5)`, which can now be subtracted from `z` in the element-wise operation.

### Second Solution: My explanation for  def gradient(self, out_grad, node):(Prefer this one🌟)
```python
def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        if self.axes is None:
            self.axes = tuple(range(len(node.inputs[0].shape)))
        z = node.inputs[0]
        shape = [1 if i in self.axes else z.shape[i] for i in range(len(z.shape))]
        gradient = exp(z - node.reshape(shape).broadcast_to(z.shape))
        return out_grad.reshape(shape).broadcast_to(z.shape)*gradient
        ### END YOUR SOLUTION
```

$$f = \log \sum_{i=1}^{n} \exp(z_i - \max z) + \max z$$

#### Prove $\log\left(\sum \exp(z - C)\right) + C = \log\left(\sum \exp(z)\right)$ 
The expression $\log\left(\sum \exp(z - C)\right) + C = \log\left(\sum \exp(z)\right)$ is true because the term $C$ inside the exponential function acts as a constant offset that can be factored out of the sum, simplifying the logarithm.

Here's how it works:

#### Starting with the Left-Hand Side

$$\log\left(\sum \exp(z - C)\right) + C$$

#### Factor the Exponential Term
Inside the summation, $z - C$ can be rewritten as $\exp(z) \cdot \exp(-C)$. Since $\exp(-C)$ is independent of $z$, it can be factored out of the sum:

$$\log\left(\sum \exp(z - C)\right) = \log\left(\sum \exp(z) \cdot \exp(-C)\right)$$

$$= \log\left(\exp(-C) \sum \exp(z)\right)$$

#### Apply the Logarithm Property
Use the logarithm property $\log(ab) = \log(a) + \log(b)$:

$$\log\left(\exp(-C) \sum \exp(z)\right) = \log\left(\exp(-C)\right) + \log\left(\sum \exp(z)\right)$$

Since $\log(\exp(-C)) = -C$, this becomes:

$$-C + \log\left(\sum \exp(z)\right)$$


#### Add $C$ to Both Sides
Now, add the $C$ term that was outside the logarithm:

$$-C + \log\left(\sum \exp(z)\right) + C$$

The $-C$ and $+C$ cancel out, leaving:

$$\log\left(\sum \exp(z)\right)$$


#### Conclusion
Therefore, the equality holds:

$$\log\left(\sum \exp(z - C)\right) + C = \log\left(\sum \exp(z)\right)$$


This property is useful for numerical stability when computing the log-sum-exp function, especially when \(z\) contains large values that might cause overflow in the exponential calculation. By subtracting a constant $C$, often chosen as $\max(z)$, you can stabilize the computation without changing the result of the logarithm.

### Gradient of the Log-Sum-Exp Function

Let's find the gradient of the function:

$$f = \log \sum_{i=1}^{n} \exp(z_i - \max z) + \max z = f(z) = \log\left(\sum_{i=1}^{n} \exp(z_i)\right)$$

with respect to each $z_j$.

#### Step 1: Differentiate the Log-Sum-Exp Function
The function can be rewritten as:

$$f(z) = \log S(z),$$

where

$$S(z) = \sum_{i=1}^{n} \exp(z_i).$$

The gradient of $f(z)$ with respect to $z_j$ can be found using the chain rule:

$$\frac{\partial f(z)}{\partial z_j} = \frac{1}{S(z)} \cdot \frac{\partial S(z)}{\partial z_j}.$$

#### Step 2: Differentiate $S(z)$ with Respect to $z_j$
The function $S(z) = \sum_{i=1}^{n} \exp(z_i)$ is a sum of exponentials. The derivative of \( S(z) \) with respect to $z_j$ is simply the derivative of the $j$-th term in the sum, since all other terms are independent of $z_j$:

$$\frac{\partial S(z)}{\partial z_j} = \exp(z_j).$$

#### Step 3: Combine the Gradients
Substituting the result from Step 2 into the chain rule from Step 1:

$$\frac{\partial f(z)}{\partial z_j} = \frac{\exp(z_j)}{\sum_{i=1}^{n} \exp(z_i)}.$$

#### Conclusion
The gradient of 

$$f(z) = \log\left(\sum_{i=1}^{n} \exp(z_i)\right)$$

with respect to each $z_j$ is:

$$\frac{\partial f(z)}{\partial z_j} = \frac{\exp(z_j)}{\sum_{i=1}^{n} \exp(z_i)}.$$

This expression is also known as the softmax function, where each gradient term corresponds to the normalized exponential of the input values $z_i$.

### Rewriting the Gradient in Exponential Form

The expression

$$\frac{\exp(z_j)}{\sum_{i=1}^{n} \exp(z_i)}$$

can be rewritten as

$$\exp\left(z_j - \log\left(\sum_{i=1}^{n} \exp(z_i)\right)\right).$$

#### Explanation:

- **Starting Point**:
  
  We have the gradient expression:

  $$\frac{\partial f(z)}{\partial z_j} = \frac{\exp(z_j)}{\sum_{i=1}^{n} \exp(z_i)}.$$

- **Rewriting the Denominator**:
  
  We know that the denominator is:

  $$ \sum_{i=1}^{n} \exp(z_i).$$

  Taking the logarithm of this sum, we can express the gradient as:

  $$\frac{\partial f(z)}{\partial z_j} = \frac{\exp(z_j)}{\exp\left(\log\left(\sum_{i=1}^{n} \exp(z_i)\right)\right)}.$$

- **Simplifying Using Exponential and Logarithm Properties**:
  
  Since $\exp(\log(x)) = x$, we can rewrite the expression as:

  $$\frac{\partial f(z)}{\partial z_j} = \exp\left(z_j - \log\left(\sum_{i=1}^{n} \exp(z_i)\right)\right).$$

- **Conclusion**:
  
  Thus, the gradient can be expressed equivalently as:

  $$\frac{\partial f(z)}{\partial z_j} = \exp\left(z_j - \log\left(\sum_{i=1}^{n} \exp(z_i)\right)\right).$$

This formulation shows that the gradient of the log-sum-exp function is essentially a "softmax" function in a different form.

___


### SoftmaxLoss

  
`needle.nn.SoftmaxLoss()`

  
Applies the softmax loss as defined below (and as implemented in Homework 1), taking in as input a Tensor of logits and a Tensor of the true labels (expressed as a list of numbers, *not* one-hot encoded).


Note that you can use the `init.one_hot` function now instead of writing this yourself. Note: You will need to use the numerically stable logsumexp operator you just implemented for this purpose.

 
$$\ell_\text{softmax}(z,y) = \log \sum_{i=1}^k \exp z_i - z_y$$

  
Code Implementation:
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
        y_one_hot = init.one_hot(num_class, y)
        # Compute the correct class logits by multiplying logits with y_one_hot and summing over the class dimension. logits shape is (batch_size, num_class), y_one_hot shape is (batch_size, num_class), they multiplied element-wise. The shape of correct_class_logits is (batch_size,)
        correct_class_logits = ops.summation(logits * y_one_hot, axes=(1,))
    
        # Step 3: Compute the loss for each sample, the shape of losses is (batch_size,)
        losses = log_sum_exp - correct_class_logits
        
        # Step 4: Return the average loss across the batch, it is a scalar value
        return ops.summation(losses) / batch_size
        ### END YOUR SOLUTION
```

Example
```python
import numpy as np

# Logits for a batch of 3 samples and 4 classes
Z = np.array([[2.0, 1.0, 0.1, 0.5],
              [1.5, 2.1, 0.2, 0.7],
              [1.1, 1.8, 0.3, 0.4]])

# True labels for the 3 samples (in integer form, not one-hot encoded)
y = np.array([0, 1, 2])  # Corresponds to classes 0, 1, and 2 for each sample

print("Logits (Z):")
print(Z)
# Output:
# [[2.0 1.0 0.1 0.5]
#  [1.5 2.1 0.2 0.7]
#  [1.1 1.8 0.3 0.4]]

print("\nTrue labels (y):")
print(y)
# Output:
# [0 1 2]
--------------------------------------
# For the first sample:
# log(sum(exp([2.0, 1.0, 0.1, 0.5]))) = log(exp(2.0) + exp(1.0) + exp(0.1) + exp(0.5))
# = log(7.3891 + 2.7183 + 1.1052 + 1.6487) = log(12.8613) ≈ 2.554

# For the second sample:
# log(sum(exp([1.5, 2.1, 0.2, 0.7]))) = log(exp(1.5) + exp(2.1) + exp(0.2) + exp(0.7))
# = log(4.4817 + 8.1662 + 1.2214 + 2.0138) = log(15.8831) ≈ 2.764

# For the third sample:
# log(sum(exp([1.1, 1.8, 0.3, 0.4]))) = log(exp(1.1) + exp(1.8) + exp(0.3) + exp(0.4))
# = log(3.0042 + 6.0496 + 1.3499 + 1.4918) = log(11.8955) ≈ 2.476

log_sum_exp = np.array([2.554, 2.764, 2.476])
print("\nLog-Sum-Exp for each sample:")
print(log_sum_exp)
# Output:
# [2.554 2.764 2.476]
--------------------------------------
# Convert y (true labels) to one-hot encoded matrix
y_one_hot = np.eye(4)[y]

print("\nOne-hot encoded true labels (y_one_hot):")
print(y_one_hot)
# Output:
# [[1. 0. 0. 0.]
#  [0. 1. 0. 0.]
#  [0. 0. 1. 0.]]
--------------------------------------
# Element-wise multiplication of Z and y_one_hot
Z_times_y = Z * y_one_hot

print("\nElement-wise multiplication of Z and y_one_hot:")
print(Z_times_y)
# Output:
# [[2.0 0.0 0.0 0.0]
#  [0.0 2.1 0.0 0.0]
#  [0.0 0.0 0.3 0.0]]

# Summation along axis 1 to extract the correct class logits
correct_class_logits = np.sum(Z_times_y, axis=1)

print("\nCorrect class logits after summation:")
print(correct_class_logits)
# Output: [2.0 2.1 0.3]
--------------------------------------
# Subtract the correct class logits from the log-sum-exp values
losses = log_sum_exp - correct_class_logits

print("\nLosses for each sample:")
print(losses)
# Output:
# [0.554 0.664 2.176]
--------------------------------------
# Average the losses across all samples
average_loss = np.mean(losses)

print("\nAverage softmax loss:")
print(average_loss)
# Output: 1.131
```

## Prove the equation $\ell_\text{softmax}(z,y) = \log \sum_{i=1}^k \exp z_i - z_y$

**Equation for All Training Examples**:

$$H(Y, P) = -\sum_{i=1}^k Y_i \log(P_i) = H(Y, \sigma(z)) = -\sum\limits_{i=1}^k Y_i \log(\sigma(z)_i)$$

**Equation for One Training Example**:

$$H(Y, \sigma(z)) = -\log(\sigma(z)y) = -\log\left( \frac{\exp(z_y)}{\sum\limits_{j=1}^k \exp(z_j)} \right)$$


**Simplified Equation for One Training Example**:

$$H(Y, \sigma(z)) = -z_y + \log\left( \sum\limits_{j=1}^k \exp(z_j) \right)$$

This equation can be rewritten using the LogSumExp trick for numerical stability:

$$H(Y, \sigma(z)) = -z_y + \text{LogSumExp}(z)$$

Where:

$$\text{LogSumExp}(z) = \log \left(\sum_{j=1}^k \exp \left(z_j - \max(z)\right)\right) + \max(z)$$

#### Softmax Function

The softmax function converts logits (raw scores) into probabilities. For a vector of logits $z$ of length $k$, the softmax function $\sigma(z)$ is defined as:

$$\sigma(z)i = \frac{\exp(z_i)}{\sum\limits_{j=1}^k \exp(z_j)}$$

for $i = 1, \ldots, k$.

#### Cross-Entropy Loss

The cross-entropy loss measures the difference between the true labels and the predicted probabilities. For a true label vector $Y$ (one-hot encoded) and a predicted probability vector $P$ (output of the softmax function), the cross-entropy loss $H(Y, P)$ is defined as:

$$H(Y, P) = -\sum_{i=1}^k Y_i \log(P_i)$$

#### Connection Between Softmax and Cross-Entropy

When using the softmax function as the final layer in a neural network for multi-class classification, the predicted probability vector $P$ is given by:

$$P_i = \sigma(z) i = \frac{\exp(z_i)}{\sum\limits_{j=1}^k \exp(z_j)}$$

The cross-entropy loss then becomes:

$$H(Y, \sigma(z)) = -\sum_{i=1}^k Y_i \log(\sigma(z)_i)$$

For a single training example where the true class is $y$, $Y$ is a one-hot encoded vector where $Y_y = 1$ and $Y_i = 0$ for $i \neq y$. Thus, the cross-entropy loss simplifies to:

$$H(Y, \sigma(z)) = -\log(\sigma(z)y) = -\log\left( \frac{\exp(z_y)}{\sum\limits_{j=1}^k \exp(z_j)} \right)$$

Using properties of logarithms, this can be rewritten as:

$$H(Y, \sigma(z)) = -\left( \log(\exp(z_y)) - \log\left( \sum\limits_{j=1}^k \exp(z_j) \right) \right)$$

$$H(Y, \sigma(z)) = -z_y + \log\left( \sum\limits_{j=1}^k \exp(z_j) \right)$$


___
## Reference:
1. logsumexp的first solution参考了知乎大佬潜龙勿用的推导
