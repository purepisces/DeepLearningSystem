
### LayerNorm1d

`needle.nn.LayerNorm1d(dim, eps=1e-5, device=None, dtype="float32")`

  

Applies layer normalization over a mini-batch of inputs as described in the paper [Layer Normalization](https://arxiv.org/abs/1607.06450).

  
$$y = w \circ \frac{x_i - \textbf{E}[x]}{((\textbf{Var}[x]+\epsilon)^{1/2})} + b$$
  

where $\textbf{E}[x]$ denotes the empirical mean of the inputs, $\textbf{Var}[x]$ denotes their empirical variance (note that here we are using the "biased" estimate of the variance, i.e., dividing by $N$ rather than by $N-1$), and $w$ and $b$ denote learnable scalar weights and biases respectively. Note you can assume the input to this layer is a 2D tensor, with batches in the first dimension and features in the second. You might need to broadcast the weight and bias before applying them.

  

##### Parameters

- `dim` - number of channels

- `eps` - a value added to the denominator for numerical stability.

  

##### Variables

- `weight` - the learnable weights of size `dim`, elements initialized to 1.

- `bias` - the learnable bias of shape `dim`, elements initialized to 0.


Code Implementation:
```python
class LayerNorm1d(Module):
    def __init__(self, dim, eps=1e-5, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.ones(dim, device=device, dtype=dtype))
        self.bias = Parameter(init.zeros(dim, device=device, dtype=dtype))
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
```

___
in the `LayerNorm1d` class, `self.dim` is intended to represent the `feature_size`, which is the number of features in each input tensor `x`.

### Explanation for LayerNorm1d

`LayerNorm1d` is implemented in this way to provide stable and effective feature normalization across the input features for each sample in the batch. In the LayerNorm1d implementation, each row (which corresponds to an individual input example) will have the same mean subtracted from all its elements, and it will be divided by the same variance. Specifically:

-   For each input example (each row in the tensor), the mean is computed across all the features in that row.
-   The same mean is then subtracted from each feature in that row.
-   The variance is also computed across all the features in that row.
-   Each feature in the row is divided by the same variance (after adding the small epsilon for numerical stability).

This means that all features within a single example (row) are normalized using the same mean and variance, ensuring that each feature in the example is standardized relative to the others in the same row. This helps the model to focus on the relationships between features within each example, rather than being influenced by the absolute values of those features.

####  Why This Matters:

-   **Without Normalization:** The model could be biased towards features with larger values, interpreting them as more significant purely based on their magnitude.
-   **With Normalization:** The model treats all features on a more equal footing, focusing more on the relationships between features (such as how they differ from the mean) rather than their raw magnitudes.
### Difference Between LayerNorm1d and BatchNorm

While `LayerNorm1d` normalizes features within each individual sample (row) in the batch, `BatchNorm` normalizes features across all samples in the batch. Here's how they differ:

-   **LayerNorm1d**:
    
    -   Normalizes each feature within a single sample (row).
    -   The mean and variance are computed across the features in each individual sample.
    -   It ensures that each feature in a sample is standardized relative to other features in that same sample.
    -   Used primarily when the relationships between features within each sample are critical, such as in NLP tasks or RNNs.
-   **BatchNorm**:
    
    -   Normalizes each feature across all samples in the batch.
    -   The mean and variance are computed across the batch for each feature (column).
    -   It ensures that each feature has a mean of 0 and a variance of 1 across the entire batch.
    -   Commonly used in deep CNNs to stabilize and accelerate training by normalizing the input to each layer.

In summary, `LayerNorm1d` focuses on normalizing the internal relationships within each sample, while `BatchNorm` focuses on normalizing the distribution of each feature across the batch. These different approaches make `LayerNorm1d` more suitable for tasks where the internal structure of each sample is important, whereas `BatchNorm` is better for tasks where consistent feature scaling across the batch is beneficial.

### Explanation of $\textbf{E}[x]$

$\textbf{E}[x]$ represents the expected value (or mean) of the random variable $x$. In the context of the implementation and in many statistical applications, $\textbf{E}[x]$ is often used to denote the mean or average value of $x$.

So, in the formula:


$$ \textbf{E}[x] = \mu = \frac{1}{N} \sum_{i=1}^{N} x_i$$

$\mu$ is the mean (or expected value) of the features, calculated as the sum of all feature values $x_i$ divided by the total number of features $N$.

### Explanation of $\textbf{Var}[x]$

The formula for variance ($\textbf{Var}[x]$) in the context of layer normalization is as follows:

$$\textbf{Var}[x] = \frac{1}{N} \sum_{i=1}^{N} (x_i - \mu)^2 = \frac{1}{N} \sum_{i=1}^{N} (x_i - E(x))^2$$

Where:

- $x_i$ is the input feature.
- $\mu$ is the mean of the input features (calculated across the features axis).
- $N$ is the number of features.

In the context of your implementation for `LayerNorm1d`, the variance is computed as:

1. **Compute the mean** $\mu$ across the features for each sample in the batch.
2. **Subtract the mean** from each feature value $x_i$ to get the deviation: $x_i - \mu$.
3. **Square** the deviation: $(x_i - \mu)^2$.
4. **Sum** the squared deviations across the features.
5. **Divide** by the number of features $N$ to get the variance.

So, the variance formula applied in code would be:

$$\text{var} = \frac{1}{\text{features}} \sum_{i=1}^{\text{features}} (x_i - \text{mean})^2$$

Code:
```python
x_minus_mean = x - broadcast_mean
var = ops.divide_scalar(ops.summation(x_minus_mean ** 2, axes=(1,)), features)
```

### An additional Explanation of reshape(-1,1)
`(-1, 1)` can be used to reshape tensors of various dimensions, not just 1D tensors. The key idea behind using `-1` is that it allows the reshape operation to infer the size of that dimension based on the number of elements in the original tensor and the specified size of the other dimensions.

#### How `(-1, 1)` Works:

-   **`-1`**: This dimension is automatically determined by the reshape operation. It calculates how many elements are needed to fill the new shape given the size of the other dimensions.
-   **`1`**: This explicitly sets the dimension to 1, turning the reshaped tensor into a column vector for that dimension.

#### Examples with Different Dimensions:

##### Example 1: Reshape a 1D Tensor to a 2D Column Vector
```python
a = np.array([1, 2, 3, 4, 5, 6])  # Shape: (6,)
reshaped_a = np.reshape(a, (-1, 1))  # Shape: (6, 1)
```
This reshapes the 1D tensor with 6 elements into a 2D tensor with 6 rows and 1 column.

##### Example 2: Reshape a 2D Tensor to Another 2D Tensor
```python
a = np.array([[1, 2, 3], [4, 5, 6]])  # Shape: (2, 3)
reshaped_a = np.reshape(a, (-1, 1))   # Shape: (6, 1)
```
Here, the 2D tensor with shape `(2, 3)` is reshaped into a 2D tensor with 6 rows and 1 column.

##### Example 3: Reshape a 3D Tensor to a 2D Tensor
```python
a = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])  # Shape: (2, 2, 2)
reshaped_a = np.reshape(a, (-1, 1))  # Shape: (8, 1)
```
In this case, the 3D tensor with shape `(2, 2, 2)` is flattened and reshaped into a 2D tensor with 8 rows.

#### Whole Example
```python
sums = ops.summation(x, axes=1)
mean = ops.divide_scalar(sums, features)
tmp = ops.reshape(mean, (-1, 1))
broadcast_mean = ops.broadcast_to(tmp, x.shape)
```
### Example Tensor `x`:

Let's assume `x` is a 2D tensor (matrix) with shape `(3, 4)`:

```python
import numpy as np
#Example tensor `x` with shape (3, 4)
x = np.array([[1, 2, 3, 4],
              [5, 6, 7, 8],
              [9, 10, 11, 12]])

print("Tensor x:")
print(x)

Tensor x:
[[ 1  2  3  4]
 [ 5  6  7  8]
 [ 9 10 11 12]]
sums = ops.summation(x, axes=1)
sums = np.sum(x, axis=1)
print("Summation across axis 1:", sums)
Summation across axis 1: [10 26 42]
mean = ops.divide_scalar(sums, features)
mean = sums / 4
print("Mean of each row:", mean)
Mean of each row: [ 2.5  6.5 10.5]
tmp = ops.reshape(mean, (-1, 1))
tmp = mean.reshape(-1, 1)
print("Reshaped mean (column vector):")
print(tmp)
Reshaped mean (column vector):
[[ 2.5]
 [ 6.5]
 [10.5]]
broadcast_mean = ops.broadcast_to(tmp, x.shape)
broadcast_mean = np.broadcast_to(tmp, x.shape)
print("Broadcasted mean:")
print(broadcast_mean)
Broadcasted mean:
[[ 2.5  2.5  2.5  2.5]
 [ 6.5  6.5  6.5  6.5]
 [10.5 10.5 10.5 10.5]]
 ```
