## Question 0

This homework builds off of Homework 1. First, in your Homework 2 directory, copy the files `python/needle/autograd.py`, `python/needle/ops/ops_mathematic.py` from your Homework 1.

***NOTE***: The default data type for the tensor is `float32`. If you want to change the data type, you can do so by setting the `dtype` parameter in the `Tensor` constructor. For example, `Tensor([1, 2, 3], dtype='float64')` will create a tensor with `float64` data type. 
In this homework, make sure any tensor you create has `float32` data type to avoid any issues with the autograder.

## Question 1

In this first question, you will implement a few different methods for weight initialization.  This will be done in the `python/needle/init/init_initializers.py` file, which contains a number of routines for initializing needle Tensors using various random and constant initializations.  Following the same methodology of the existing initializers (you will want to call e.g. `init.rand` or `init.randn` implemented in `python/needle/init/init_basic.py` from your functions below, implement the following common initialization methods.  In all cases, the functions should return `fan_in` by `fan_out` 2D tensors (extensions to other sizes can be done via e.g., reshaping).


### Xavier uniform
`xavier_uniform(fan_in, fan_out, gain=1.0, **kwargs)`

Fills the input Tensor with values according to the method described in [Understanding the difficulty of training deep feedforward neural networks](https://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf), using a uniform distribution. The resulting Tensor will have values sampled from $\mathcal{U}(-a, a)$ where 
\begin{equation}
a = \text{gain} \times \sqrt{\frac{6}{\text{fan_in} + \text{fan_out}}}
\end{equation}

Pass remaining `**kwargs` parameters to the corresponding `init` random call.

##### Parameters
- `fan_in` - dimensionality of input
- `fan_out` - dimensionality of output
- `gain` - optional scaling factor
___

### Xavier normal
`xavier_normal(fan_in, fan_out, gain=1.0, **kwargs)`

Fills the input Tensor with values according to the method described in [Understanding the difficulty of training deep feedforward neural networks](https://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf), using a normal distribution. The resulting Tensor will have values sampled from $\mathcal{N}(0, \text{std}^2)$ where 
\begin{equation}
\text{std} = \text{gain} \times \sqrt{\frac{2}{\text{fan_in} + \text{fan_out}}}
\end{equation}

##### Parameters
- `fan_in` - dimensionality of input
- `fan_out` - dimensionality of output
- `gain` - optional scaling factor
___

### Kaiming uniform
`kaiming_uniform(fan_in, fan_out, nonlinearity="relu", **kwargs)`

Fills the input Tensor with values according to the method described in [Delving deep into rectifiers: Surpassing human-level performance on ImageNet classification](https://arxiv.org/pdf/1502.01852.pdf), using a uniform distribution. The resulting Tensor will have values sampled from $\mathcal{U}(-\text{bound}, \text{bound})$ where 
\begin{equation}
\text{bound} = \text{gain} \times \sqrt{\frac{3}{\text{fan_in}}}
\end{equation}

Use the recommended gain value for ReLU: $\text{gain}=\sqrt{2}$.

##### Parameters
- `fan_in` - dimensionality of input
- `fan_out` - dimensionality of output
- `nonlinearity` - the non-linear function
___

### Kaiming normal
`kaiming_normal(fan_in, fan_out, nonlinearity="relu", **kwargs)`

Fills the input Tensor with values according to the method described in [Delving deep into rectifiers: Surpassing human-level performance on ImageNet classification](https://arxiv.org/pdf/1502.01852.pdf), using a uniform distribution. The resulting Tensor will have values sampled from $\mathcal{N}(0, \text{std}^2)$ where 
\begin{equation}
\text{std} = \frac{\text{gain}}{\sqrt{\text{fan_in}}}
\end{equation}

Use the recommended gain value for ReLU: $\text{gain}=\sqrt{2}$.

##### Parameters
- `fan_in` - dimensionality of input
- `fan_out` - dimensionality of output
- `nonlinearity` - the non-linear function
