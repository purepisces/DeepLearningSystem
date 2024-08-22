## Question 3

Implement the `step` function of the following optimizers in `python/needle/optim.py`.

Make sure that your optimizers _don't_ modify the gradients of tensors in-place.

We have included some tests to ensure that you are not consuming excessive memory, which can happen if you are not using `.data` or `.detach()` in the right places, thus building an increasingly large computational graph (not just in the optimizers, but in the previous modules as well).

You can ignore these tests, which include the string `memory_check` at your own discretion.

___

### SGD

`needle.optim.SGD(params, lr=0.01, momentum=0.0, weight_decay=0.0)`
  
Implements stochastic gradient descent (optionally with momentum, shown as $\beta$ below).

$$\begin{equation}
\begin{split}
u_{t+1} &= \beta u_t + (1-\beta) \nabla_\theta f(\theta_t) \\
\theta_{t+1} &= \theta_t - \alpha u_{t+1}
\end{split}
\end{equation}$$

##### Parameters

- `params` - iterable of parameters of type `needle.nn.Parameter` to optimize

- `lr` (*float*) - learning rate

- `momentum` (*float*) - momentum factor

- `weight_decay` (*float*) - weight decay (L2 penalty)


Code Implementation:
```python
class SGD(Optimizer):
    def __init__(self, params, lr=0.01, momentum=0.0, weight_decay=0.0):
        super().__init__(params)
        self.lr = lr
        self.momentum = momentum
        self.u = {}
        self.weight_decay = weight_decay

    def step(self):
        ## BEGIN YOUR SOLUTION
        for param in self.params:
	        # Calculate the gradient with L2 regularization (weight decay)
	        regularized_grad = ndl.Tensor(param.grad, dtype='float32').data + self.weight_decay * param.data
	        # Retrieve the previous velocity (if it exists), or use 0 if it doesn't
	        u_t = self.u.get(param, 0)
	        # Update the velocity (u_t_plus_1) using the momentum term and the current gradient
	        u_t_plus_1 = self.momentum * u_t + (1 - self.momentum) * regularized_grad
	        # Update the parameter using the velocity and learning rate
	        param.data = param.data - self.lr * u_t_plus_1
	        # Store the updated velocity for the next iteration
	        self.u[param] = u_t_plus_1
        ## END YOUR SOLUTION

    def clip_grad_norm(self, max_norm=0.25):
        """
        Clips gradient norm of parameters.
        """
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION
```
___


### Explanation of `self.momentum` and  `self.u`

**Without Momentum**: The standard vanilla Stochastic Gradient Descent (SGD) update rule is:

$$\theta_{t+1} = \theta_t - \alpha \nabla_\theta f(\theta_t)$$

Here, $\theta_t$​ represents the parameters at time step $t$, $\alpha$ is the learning rate, and $\nabla_\theta f(\theta_t)$ is the gradient of the loss function with respect to the parameters.

**With Momentum**: The Stochastic Gradient Descent (SGD) update rule is:

$$\begin{equation}
\begin{split}
u_{t+1} &= \beta u_t + (1-\beta) \nabla_\theta f(\theta_t) \\
\theta_{t+1} &= \theta_t - \alpha u_{t+1}
\end{split}
\end{equation}$$

-   **$u_{t+1}$** is the velocity update, which is a combination of the previous momentum term $u_t$ and the current gradient $\nabla_\theta f(\theta_t)$. The velocity is a moving average of the gradients, which accumulates over time based on the momentum factor $\beta$ and the current gradient.

-   **$\beta$**  is the momentum factor (or smoothing factor) that determines how much weight is given to the previous moving average $u_t$. A larger $\beta$ places more emphasis on past gradients, resulting in a smoother average, while a smaller $\beta$ makes the average more responsive to recent changes.   Typically, $\beta$ is a constant between 0 and 1 (e.g., 0.9), that determines the influence of past gradients.

-   **$\theta_{t+1}$** is the parameter update, where the parameters are adjusted based on the momentum-scaled gradient.

```python
class SGD(Optimizer):
    def __init__(self, params, lr=0.01, momentum=0.0, weight_decay=0.0):
        super().__init__(params)
        self.lr = lr
        self.momentum = momentum
        self.u = {}
        self.weight_decay = weight_decay
```
`self.momentum`:

-   **Purpose**: `self.momentum` is a scalar value (typically between 0 and 1) that controls how much of the previous update (or "velocity") is retained when computing the current update. It essentially smooths out the updates and can help accelerate the convergence of the optimization process, especially in the presence of noisy gradients.
-   **Role in Update Formula**: Momentum is used to compute a running average of the gradients. When momentum is 0, the optimizer behaves like standard vanilla SGD (without momentum). As momentum increases, the influence of previous gradients on the current update increases.

 `self.u`:

-   **Purpose**: `self.u` is a dictionary that stores the velocity (or the running average of gradients) for each parameter in the model. Each entry in this dictionary corresponds to a parameter and tracks its momentum-based update history. In this context, $u_t$ is often referred to as the "velocity" because it describes the direction and speed of the parameter updates.
-   **Role in Update Formula**: In the step function, `self.u` is updated with a weighted combination of the previous velocity and the current gradient.

-   **Update Mechanism**:
    
    -   For each parameter, `self.u.get(param, 0)` retrieves the previous velocity (or initializes it to 0 if not yet set).
    -   The new velocity $u_{t+1}$ is then computed as a weighted sum of the previous velocity (`self.u.get(param, 0)`) and the current gradient (`grad`). The weight for the previous velocity is `self.momentum`, and the weight for the current gradient is `(1 - self.momentum)`.
    
    Specifically, the update formula is:
    ```python
    u_t_plus_1 = self.momentum * u_t + (1 - self.momentum) * regularized_grad
    ```
    -   This means `u_t_plus_1` is a combination of the previous velocity (scaled by `self.momentum`) and the current gradient (scaled by `(1 - self.momentum)`).
    
-   **Parameter Update**: The parameter `param` is updated by applying the computed update step, `u_t_plus_1`, scaled by the learning rate `self.lr`. The parameter's value is modified in place, and the updated momentum term `u_t_plus_1` is stored in `self.u` for use in the next iteration.

```python
param.data = param.data - self.lr * u_t_plus_1
self.u[param] = u_t_plus_1
```

### In Summary

-   **`self.momentum`**: Controls the contribution of past gradients to the current update.
-   **`self.u`**: Stores the running average (velocity) of the gradients for each parameter, which is influenced by the momentum.

In the code, `self.u` is not "composed of" `self.momentum`, but rather, `self.momentum` determines how `self.u` is updated at each step, by blending the old velocity with the current gradient. This blending is what gives momentum-based SGD its characteristic smoothing effect.

### Explanation of moving/running average

A moving or running average is a technique used to smooth out fluctuations in data over time by calculating the average of a subset of data points, typically a fixed number of recent observations. In the context of optimization, such as in momentum-based gradient descent, a moving average is used to maintain a running total of past gradients. This approach helps in stabilizing the updates by reducing the impact of short-term variations in the gradients, thereby allowing the optimization process to focus on more consistent trends. The moving average is continuously updated as new data comes in, making it a dynamic tool for tracking trends over time.


### Explanation of `regularized_grad = ndl.Tensor(param.grad, dtype='float32').data + self.weight_decay * param.data`

In the context of stochastic gradient descent (SGD) with momentum and L2 regularization, the update rule can be expressed as:

$$u_{t+1} = \beta u_t + (1-\beta) \nabla_\theta f(\theta_t)$$

Here, the gradient term $\nabla_\theta f(\theta_t)$ is computed as:

$$\frac{\partial \mathcal{L}(\theta_t)}{\partial \theta_t} = \nabla_\theta f(\theta_t)$$

When incorporating L2 regularization (also known as weight decay), the gradient update becomes:

$$u_{t+1} = \beta u_t + (1-\beta) \left( \frac{\partial \mathcal{L}(\theta_t)}{\partial \theta_t} + \lambda \theta_t \right)$$

#### Detailed Explanation

1. **Original Loss Function with L2 Regularization:**

   Given a loss function $L(\theta)$, the L2 regularized loss function is expressed as:

   $$L_{\text{reg}}(\theta) = L(\theta) + \frac{\lambda}{2} \sum_{i} \theta_i^2$$

2. **Gradient of the Regularized Loss:**

   When you calculate the gradient of this regularized loss function with respect to the weights $\theta$, it results in:

   $$\frac{\partial L_{\text{reg}}(\theta)}{\partial \theta} = \frac{\partial L(\theta)}{\partial \theta} + \lambda \theta$$

   This equation shows that the gradient of the regularized loss is composed of the gradient of the original loss function $\frac{\partial L(\theta)}{\partial \theta}$ plus an additional term proportional to the weights $\lambda \theta$, which is introduced by the L2 regularization.

### Connection to the Code

The code `regularized_grad = ndl.Tensor(param.grad, dtype='float32').data + self.weight_decay * param.data` is performing this exact operation. It adds the weight decay term $\lambda \theta$ (where $\lambda$ is `self.weight_decay`) to the original gradient $\frac{\partial L(\theta)}{\partial \theta}$ to compute the regularized gradient before applying the momentum update.


### Explanation of param.grad

In python/needle/autograd.py
```python
def backward(self, out_grad=None):
        out_grad = (
            out_grad
            if out_grad
            else init.ones(*self.shape, dtype=self.dtype, device=self.device)
        )
        compute_gradient_of_variables(self, out_grad)
```
We define `node.grad` in `compute_gradient_of_variables`, the `grad` attribute is dynamically added to the tensor when needed (e.g., during backpropagation). Python allows this dynamic behavior, so you can access `param.grad` after it has been set by the gradient computation process, even though it wasn't initialized in the `__init__` method.

```python
def compute_gradient_of_variables(output_tensor, out_grad):
    """Take gradient of output node with respect to each node in node_list.

    Store the computed result in the grad field of each Variable.
    """
    node_to_output_grads_list: Dict[Tensor, List[Tensor]] = {}
    node_to_output_grads_list[output_tensor] = [out_grad]
    reverse_topo_order = list(reversed(find_topo_sort([output_tensor])))
    for node in reverse_topo_order:
        node_grad = sum_node_list(node_to_output_grads_list[node])
        if node.is_leaf():
            node.grad = node_grad
            continue
        gradients = node.op.gradient_as_tuple(node_grad, node)
        for i, inp in enumerate(node.inputs):
            if inp not in node_to_output_grads_list:
                node_to_output_grads_list[inp] = []
            node_to_output_grads_list[inp].append(gradients[i])
        node.grad = node_grad
    return
    ### END YOUR SOLUTION
    ```
Example
```python
class Example:
    def __init__(self, value):
        self.value = value

obj = Example(10)
print(hasattr(obj, 'grad'))  # False, 'grad' not defined yet

# Dynamically adding 'grad' attribute
obj.grad = 5
print(obj.grad)  # Outputs: 5
```
___
