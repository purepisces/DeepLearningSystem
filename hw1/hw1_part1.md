## Question 1: Implementing forward computation

```python
class EWiseAdd(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a + b

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad, out_grad

def add(a, b):
    return EWiseAdd()(a, b)
```

- **Using `NDArray` in `compute`**: Keeps the method focused on efficient numerical operations.
- **Using `Tensor` in the Rest of the Framework**: Manages the computational graph, tracks operations, and stores gradients for automatic differentiation.
- **Separation of Concerns**: Ensures that numerical computations and gradient tracking are handled separately, leveraging the strengths of numpy for numerical tasks and `Tensor` for managing the differentiation process.
