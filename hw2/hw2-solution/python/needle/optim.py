"""Optimization module"""
import needle as ndl
import numpy as np


class Optimizer:
    def __init__(self, params):
        self.params = params

    def step(self):
        raise NotImplementedError()

    def reset_grad(self):
        for p in self.params:
            p.grad = None


class SGD(Optimizer):
    def __init__(self, params, lr=0.01, momentum=0.0, weight_decay=0.0):
        super().__init__(params)
        self.lr = lr
        self.momentum = momentum
        self.u = {}
        self.weight_decay = weight_decay
        for param in self.params:
          self.u[param] = ndl.zeros_like(param.data)


    def step(self):
        for param in self.params:
          if self.weight_decay:
            grad = param.grad + self.weight_decay * param.data
          else:
            grad = param.grad
          # print(type(param.grad))
          if self.momentum:
            self.u[param] = self.momentum * self.u[param] + (1 - self.momentum) * grad
            param.data -= self.lr * self.u[param]
          else:
            param.data -= self.lr * grad

    def clip_grad_norm(self, max_norm=0.25):
        """
        Clips gradient norm of parameters.
        """
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


class Adam(Optimizer):
    def __init__(
        self,
        params,
        lr=0.01,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        weight_decay=0.0,
    ):
        super().__init__(params)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0

        self.m = {}
        self.v = {}

    def step(self):
        ### BEGIN YOUR SOLUTION
        self.t += 1

        for param in self.params:
            # Skip parameters without gradients
            if param.grad is None:
                continue

            if self.weight_decay:
              grad = param.grad + self.weight_decay * param.data
            else:
              # print("no weight_decay")
              grad = param.grad
            # Initialize m and v for parameter if not already done
            if param not in self.m:
                self.m[param] = ndl.zeros_like(param.data)
            if param not in self.v:
                self.v[param] = ndl.zeros_like(param.data)

            # Calculate m and v updates
            self.m[param] = self.beta1 * self.m[param] + (1 - self.beta1) * grad
            self.v[param] = self.beta2 * self.v[param] + (1 - self.beta2) * grad**2

            # Bias correction
            m_corrected = self.m[param] / (1 - self.beta1**self.t)
            v_corrected = self.v[param] / (1 - self.beta2**self.t)
            # print(v_corrected.dtype)

            # Update parameter
            # print("before: ", param.data.shape)
            param.data -= self.lr * m_corrected / (ndl.power_scalar(v_corrected, 0.5) + self.eps)
            # print("after: ", param.data.shape)
        ### END YOUR SOLUTION
