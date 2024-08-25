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

    def step(self):
        ### BEGIN YOUR SOLUTION
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
        ### END YOUR SOLUTION

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
        self.t += 1  # Increment time step

        for param in self.params:
            # Compute the gradient with L2 regularization (weight decay)
            regularized_grad = ndl.Tensor(param.grad, dtype='float32').data + self.weight_decay * param.data
            
            # Retrieve the previous first moment (m_t) and second moment (v_t) estimates, or use 0 if they don't exist
            m_t = self.m.get(param, 0)
            v_t = self.v.get(param, 0)
            
            # Update the first moment (m_t_plus_1) using the beta1 coefficient and the current gradient
            m_t_plus_1 = self.beta1 * m_t + (1 - self.beta1) * regularized_grad
            
            # Update the second moment (v_t_plus_1) using the beta2 coefficient and the square of the current gradient
            v_t_plus_1 = self.beta2 * v_t + (1 - self.beta2) * (regularized_grad ** 2)
    
            # Apply bias correction to the first moment (m_hat_plus_1)
            m_hat_plus_1 = m_t_plus_1 / (1 - self.beta1 ** self.t)
            
            # Apply bias correction to the second moment (v_hat_plus_1)
            v_hat_plus_1 = v_t_plus_1 / (1 - self.beta2 ** self.t)
    
            # Update the parameter using the bias-corrected moment estimates, the learning rate, and epsilon for numerical stability
            param.data -= self.lr * m_hat_plus_1 / (v_hat_plus_1 ** 0.5 + self.eps)
            
            # Store the updated first moment and second moment estimates for the next iteration
            self.m[param] = m_t_plus_1
            self.v[param] = v_t_plus_1
        ### END YOUR SOLUTION
