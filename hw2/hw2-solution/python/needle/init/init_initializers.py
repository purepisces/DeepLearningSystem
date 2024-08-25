import math
from .init_basic import *


def xavier_uniform(fan_in, fan_out, gain=1.0, **kwargs):
    ### BEGIN YOUR SOLUTION
    # Calculate the range for uniform distribution
    a = gain * math.sqrt(6.0 / (fan_in + fan_out))
    # Generate a tensor with values from the uniform distribution
    return rand(fan_in, fan_out, low=-a, high=a, **kwargs)
    ### END YOUR SOLUTION


def xavier_normal(fan_in, fan_out, gain=1.0, **kwargs):
    ### BEGIN YOUR SOLUTION
    # Calculate the standard deviation
    std = gain * math.sqrt(2.0 / (fan_in + fan_out))
    # Generate a tensor with the normal distribution
    return randn(fan_in, fan_out, mean=0.0, std=std, **kwargs)
    ### END YOUR SOLUTION


def kaiming_uniform(fan_in, fan_out, nonlinearity="relu", **kwargs):
    assert nonlinearity == "relu", "Only relu supported currently"
    ### BEGIN YOUR SOLUTION
    # Use the recommended gain value for ReLU: gain = sqrt(2)
    gain = math.sqrt(2.0)

    # Calculate the bound for the uniform distribution
    bound = gain * math.sqrt(3.0 / fan_in)
    
    # Generate and return a tensor with values uniformly distributed between -bound and bound
    return rand(fan_in, fan_out, low=-bound, high=bound, **kwargs)
    ### END YOUR SOLUTION


def kaiming_normal(fan_in, fan_out, nonlinearity="relu", **kwargs):
    assert nonlinearity == "relu", "Only relu supported currently"
    ### BEGIN YOUR SOLUTION
    # Use the recommended gain value for ReLU: gain = sqrt(2)
    gain = math.sqrt(2.0)
    
    # Calculate the standard deviation for the normal distribution
    std = gain / math.sqrt(fan_in)
    
    # Generate and return a tensor with values from the normal distribution
    return randn(fan_in, fan_out, mean=0.0, std=std, **kwargs)
    ### END YOUR SOLUTION
