import math
from .init_basic import *


def xavier_uniform(fan_in, fan_out, gain=1.0, **kwargs):
    ### BEGIN YOUR SOLUTION
    a = gain * (6.0 / (fan_in + fan_out)) ** 0.5
    return rand(fan_in, fan_out, low = -a, high = a, **kwargs)
    ### END YOUR SOLUTION


def xavier_normal(fan_in, fan_out, gain=1.0, **kwargs):
    ### BEGIN YOUR SOLUTION
    std = gain * (2.0 / (fan_in + fan_out)) ** 0.5
    return randn(fan_in, fan_out, mean = 0, std = std, **kwargs)
    ### END YOUR SOLUTION


def kaiming_uniform(fan_in, fan_out, shape=None, nonlinearity="relu", **kwargs):
    assert nonlinearity == "relu", "Only relu supported currently"
    ### BEGIN YOUR SOLUTION
    if shape is not None:
      receptive_field_size = shape[0] * shape[1]
      fan_in = shape[2] * receptive_field_size
      fan_out = shape[3] * receptive_field_size

    bound =  (6.0 / fan_in ) ** 0.5
    if shape:
      return rand(*shape, low = -bound, high = bound, **kwargs)
    else:
      # print("enter here?-----------")
      return rand(fan_in, fan_out, low = -bound, high = bound, **kwargs)
    ### END YOUR SOLUTION


def kaiming_normal(fan_in, fan_out, nonlinearity="relu", **kwargs):
    assert nonlinearity == "relu", "Only relu supported currently"
    ### BEGIN YOUR SOLUTION
    std = (2)**0.5 / ((fan_in)** 0.5)
    return randn(fan_in, fan_out, mean = 0, std = std, **kwargs)
    ### END YOUR SOLUTION
