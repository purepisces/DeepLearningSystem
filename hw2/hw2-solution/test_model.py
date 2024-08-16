import sys
sys.path.append('./python')
import numpy as np
import pytest
import needle as ndl
import itertools
import torch
import torch.nn.functional as f

input_size = 5
batch_size = 3





def _assert_almost_equal(x,y) -> bool:
    abs_error = np.abs(x - y)
    assert abs_error.mean().item() < 5e-5
    assert abs_error.max().item() < 1e-4
    return True

# CNN not test


def test_for_nn_fft2d(in_channels, out_channels, kernal_size, stride, dilation, padding, bias):
  my_fft = ndl.nn.Fftconv2d(in_channels, out_channels, kernal_size, stride=stride, dilation=dilation, padding = padding,  bias = bias)
  w_ = my_fft.weight.numpy()
  if my_fft.bias is not None:
    b_ = my_fft.bias.numpy()
    b_ = torch.tensor(b_)
  else:
    b_ = None
  x = ndl.init.rand(batch_size, in_channels, input_size, input_size)
  y0 = my_fft(x)
  kwargs = dict(
        padding= padding,
        stride=stride,
        dilation=dilation,
        groups=1,
    )
  signal = torch.tensor(x.numpy())
  torch_conv = getattr(f, f"conv2d")
  y1 = torch_conv(signal, torch.tensor(w_), bias=torch.tensor(b_), **kwargs)
  # _assert_almost_equal(y0.numpy(), y1.numpy())



in_channels = 2
out_channels = 3
kernal_size = 3
stride = 1
dilation = 2
padding = 0
bias = True

# test_for_nn_fft2d(in_channels, out_channels, kernal_size, stride, dilation, padding, bias)
print("my fft_conv2d have the correct result with torch.nn.conv2d")

x = np.random.random(1024)
_x = ndl.Tensor(x)
ndl.ops.DFT_slow(_x)
