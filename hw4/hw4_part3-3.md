### Convolution forward

Implement the forward pass of 2D multi-channel convolution in `ops.py`. You should probably refer to [this notebook](https://github.com/dlsyscourse/public_notebooks/blob/main/convolution_implementation.ipynb) from lecture, which implements 2D multi-channel convolution using im2col in numpy.

**Note:** Your convolution op should accept tensors in the NHWC format, as in the example above, and weights in the format (kernel_size, kernel_size, input_channels, output_channels).

However, you will need to add two additional features. Your convolution function should accept arguments for `padding` (default 0) and `stride` (default 1). For `padding`, you should simply apply your padding function to the spatial dimensions (i.e., axes 1 and 2).

Implementing strided convolution should consist of a relatively small set of changes to your plain convolution implementation.

We recommend implementing convolution without stride first, ensuring you pass some of the tests below, and then adding in stride.
