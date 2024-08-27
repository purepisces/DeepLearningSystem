# Homework 3: Building an NDArray library

In this homework, you will build a simple backing library for the processing that underlies most deep learning systems: the n-dimensional array (a.k.a. the NDArray).  Up until now, you have largely been using numpy for this purpose, but this homework will walk you through developing what amounts to your own (albeit much more limited) variant of numpy, which will support both CPU and GPU backends.  What's more, unlike numpy (and even variants like PyTorch), you won't simply call out to existing highly-optimized variants of matrix multiplication or other manipulation code, but actually write your own versions that are reasonably competitive will the highly optimized code backing these standard libraries (by some measure, i.e., "only 2-3x slower" ... which is a whole lot better than naive code that can easily be 100x slower).  This class will ultimately be integrated into `needle`, but for this assignment you can _only_ focus on the ndarray module, as this will be the only subject of the tests.

## Getting familiar with the NDArray class

As you get started with this homework, you should first familiarize yourself with the `NDArray.py` class we have provided as a starting point for the assignment.  The code is fairly brief (it's ~500 lines, but a lot of these are comments provided for the functions you'll implement).

At its core, the NDArray class is a Python wrapper for handling operations on generic n-dimensional arrays.  Recall that virtually any such array will be stored internally as a vector of floating point values, i.e.,

```c++
float data[size];
```

and then the actual access to different dimensions of the array are all handled by additional fields (such as the array shape, strides, etc) that indicates how this "flat" array maps to n-dimensional structure.  In order to achieve any sort of reasonable speed, the "raw" operations (like adding, binary operations, but also more structured operations like matrix multiplication, etc), all need to be written at some level in some native language like C++ (including e.g., making CUDA calls).  But a large number of operations likes transposing, broadcasting, sub-setting of matrices, and other, can all be handled by just adjusting the high-level structure of the array, like it's strides.

The philosophy behind the NDArray class is that we want _all_ the logic for handling this structure of the array to be written in Python.  Only the "true" low level code that actually performs the raw underlying operations on the flat vector (as well as the code to manage these flat vectors, as they might need to e.g., be allocated on GPUs), is written in C++.  The precise nature of this separation will likely start to make more sense to you as you work through the assignment, but generally speaking everything that can be done in Python, is done in Python; often e.g., at the cost of some inefficiencies ... we call `.compact()` (which copies memory) liberally in order to make the underlying C++ implementations simpler.

In more detail, there are five fields within the NDArray class that you'll need to be familiar with (note that the real class member these all these fields is preceded by an underscore, e.g., `_handle`, `_strides`, etc, some of which are then exposed as a public property ... for all your code it's fine to use the internal, underscored version).

1. `device` - A object of type `BackendDevice`, which is a simple wrapper that contains a link to the underlying device backend (e.g., CPU or CUDA).
2. `handle` - A class objected that stores the underlying memory of the array.  This is allocated as a class of type `device.Array()`, though this allocation all happens in the provided code (specifically the `NDArray.make` function), and you don't need to worry about calling it yourself.
3. `shape` - A tuple specifying the size of each dimension in the array.
4. `strides` - A tuple specifying the strides of each dimension in the array.
5. `offset` - An integer indicating where in the underlying `device.Array` memory the array actually starts (it's convenient to store this so we can more easily manage pointing back to existing memory, without having to track allocations).

By manipulating these fields, even pure Python code can perform a lot of the needed operations on the array, such as permuting the dimensions (i.e., transposing), broadcasting, and more.  And then for the raw array manipulation calls, the `device` class has a number of methods (implemented in C++) that contains the necessary implementations.

There are a few points to note:

* Internally, the class can use _any_ efficient means of operating on arrays of data as a "device" backend.  Even, for example, a numpy array, but where instead of actually using the `numpy.ndarray` to represent the n-dimensional array, we just represent a "flat" 1D array in numpy, then call the relevant numpy methods to implement all the needed operators on this raw memory.  This is precisely what we do in the `ndarray_backend_numpy.py` file, which essentially provided a "stub reference" that just uses numpy for everything.  You can use this class to help you  better debug your own "real" implementations for the "native" CPU and GPU backends.
* Of particular importance for many of your Python implementations will be the `NDArray.make` call:
```python
def make(shape, strides=None, device=None, handle=None, offset=0):
```
which creates a new NDArray with the given shape, strides, device, handle, and offset.  If `handle` is not specified (i.e., no pre-existing memory is referenced), then the call will allocate the needed memory, but if handle _is_ specified then no new memory is allocated, but the new NDArray points the same memory as the old one.  It is important to efficient implementations that as many of your functions as possible _don't_ allocate new memory, so you will want to use this call in many cases to accomplish this.
* The NDArray has a `.numpy()` call that converts the array to numpy.  This is _not_ the same as the "numpy_device" backend: this creates an actual `numpy.ndarray` that is equivalent to the given NDArray, i.e., the same dimensions, shape, etc, though not necessarily the same strides (Pybind11 will reallocate memory for matrices that are returned in this manner, which can change the striding).

## Part 1: Python array operations

As a starting point for your class, implement the following functions in the `ndarray.py` file:

- `reshape()`
- `permute()`
- `broadcast_to()`
- `__getitem__()`

The inputs/outputs of these functions are all described in the docstring of the function stub.  It's important to emphasize that _none_ of these functions should reallocate memory, but should instead return NDArrays that share the same memory with `self`, and just use clever manipulation of shape/strides/etc in order to obtain the necessary transformations.

One thing to note is that the `__getitem__()` call, unlike numpy, will never change the number of dimensions in the array.  So e.g., for a 2D NDArray `A`, `A[2,2]` would return a still-2D with one row and one column.  And e.g. `A[:4,2]` would return a 2D NDarray with 4 rows and 1 column.

You can rely on the `ndarray_backend_numpy.py` module for all the code in this section.  You can also look at the results of equivalent numpy operations (the test cases should illustrate what these are).

After implementing these functions, you should pass/submit the following tests.  Note that we test all of these four functions within the test below, and you can incrementally try to pass more tests as you implement each additional function.
