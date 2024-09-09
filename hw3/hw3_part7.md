## Part 7: CUDA Backend - Elementwise and scalar operations

Implement the following functions in `ndarray_backend_cuda.cu`:

* `EwiseMul()`, `ScalarMul()`

* `EwiseDiv()`, `ScalarDiv()`

* `ScalarPower()`

* `EwiseMaximum()`, `ScalarMaximum()`

* `EwiseEq()`, `ScalarEq()`

* `EwiseGe()`, `ScalarGe()`

* `EwiseLog()`

* `EwiseExp()`

* `EwiseTanh()`

Again, we don't provide these function prototypes, and you're welcome to use C++ templates or macros to make this implementation more compact. You will also want to uncomment the appropriate regions of the Pybind11 code once you've implemented each function.

**Code Implementation**
```cpp
template <typename Func>
__global__ void EwiseOpKernel(const scalar_t* a, const scalar_t* b, scalar_t* out, size_t size, Func func) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) {
    out[gid] = func(a[gid], b[gid]);
  }
}

template <typename Func>
void EwiseOp(const CudaArray& a, const CudaArray& b, CudaArray* out, Func func) {
  CudaDims dim = CudaOneDim(out->size);
  EwiseOpKernel<<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, out->size, func);
}
template <typename Func>
__global__ void ScalarOpKernel(const scalar_t* a, scalar_t val, scalar_t* out, size_t size, Func func) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) {
    out[gid] = func(a[gid], val);
  }
}

template <typename Func>
void ScalarOp(const CudaArray& a, scalar_t val, CudaArray* out, Func func) {
  CudaDims dim = CudaOneDim(out->size);
  ScalarOpKernel<<<dim.grid, dim.block>>>(a.ptr, val, out->ptr, out->size, func);
}
// Element-wise multiplication
void EwiseMul(const CudaArray& a, const CudaArray& b, CudaArray* out) {
  EwiseOp(a, b, out, [] __device__ (scalar_t x, scalar_t y) { return x * y; });
}

// Scalar multiplication
void ScalarMul(const CudaArray& a, scalar_t val, CudaArray* out) {
  ScalarOp(a, val, out, [] __device__ (scalar_t x, scalar_t y) { return x * y; });
}

// Element-wise division
void EwiseDiv(const CudaArray& a, const CudaArray& b, CudaArray* out) {
  EwiseOp(a, b, out, [] __device__ (scalar_t x, scalar_t y) { return x / y; });
}

// Scalar division
void ScalarDiv(const CudaArray& a, scalar_t val, CudaArray* out) {
  ScalarOp(a, val, out, [] __device__ (scalar_t x, scalar_t y) { return x / y; });
}

// Scalar power
void ScalarPower(const CudaArray& a, scalar_t val, CudaArray* out) {
  ScalarOp(a, val, out, [] __device__ (scalar_t x, scalar_t y) { return pow(x, y); });
}

// Element-wise maximum
void EwiseMaximum(const CudaArray& a, const CudaArray& b, CudaArray* out) {
  EwiseOp(a, b, out, [] __device__ (scalar_t x, scalar_t y) { return max(x, y); });
}

// Scalar maximum
void ScalarMaximum(const CudaArray& a, scalar_t val, CudaArray* out) {
  ScalarOp(a, val, out, [] __device__ (scalar_t x, scalar_t y) { return max(x, y); });
}

// Element-wise equality
void EwiseEq(const CudaArray& a, const CudaArray& b, CudaArray* out) {
  EwiseOp(a, b, out, [] __device__ (scalar_t x, scalar_t y) { return (x == y) ? 1.0f : 0.0f; });
}

// Scalar equality
void ScalarEq(const CudaArray& a, scalar_t val, CudaArray* out) {
  ScalarOp(a, val, out, [] __device__ (scalar_t x, scalar_t y) { return (x == y) ? 1.0f : 0.0f; });
}

// Element-wise greater than or equal
void EwiseGe(const CudaArray& a, const CudaArray& b, CudaArray* out) {
  EwiseOp(a, b, out, [] __device__ (scalar_t x, scalar_t y) { return (x >= y) ? 1.0f : 0.0f; });
}

// Scalar greater than or equal
void ScalarGe(const CudaArray& a, scalar_t val, CudaArray* out) {
  ScalarOp(a, val, out, [] __device__ (scalar_t x, scalar_t y) { return (x >= y) ? 1.0f : 0.0f; });
}

// Element-wise logarithm
void EwiseLog(const CudaArray& a, CudaArray* out) {
  ScalarOp(a, 0.0f, out, [] __device__ (scalar_t x, scalar_t) { return log(x); });
}

// Element-wise exponential
void EwiseExp(const CudaArray& a, CudaArray* out) {
  ScalarOp(a, 0.0f, out, [] __device__ (scalar_t x, scalar_t) { return exp(x); });
}

// Element-wise tanh
void EwiseTanh(const CudaArray& a, CudaArray* out) {
  ScalarOp(a, 0.0f, out, [] __device__ (scalar_t x, scalar_t) { return tanh(x); });
}
```
```cpp
  m.def("ewise_mul", EwiseMul);
  m.def("scalar_mul", ScalarMul);
  m.def("ewise_div", EwiseDiv);
  m.def("scalar_div", ScalarDiv);
  m.def("scalar_power", ScalarPower);

  m.def("ewise_maximum", EwiseMaximum);
  m.def("scalar_maximum", ScalarMaximum);
  m.def("ewise_eq", EwiseEq);
  m.def("scalar_eq", ScalarEq);
  m.def("ewise_ge", EwiseGe);
  m.def("scalar_ge", ScalarGe);

  m.def("ewise_log", EwiseLog);
  m.def("ewise_exp", EwiseExp);
  m.def("ewise_tanh", EwiseTanh);
```

### Explain 

This code is part of a CUDA backend implementation for performing a variety of element-wise and scalar operations on arrays in parallel. It uses **CUDA kernels** to execute these operations on the GPU, leveraging Pybind11 to expose the functions to Python, allowing users to invoke these operations from Python code. Below is a detailed explanation of each section.

1. **CUDA Kernel Templates for Element-wise and Scalar Operations**

The code defines two key kernel templates:

#### `EwiseOpKernel` (Element-wise operation kernel)
```cpp
template <typename Func>
__global__ void EwiseOpKernel(const scalar_t* a, const scalar_t* b, scalar_t* out, size_t size, Func func) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) {
    out[gid] = func(a[gid], b[gid]);
  }
}
```
-   This kernel performs element-wise operations on arrays `a` and `b`.
-   **Inputs**: Two arrays (`a` and `b`), an output array (`out`), the array size, and a custom operation (`Func func`).
-   **Operation**: For each element at index `gid`, it applies the function `func` to `a[gid]` and `b[gid]`, storing the result in `out[gid]`.

#### `ScalarOpKernel` (Scalar operation kernel)
```cpp
template <typename Func>
__global__ void ScalarOpKernel(const scalar_t* a, scalar_t val, scalar_t* out, size_t size, Func func) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) {
    out[gid] = func(a[gid], val);
  }
}
```

-   This kernel performs scalar operations, where the second input is a scalar value (`val`).
-   **Inputs**: An array `a`, a scalar `val`, an output array `out`, the array size, and a custom operation (`Func func`).
-   **Operation**: For each element at index `gid`, it applies the function `func` to `a[gid]` and `val`, storing the result in `out[gid]`.

2. **Helper Functions for Launching CUDA Kernels**

The kernel templates are invoked using the following helper functions:

#### `EwiseOp` (Element-wise operation launcher)

```cpp
template <typename Func>
void EwiseOp(const CudaArray& a, const CudaArray& b, CudaArray* out, Func func) {
  CudaDims dim = CudaOneDim(out->size);
  EwiseOpKernel<<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, out->size, func);
}
```

-   **Purpose**: To launch the element-wise kernel (`EwiseOpKernel`).
-   **Inputs**: Two input arrays (`a`, `b`), an output array (`out`), and a lambda function `func` representing the operation.
-   **Execution**: The function calculates the CUDA grid and block dimensions and launches the kernel on the GPU.

#### `ScalarOp` (Scalar operation launcher)
```cpp
template <typename Func>
void ScalarOp(const CudaArray& a, scalar_t val, CudaArray* out, Func func) {
  CudaDims dim = CudaOneDim(out->size);
  ScalarOpKernel<<<dim.grid, dim.block>>>(a.ptr, val, out->ptr, out->size, func);
}
```
-   **Purpose**: To launch the scalar operation kernel (`ScalarOpKernel`).
-   **Inputs**: An input array (`a`), a scalar value (`val`), an output array (`out`), and a lambda function `func` representing the operation.
-   **Execution**: Similar to `EwiseOp`, this function launches the scalar operation kernel.

 3. **Operations Implemented**

Various element-wise and scalar operations are defined using the above templates. These functions take in `CudaArray` objects, which are abstractions over GPU memory, and perform computations. The lambda expressions define the operations:

#### Element-wise Operations:
```cpp
void EwiseMul(const CudaArray& a, const CudaArray& b, CudaArray* out) {
  EwiseOp(a, b, out, [] __device__ (scalar_t x, scalar_t y) { return x * y; });
}
```
-   **Purpose**: Element-wise multiplication of `a` and `b`.
-   **Lambda**: The lambda defines the multiplication operation: `x * y`.

Similarly, other element-wise operations are defined:

-   **`EwiseDiv`**: Division of two arrays.
-   **`EwiseMaximum`**: Maximum of two arrays.
-   **`EwiseEq`**: Equality comparison of two arrays.
-   **`EwiseGe`**: Greater-than-or-equal comparison of two arrays.

#### Scalar Operations:
```cpp
void ScalarMul(const CudaArray& a, scalar_t val, CudaArray* out) {
  ScalarOp(a, val, out, [] __device__ (scalar_t x, scalar_t y) { return x * y; });
}
```
-   **Purpose**: Scalar multiplication, multiplying each element of `a` by `val`.
-   **Lambda**: The lambda defines the scalar multiplication: `x * y`.

Similarly, other scalar operations are defined:

-   **`ScalarDiv`**: Division by a scalar.
-   **`ScalarPower`**: Raise elements of `a` to a scalar power.
-   **`ScalarMaximum`**: Maximum between elements of `a` and a scalar.
-   **`ScalarEq`**: Equality comparison between elements of `a` and a scalar.
-   **`ScalarGe`**: Greater-than-or-equal comparison between elements of `a` and a scalar.

4. **Unary Operations**

Some operations like logarithm, exponential, and hyperbolic tangent operate on single arrays, and the same scalar template is reused with fixed lambda functions:

```cpp
void EwiseLog(const CudaArray& a, CudaArray* out) {
  ScalarOp(a, 0.0f, out, [] __device__ (scalar_t x, scalar_t) { return log(x); });
}
```
-   **Purpose**: Element-wise logarithm (`log(x)`).
-   **Lambda**: The lambda defines the logarithmic function.

Similarly:

-   **`EwiseExp`**: Element-wise exponential function (`exp(x)`).
-   **`EwiseTanh`**: Element-wise hyperbolic tangent (`tanh(x)`).

5. **Pybind11 Integration**

At the bottom of the code, these functions are bound to Python using Pybind11:

```cpp
m.def("ewise_mul", EwiseMul);
m.def("scalar_mul", ScalarMul);
m.def("ewise_div", EwiseDiv);
m.def("scalar_div", ScalarDiv);
m.def("scalar_power", ScalarPower);
...
```
-   This exposes the C++/CUDA functions to Python, allowing users to call these CUDA operations in Python code.
-   Example: Calling `ewise_mul` from Python will invoke the `EwiseMul` function implemented in C++/CUDA.

#### Summary

-   **CUDA Kernels**: Perform parallel element-wise and scalar operations on arrays.
-   **Templates**: Ensure code reuse for different operations (multiplication, division, etc.).
-   **Lambdas**: Define the specific operations performed by the kernels.
-   **Pybind11**: Exposes these functions to Python for easy usage in Python code.

This structure provides a scalable and compact implementation for running mathematical operations on GPUs using CUDA.
