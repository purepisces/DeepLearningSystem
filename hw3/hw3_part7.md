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
// Define functors for each operation
struct Mul {
  __device__ scalar_t operator()(scalar_t x, scalar_t y) const { return x * y; }
};

struct Div {
  __device__ scalar_t operator()(scalar_t x, scalar_t y) const { return x / y; }
};

struct Pow {
  __device__ scalar_t operator()(scalar_t x, scalar_t y) const { return pow(x, y); }
};

struct Max {
  __device__ scalar_t operator()(scalar_t x, scalar_t y) const { return max(x, y); }
};

struct Eq {
  __device__ scalar_t operator()(scalar_t x, scalar_t y) const { return (x == y) ? 1.0f : 0.0f; }
};

struct Ge {
  __device__ scalar_t operator()(scalar_t x, scalar_t y) const { return (x >= y) ? 1.0f : 0.0f; }
};

struct Log {
  __device__ scalar_t operator()(scalar_t x, scalar_t) const { return log(x); }
};

struct Exp {
  __device__ scalar_t operator()(scalar_t x, scalar_t) const { return exp(x); }
};

struct Tanh {
  __device__ scalar_t operator()(scalar_t x, scalar_t) const { return tanh(x); }
};

// Kernel for element-wise operations
template <typename Func>
__global__ void EwiseOpKernel(const scalar_t* a, const scalar_t* b, scalar_t* out, size_t size, Func func) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) {
    out[gid] = func(a[gid], b[gid]);
  }
}

// Kernel for scalar operations
template <typename Func>
__global__ void ScalarOpKernel(const scalar_t* a, scalar_t val, scalar_t* out, size_t size, Func func) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) {
    out[gid] = func(a[gid], val);
  }
}

// Launch function for element-wise operations
template <typename Func>
void EwiseOp(const CudaArray& a, const CudaArray& b, CudaArray* out, Func func) {
  CudaDims dim = CudaOneDim(out->size);
  EwiseOpKernel<<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, out->size, func);
}

// Launch function for scalar operations
template <typename Func>
void ScalarOp(const CudaArray& a, scalar_t val, CudaArray* out, Func func) {
  CudaDims dim = CudaOneDim(out->size);
  ScalarOpKernel<<<dim.grid, dim.block>>>(a.ptr, val, out->ptr, out->size, func);
}

// Element-wise multiplication
void EwiseMul(const CudaArray& a, const CudaArray& b, CudaArray* out) {
  EwiseOp(a, b, out, Mul());
}

// Scalar multiplication
void ScalarMul(const CudaArray& a, scalar_t val, CudaArray* out) {
  ScalarOp(a, val, out, Mul());
}

// Element-wise division
void EwiseDiv(const CudaArray& a, const CudaArray& b, CudaArray* out) {
  EwiseOp(a, b, out, Div());
}

// Scalar division
void ScalarDiv(const CudaArray& a, scalar_t val, CudaArray* out) {
  ScalarOp(a, val, out, Div());
}

// Scalar power
void ScalarPower(const CudaArray& a, scalar_t val, CudaArray* out) {
  ScalarOp(a, val, out, Pow());
}

// Element-wise maximum
void EwiseMaximum(const CudaArray& a, const CudaArray& b, CudaArray* out) {
  EwiseOp(a, b, out, Max());
}

// Scalar maximum
void ScalarMaximum(const CudaArray& a, scalar_t val, CudaArray* out) {
  ScalarOp(a, val, out, Max());
}

// Element-wise equality check
void EwiseEq(const CudaArray& a, const CudaArray& b, CudaArray* out) {
  EwiseOp(a, b, out, Eq());
}

// Scalar equality check
void ScalarEq(const CudaArray& a, scalar_t val, CudaArray* out) {
  ScalarOp(a, val, out, Eq());
}

// Element-wise greater-than-or-equal check
void EwiseGe(const CudaArray& a, const CudaArray& b, CudaArray* out) {
  EwiseOp(a, b, out, Ge());
}

// Scalar greater-than-or-equal check
void ScalarGe(const CudaArray& a, scalar_t val, CudaArray* out) {
  ScalarOp(a, val, out, Ge());
}

// Element-wise log
void EwiseLog(const CudaArray& a, CudaArray* out) {
  ScalarOp(a, 0.0f, out, Log());
}

// Element-wise exp
void EwiseExp(const CudaArray& a, CudaArray* out) {
  ScalarOp(a, 0.0f, out, Exp());
}

// Element-wise tanh
void EwiseTanh(const CudaArray& a, CudaArray* out) {
  ScalarOp(a, 0.0f, out, Tanh());
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

1. **Functor Definition for Operations**


In this code, **functors** (implemented as **`struct`** in C++) encapsulate specific mathematical operations such as multiplication, division, and exponentiation. Functors make the code more modular, reusable, and extendable by allowing operations to be passed as parameters to CUDA kernels. Each `struct` functor contains an overloaded `operator()` method, which defines the mathematical operation performed on its inputs. This design pattern enables easy substitution of different operations while maintaining a consistent structure for element-wise and scalar operations on arrays.

#### What is a `struct` in C++?

In C++ (and CUDA C++), a **`struct`** is a user-defined data type that can hold multiple data members and member functions. It differs from a `class` primarily in its default access level: members of a `struct` are **public** by default, making it useful for simple, function-like objects such as functors. In this context, **functors** are implemented as `struct` objects to represent mathematical operations that can be passed as arguments to CUDA kernels for execution on the GPU.

#### What is a Functor?

A **functor** is a class or `struct` that overloads the `operator()`, making objects of the class behave like a function. Functors provide an easy way to define inline, reusable operations, which can be passed to templates or kernels for execution. In CUDA, each functor's `operator()` is decorated with the **`__device__`** keyword, enabling it to run on the GPU.

#### Example: `Mul` Functor
```cpp
struct Mul {
  __device__ scalar_t operator()(scalar_t x, scalar_t y) const { return x * y; }
};
```
-   **`Mul`**: This functor defines a multiplication operation.
-   **`__device__`**: Indicates that the operation will be executed on the GPU.
-   **`operator()`**: Overloads the `()` operator, allowing the functor to behave like a function that takes two arguments (`x` and `y`), performs the multiplication, and returns the result (`x * y`).

Similarly, other operations are encapsulated in different functors:

-   **`Div`** for division,
-   **`Pow`** for power,
-   **`Max`** for maximum,
-   **`Eq`** for equality comparison, and so on.

#### Why Use Functors?

Using functors provides several advantages in the context of CUDA programming:

1.  **Modularity**: Each operation is encapsulated in its own functor (`struct`), making the code cleaner and easier to manage. The functor abstraction separates the operation logic from the kernel itself.
    
2.  **Reusability**: The same CUDA kernel (e.g., `EwiseOpKernel` and `ScalarOpKernel`) can perform different operations by simply passing a different functor. This avoids duplicating kernel code for every operation, enabling efficient reuse of kernel templates.
    
3.  **Efficiency**: By inlining operations via functors, we avoid function call overhead within CUDA kernels. This ensures high performance on the GPU, as operations are directly integrated into the kernel’s execution flow.

#### Explain Operator

**What is `operator()`?**

In C++, an **operator** is a symbol or function that operates on data. For example, the `+` operator adds two numbers, and the `()` operator calls a function. C++ allows you to **overload operators**, which means you can define custom behavior for these operators when they are used with objects of your class or `struct`.

In C++, **`operator()`** is known as the **function call operator**. It allows an object of a class or a `struct` to be invoked (called) like a regular function. When you define the `operator()` in a class or `struct`, you essentially allow objects of that class or `struct` to behave like functions. This is a feature that enables **functors** (function objects) in C++.


**What is Overloading?**

In C++, operators like `+`, `*`, and `()` can be **overloaded**, which means you can define how they work with objects of a class. When you overload `operator()`, it allows you to define what happens when an object is "called" like a function. This is how you make a functor. A **functor** is a class or `struct` that overloads the `operator()`. This allows objects of the class to behave like a function.


> In C++, an **object** is an instance of a class or `struct`, and a **functor** (also known as a function object) is a class or `struct` that defines an `operator()` function. When you create an instance (object) of this class, you can "call" that object like a regular function, even though it's an instance of a class.

In C++, **overloading the function call operator `()`** means that you can define how an object of a class or `struct` behaves when it is "called" like a function. Normally, functions in C++ are called using parentheses, like `func()`, but by overloading `operator()`, you can make objects of your class behave similarly.

In C++, the function call operator `()` has a predefined meaning—it's used to call functions. But when you define `operator()` inside a class, you **override** this predefined meaning for objects of that class. This is called **overloading** because you're providing a new definition for an existing operator (`()` in this case) when used with objects of your class.

**Example of `operator()` and Overloading**

Consider this example:
```cpp
#include <iostream>

struct Mul {
    int operator()(int x, int y) const {  // Overloading the () operator
        return x * y;  // Custom behavior: Multiply the two arguments
    }
};

int main() {
    Mul mul;  // Creating an object of type Mul
    int result = mul(3, 4);  // Using mul object like a function, result = 12
    std::cout << "Result: " << result << std::endl;
    return 0;
}
```
In this code:

1.  **Overloading `operator()`**:  
    Inside the `Mul` structure, we have defined the function `operator()(int x, int y)`, which specifies what happens when we "call" an object of `Mul` with two arguments (in this case, it multiplies `x` and `y`).
    
2.  **mul(3, 4)**:  
    Here, `mul` is an object of type `Mul`. Normally, objects don’t have the ability to be "called" like functions. However, because we've overloaded the function call operator `()`, we can call the `mul` object as if it were a function. When we write `mul(3, 4)`, it is equivalent to calling the `operator()` function inside the `Mul` struct, which multiplies the two arguments.

2. **CUDA Kernel Templates for Element-wise and Scalar Operations**

The code defines two key kernel templates:

#### `EwiseOpKernel` (Element-wise operation kernel)
```cpp
// Kernel for element-wise operations
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
// Kernel for scalar operations
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
// Launch function for element-wise operations
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
// Launch function for scalar operations
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

Various element-wise and scalar operations are defined using the above templates. These functions take in `CudaArray` objects, which are abstractions over GPU memory, and perform computations. The functors define the operations:

#### Element-wise Operations:
```cpp
// Element-wise multiplication
void EwiseMul(const CudaArray& a, const CudaArray& b, CudaArray* out) {
  EwiseOp(a, b, out, Mul());
}
```

-   **Purpose**: Element-wise multiplication of `a` and `b`.
-   **Functor**: The `Mul` functor defines the multiplication operation: `x * y`.

Similarly, other element-wise operations are defined:

-   **`EwiseDiv`**: Division of two arrays using the `Div` functor.
-   **`EwiseMaximum`**: Maximum of two arrays using the `Max` functor.
-   **`EwiseEq`**: Equality comparison of two arrays using the `Eq` functor.
-   **`EwiseGe`**: Greater-than-or-equal comparison of two arrays using the `Ge` functor.

#### Scalar Operations:
```cpp
// Scalar multiplication
void ScalarMul(const CudaArray& a, scalar_t val, CudaArray* out) {
  ScalarOp(a, val, out, Mul());
}
```

-   **Purpose**: Scalar multiplication, multiplying each element of `a` by `val`.
-   **Functor**: The `Mul` functor defines the scalar multiplication: `x * y`.

Similarly, other scalar operations are defined:

-   **`ScalarDiv`**: Division by a scalar using the `Div` functor.
-   **`ScalarPower`**: Raise elements of `a` to a scalar power using the `Pow` functor.
-   **`ScalarMaximum`**: Maximum between elements of `a` and a scalar using the `Max` functor.
-   **`ScalarEq`**: Equality comparison between elements of `a` and a scalar using the `Eq` functor.
-   **`ScalarGe`**: Greater-than-or-equal comparison between elements of `a` and a scalar using the `Ge` functor.


4. **Unary Operations**

Some operations like logarithm, exponential, and hyperbolic tangent operate on single arrays, and the same scalar template is reused with fixed functors:

```cpp
// Element-wise log
void EwiseLog(const CudaArray& a, CudaArray* out) {
  ScalarOp(a, 0.0f, out, Log());
}
```

-   **Purpose**: Element-wise logarithm (`log(x)`).
-   **Functor**: The `Log` functor defines the logarithmic function.

Similarly:

-   **`EwiseExp`**: Element-wise exponential function (`exp(x)`) using the `Exp` functor.
-   **`EwiseTanh`**: Element-wise hyperbolic tangent (`tanh(x)`) using the `Tanh` functor.

5. **Pybind11 Integration**

At the bottom of the code, these functions are bound to Python using Pybind11:

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
...
```
-   This exposes the C++/CUDA functions to Python, allowing users to call these CUDA operations in Python code.
-   Example: Calling `ewise_mul` from Python will invoke the `EwiseMul` function implemented in C++/CUDA.

#### Summary

-   **CUDA Kernels**: Perform parallel element-wise and scalar operations on arrays.
-   **Templates**: Ensure code reuse for different operations (multiplication, division, etc.).
-   **Functors**: Define the specific operations performed by the kernels.
-   **Pybind11**: Exposes these functions to Python for easy usage in Python code.

This structure provides a scalable and compact implementation for running mathematical operations on GPUs using CUDA, allowing efficient execution and easy interaction with Python through the Pybind11 bindings.
