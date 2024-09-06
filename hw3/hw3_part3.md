## Part 3: CPU Backend - Elementwise and scalar operations

Implement the following functions in `ndarray_backend_cpu.cc`:

  
* `EwiseMul()`, `ScalarMul()`

* `EwiseDiv()`, `ScalarDiv()`

* `ScalarPower()`

* `EwiseMaximum()`, `ScalarMaximum()`

* `EwiseEq()`, `ScalarEq()`

* `EwiseGe()`, `ScalarGe()`

* `EwiseLog()`

* `EwiseExp()`

* `EwiseTanh()`

You can look at the included

`EwiseAdd()` and `ScalarAdd()` functions (plus the invocations from `NDArray` in order to understand the required format of these functions.

Note that unlike the remaining functions mentioned here, we do not include function stubs for each of these functions. This is because, while you can implement these naively just through implementing each function separately, though this will end up with a lot of duplicated code. You're welcome to use e.g., C++ templates or macros to address this problem (but these would only be exposed internally, not to the external interface).

**Note**: Remember to register functions in the pybind module after finishing your implementations.


**Code Implementation**
```c++
/**
 * Element-wise operation template
 */
template <typename F>
void EwiseOp(const AlignedArray& a, const AlignedArray& b, AlignedArray* out, F op) {
  for (size_t i = 0; i < a.size; i++) {
    out->ptr[i] = op(a.ptr[i], b.ptr[i]);
  }
}

/**
 * Scalar operation template
 */
template <typename F>
void ScalarOp(const AlignedArray& a, scalar_t val, AlignedArray* out, F op) {
  for (size_t i = 0; i < a.size; i++) {
    out->ptr[i] = op(a.ptr[i], val);
  }
}
```
```c++
/**
 * In the code the follows, use the above template to create analogous element-wise
 * and and scalar operators for the following functions.  See the numpy backend for
 * examples of how they should work.
 *   - EwiseMul, ScalarMul
 *   - EwiseDiv, ScalarDiv
 *   - ScalarPower
 *   - EwiseMaximum, ScalarMaximum
 *   - EwiseEq, ScalarEq
 *   - EwiseGe, ScalarGe
 *   - EwiseLog
 *   - EwiseExp
 *   - EwiseTanh
 *
 * If you implement all these naively, there will be a lot of repeated code, so
 * you are welcome (but not required), to use macros or templates to define these
 * functions (however you want to do so, as long as the functions match the proper)
 * signatures above.
 */

// Element-wise multiplication
void EwiseMul(const AlignedArray& a, const AlignedArray& b, AlignedArray* out) {
  EwiseOp(a, b, out, [](scalar_t x, scalar_t y) { return x * y; });
}

// Scalar multiplication
void ScalarMul(const AlignedArray& a, scalar_t val, AlignedArray* out) {
  ScalarOp(a, val, out, [](scalar_t x, scalar_t y) { return x * y; });
}

// Element-wise division
void EwiseDiv(const AlignedArray& a, const AlignedArray& b, AlignedArray* out) {
  EwiseOp(a, b, out, [](scalar_t x, scalar_t y) { return x / y; });
}

// Scalar division
void ScalarDiv(const AlignedArray& a, scalar_t val, AlignedArray* out) {
  ScalarOp(a, val, out, [](scalar_t x, scalar_t y) { return x / y; });
}

// Scalar power
void ScalarPower(const AlignedArray& a, scalar_t val, AlignedArray* out) {
  ScalarOp(a, val, out, [](scalar_t x, scalar_t y) { return std::pow(x, y); });
}

// Element-wise maximum
void EwiseMaximum(const AlignedArray& a, const AlignedArray& b, AlignedArray* out) {
  EwiseOp(a, b, out, [](scalar_t x, scalar_t y) { return std::max(x, y); });
}

// Scalar maximum
void ScalarMaximum(const AlignedArray& a, scalar_t val, AlignedArray* out) {
  ScalarOp(a, val, out, [](scalar_t x, scalar_t y) { return std::max(x, y); });
}

// Element-wise equality
void EwiseEq(const AlignedArray& a, const AlignedArray& b, AlignedArray* out) {
  EwiseOp(a, b, out, [](scalar_t x, scalar_t y) { return x == y ? 1.0f : 0.0f; });
}

// Scalar equality
void ScalarEq(const AlignedArray& a, scalar_t val, AlignedArray* out) {
  ScalarOp(a, val, out, [](scalar_t x, scalar_t y) { return x == y ? 1.0f : 0.0f; });
}

// Element-wise greater or equal
void EwiseGe(const AlignedArray& a, const AlignedArray& b, AlignedArray* out) {
  EwiseOp(a, b, out, [](scalar_t x, scalar_t y) { return x >= y ? 1.0f : 0.0f; });
}

// Scalar greater or equal
void ScalarGe(const AlignedArray& a, scalar_t val, AlignedArray* out) {
  ScalarOp(a, val, out, [](scalar_t x, scalar_t y) { return x >= y ? 1.0f : 0.0f; });
}

// Element-wise logarithm
void EwiseLog(const AlignedArray& a, AlignedArray* out) {
  ScalarOp(a, 0.0f, out, [](scalar_t x, scalar_t) { return std::log(x); });
}

// Element-wise exponential
void EwiseExp(const AlignedArray& a, AlignedArray* out) {
  ScalarOp(a, 0.0f, out, [](scalar_t x, scalar_t) { return std::exp(x); });
}

// Element-wise hyperbolic tangent
void EwiseTanh(const AlignedArray& a, AlignedArray* out) {
  ScalarOp(a, 0.0f, out, [](scalar_t x, scalar_t) { return std::tanh(x); });
}
```
```c++
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
### Explaination of Element-wise Operation Template (`EwiseOp`)

```c++
/**
 * Element-wise operation template
 */
template <typename F>
void EwiseOp(const AlignedArray& a, const AlignedArray& b, AlignedArray* out, F op) {
  for (size_t i = 0; i < a.size; i++) {
    out->ptr[i] = op(a.ptr[i], b.ptr[i]);
  }
}
```
#### Explanation:

-   **Purpose**:
    
    -   This function performs an element-wise operation on two arrays (`a` and `b`), applying the operation `op` to corresponding elements and storing the result in the output array (`out`).
-   **Template Parameter (`F`)**:
    
    -   `F` is a template type that represents a **callable** (like a function, lambda, or function object). This allows you to pass in any function that operates on two values.
-   **Function Arguments**:
    
    -   `const AlignedArray& a`: First input array.
    -   `const AlignedArray& b`: Second input array.
    -   `AlignedArray* out`: The output array where results are stored.
    -   `F op`: A callable (e.g., a lambda) that defines the operation to be performed on each element of `a` and `b`.
-   **Operation**:
    
    -   The function loops over all elements in the array (from `0` to `a.size`).
    -   For each index `i`, it retrieves the corresponding elements from `a` and `b` (using `a.ptr[i]` and `b.ptr[i]`), applies the operation `op` to them, and stores the result in the output array (`out->ptr[i]`).

#### Example of How It's Used:

If you wanted to multiply corresponding elements from two arrays, you could call this function like so:
```c++
EwiseOp(a, b, out, [](scalar_t x, scalar_t y) { return x * y; });
```
This would multiply each element of `a` by the corresponding element of `b` and store the result in `out`.
### Explaination of Scalar Operation Template (`ScalarOp`)

#### Explanation:

-   **Purpose**:
    
    -   This function performs an operation between each element of an array (`a`) and a scalar value (`val`), applying the operation `op` and storing the result in the output array (`out`).
-   **Template Parameter (`F`)**:
    
    -   Similar to `EwiseOp`, `F` is a template type that represents a **callable** (e.g., a function or lambda). This allows you to pass in any function that operates on an element and a scalar.
-   **Function Arguments**:
    
    -   `const AlignedArray& a`: The input array.
    -   `scalar_t val`: The scalar value to apply to each element in the array.
    -   `AlignedArray* out`: The output array where results are stored.
    -   `F op`: A callable (e.g., a lambda) that defines the operation to be performed between each element of `a` and the scalar `val`.
-   **Operation**:
    
    -   The function loops over all elements in the input array `a`.
    -   For each index `i`, it retrieves the corresponding element from `a` (using `a.ptr[i]`), applies the operation `op` between the array element and the scalar `val`, and stores the result in the output array (`out->ptr[i]`).

#### Example of How It's Used:

If you wanted to multiply each element of `a` by a scalar value `val`, you could call this function like so:

```c++
ScalarOp(a, val, out, [](scalar_t x, scalar_t y) { return x * y; });
```
This would multiply each element of `a` by `val` and store the result in `out`.

>-   **`EwiseOp`** is used for element-wise operations between two arrays. It loops over corresponding elements of two arrays (`a` and `b`) and applies the operation `op` to each pair of elements.
>  
> -   **`ScalarOp`** is used for operations between an array and a scalar value. It loops over each element of array `a` and applies the operation `op` between the array element and the scalar `val`.

### Explain of lambda function

A **lambda function** in C++ is a way to create an anonymous, inline function, often for short tasks that don't need a named function. It's especially useful when passing functions as arguments to other functions, as in the case with `EwiseOp`.

### Breakdown of the Lambda Function `[](scalar_t x, scalar_t y) { return x * y; }`

1.  **`[]`**:
    
    -   This is called the **capture clause**. It's used to capture variables from the surrounding scope.
    -   In this case, the lambda doesn't capture any external variables, so it’s empty (`[]`).
2.  **`(scalar_t x, scalar_t y)`**:
    
    -   This is the **parameter list**. It defines the input parameters the lambda will accept.
    -   Here, `x` and `y` are parameters of type `scalar_t`, which represents two values that the lambda will take as input when called.
3.  **`{ return x * y; }`**:
    
    -   This is the **body** of the lambda function, where the actual operation takes place.
    -   The function takes the two inputs (`x` and `y`), multiplies them (`x * y`), and returns the result.
    
### Explaination of `EwiseMul`, `ScalarMul`
```c++
// Element-wise multiplication
void EwiseMul(const AlignedArray& a, const AlignedArray& b, AlignedArray* out) {
  EwiseOp(a, b, out, [](scalar_t x, scalar_t y) { return x * y; });
}

// Scalar multiplication
void ScalarMul(const AlignedArray& a, scalar_t val, AlignedArray* out) {
  ScalarOp(a, val, out, [](scalar_t x, scalar_t y) { return x * y; });
}
```
`EwiseMul`:
-   **Purpose**: This function multiplies two arrays `a` and `b` element by element (element-wise) and stores the result in the `out` array.
-   **How it works**:
    
    -   `EwiseOp`: This is a **template function** (defined elsewhere in the code) that takes two arrays (`a` and `b`), an output array (`out`), and a lambda function that specifies the operation to be applied.
    -   `[](scalar_t x, scalar_t y) { return x * y; }`: This is a **lambda function** that takes two numbers (`x` and `y`), multiplies them, and returns the result. This lambda function is passed to `EwiseOp` as the operation to be applied. 
	    -   `x` is an element from array `a`.
	    -   `y` is an element from array `b`.
	    -   The lambda multiplies `x` and `y`, and `EwiseOp` stores the result in the corresponding position in the output array `out`.

`ScalarMul`:
-   **Purpose**: This function multiplies each element of the array `a` by a scalar value `val` and stores the result in the `out` array.
    
-   **How it works**:
    
    -   `ScalarOp`: This is another **template function** (defined elsewhere), similar to `EwiseOp`, but it handles operations between an array and a scalar value instead of two arrays.
    -   `[](scalar_t x, scalar_t y) { return x * y; }`: Again, this is a lambda function that multiplies `x` (an element from array `a`) by `y` (in this case, the scalar `val`). The lambda function is passed to `ScalarOp`.
    
So, `ScalarMul` multiplies each element in `a` by the scalar `val` using the lambda `x * y`, and stores the result in `out`.

### Explain `EwiseDiv`, `ScalarDiv`

```c++
// Element-wise division
void EwiseDiv(const AlignedArray& a, const AlignedArray& b, AlignedArray* out) {
  EwiseOp(a, b, out, [](scalar_t x, scalar_t y) { return x / y; });
}
```
-   **Purpose**: This function performs division of corresponding elements in arrays `a` and `b` and stores the result in the `out` array.
-   **Explanation**: The function uses `EwiseOp` (element-wise operation template), passing the lambda `[](scalar_t x, scalar_t y) { return x / y; }`, which divides two numbers. For each element `i`, it divides `a.ptr[i]` by `b.ptr[i]` and stores the result in `out->ptr[i]`.
```c++
// Scalar division
void ScalarDiv(const AlignedArray& a, scalar_t val, AlignedArray* out) {
  ScalarOp(a, val, out, [](scalar_t x, scalar_t y) { return x / y; });
}
```
-   **Purpose**: This function divides each element in array `a` by a scalar value `val` and stores the result in `out`.
-   **Explanation**: The function uses `ScalarOp`, which is a scalar operation template. The lambda `[](scalar_t x, scalar_t y) { return x / y; }` divides each element `a.ptr[i]` by the scalar `val`, and the result is stored in `out->ptr[i]`.

### Explain `ScalarPower`, `EwiseMaximum`, `ScalarMaximum`
```c++
// Scalar power
void ScalarPower(const AlignedArray& a, scalar_t val, AlignedArray* out) {
  ScalarOp(a, val, out, [](scalar_t x, scalar_t y) { return std::pow(x, y); });
}
```
-   **Purpose**: This function raises each element of array `a` to the power of the scalar value `val` and stores the result in `out`.
-   **Explanation**: `ScalarOp` is used here with the lambda `[](scalar_t x, scalar_t y) { return std::pow(x, y); }`, which raises `x` (an element from `a`) to the power of `y` (the scalar `val`). The result is stored in `out->ptr[i]`.

`std::pow(x, y)` is a function from the C++ Standard Library (specifically, it is part of the `<cmath>` header). It computes the result of raising the value `x` to the power of `y`.

-   `x`: The base.
-   `y`: The exponent.

In mathematical terms, it calculates $x^y$, which means $x$ raised to the power of $y$.

```c++
// Element-wise maximum
void EwiseMaximum(const AlignedArray& a, const AlignedArray& b, AlignedArray* out) {
  EwiseOp(a, b, out, [](scalar_t x, scalar_t y) { return std::max(x, y); });
}
```
-   **Purpose**: This function finds the maximum of corresponding elements in arrays `a` and `b` and stores the result in `out`.
-   **Explanation**: The function uses `EwiseOp` with the lambda `[](scalar_t x, scalar_t y) { return std::max(x, y); }`. For each pair of elements `a.ptr[i]` and `b.ptr[i]`, it stores the larger of the two in `out->ptr[i]`.

`std::max(x, y)` is a function from the C++ Standard Library (part of the `<algorithm>` or `<cmath>` header), and it returns the larger (maximum) of the two values `x` and `y`.

-   **x**: The first value.
-   **y**: The second value.
-   **Return Value**: The larger of `x` and `y`.

```c++
// Scalar maximum
void ScalarMaximum(const AlignedArray& a, scalar_t val, AlignedArray* out) {
  ScalarOp(a, val, out, [](scalar_t x, scalar_t y) { return std::max(x, y); });
}
```
-   **Purpose**: This function compares each element in array `a` with the scalar `val`, and stores the maximum value in `out`.
-   **Explanation**: `ScalarOp` is used with the lambda `[](scalar_t x, scalar_t y) { return std::max(x, y); }`. For each element `a.ptr[i]`, it compares it with the scalar `val` and stores the maximum in `out->ptr[i]`.

### Explain of `EwiseEq`, `ScalarEq`, `EwiseGe`, `ScalarGe`

```c++
// Element-wise equality
void EwiseEq(const AlignedArray& a, const AlignedArray& b, AlignedArray* out) {
  EwiseOp(a, b, out, [](scalar_t x, scalar_t y) { return x == y ? 1.0f : 0.0f; });
}
```
**Purpose**: This function checks if each element in the array `a` is equal to the corresponding element in array `b`. If they are equal, it stores `1.0f` in the corresponding position of `out`. If they are not equal, it stores `0.0f`.

**Explanation**:

-   `EwiseOp` is used here with the lambda `[](scalar_t x, scalar_t y) { return x == y ? 1.0f : 0.0f; }`.
-   For each pair of elements `x` from array `a` and `y` from array `b`, it checks if `x == y`.
    -   If true, it returns `1.0f`, otherwise it returns `0.0f`.
-   The result is stored in `out->ptr[i]`.

#### Explanation of `1.0f : 0.0f`:

-   `1.0f`: This is a **floating-point value** of `1` (the `f` suffix means it's a `float`).
-   `0.0f`: This is a **floating-point value** of `0`.

The conditional expression is often used to convert a Boolean condition into numeric values (typically `1.0f` for `true` and `0.0f` for `false`).

```c++
// Scalar equality
void ScalarEq(const AlignedArray& a, scalar_t val, AlignedArray* out) {
  ScalarOp(a, val, out, [](scalar_t x, scalar_t y) { return x == y ? 1.0f : 0.0f; });
}
```
**Purpose**: This function checks if each element in the array `a` is equal to the scalar value `val`. If they are equal, it stores `1.0f` in the corresponding position of `out`. If they are not equal, it stores `0.0f`.

**Explanation**:

-   `ScalarOp` is used here with the lambda `[](scalar_t x, scalar_t y) { return x == y ? 1.0f : 0.0f; }`.
-   For each element `x` from array `a`, it compares it with the scalar `val`.
    -   If `x == val`, it returns `1.0f`, otherwise it returns `0.0f`.
-   The result is stored in `out->ptr[i]`.
```c++
// Element-wise greater or equal
void EwiseGe(const AlignedArray& a, const AlignedArray& b, AlignedArray* out) {
  EwiseOp(a, b, out, [](scalar_t x, scalar_t y) { return x >= y ? 1.0f : 0.0f; });
}
```
**Purpose**: This function checks if each element in the array `a` is greater than or equal to the corresponding element in array `b`. If it is, it stores `1.0f` in the corresponding position of `out`. If not, it stores `0.0f`.

**Explanation**:

-   `EwiseOp` is used with the lambda `[](scalar_t x, scalar_t y) { return x >= y ? 1.0f : 0.0f; }`.
-   For each pair of elements `x` from array `a` and `y` from array `b`, it checks if `x >= y`.
    -   If true, it returns `1.0f`, otherwise it returns `0.0f`.
-   The result is stored in `out->ptr[i]`.
```c++
// Scalar greater or equal
void ScalarGe(const AlignedArray& a, scalar_t val, AlignedArray* out) {
  ScalarOp(a, val, out, [](scalar_t x, scalar_t y) { return x >= y ? 1.0f : 0.0f; });
}
```
**Purpose**: This function checks if each element in the array `a` is greater than or equal to the scalar value `val`. If the condition holds true for an element, it stores `1.0f` in the corresponding position of `out`. If not, it stores `0.0f`.

**Explanation**:

-   `ScalarOp` is used with the lambda `[](scalar_t x, scalar_t y) { return x >= y ? 1.0f : 0.0f; }`.
-   For each element `x` from array `a`, it checks if `x >= val` (where `val` is the scalar value `y`).
    -   If `x >= val` is true, the lambda returns `1.0f`.
    -   If `x >= val` is false, the lambda returns `0.0f`.
-   The result of each comparison (either `1.0f` or `0.0f`) is stored in `out->ptr[i]`.

#### Example:

Given an array `a = [3.5, 4.2, 2.8]` and a scalar value `val = 4.0`:

-   For `3.5 >= 4.0`: The comparison is false, so `0.0f` is stored.
-   For `4.2 >= 4.0`: The comparison is true, so `1.0f` is stored.
-   For `2.8 >= 4.0`: The comparison is false, so `0.0f` is stored.

The output array `out` will be `[0.0f, 1.0f, 0.0f]`.

### Explain of `EwiseLog`, `EwiseExp`, `EwiseTanh`
```c++
// Element-wise logarithm
void EwiseLog(const AlignedArray& a, AlignedArray* out) {
  ScalarOp(a, 0.0f, out, [](scalar_t x, scalar_t) { return std::log(x); });
}
```
**Purpose**: This function computes the natural logarithm of each element in the array `a` and stores the result in the array `out`.

**Explanation**: The function uses `ScalarOp` (scalar operation template) and passes a lambda function `[](scalar_t x, scalar_t) { return std::log(x); }`. Here, the second argument to the lambda is not used, but the lambda computes the logarithm of each element `x` in array `a`. For each element `i`, it calculates `std::log(a.ptr[i])` and stores the result in `out->ptr[i]`.

In the function `EwiseLog`, the second argument (`0.0f`) passed to `ScalarOp` is a placeholder because the lambda function `[](scalar_t x, scalar_t) { return std::log(x); }` only uses the first argument (`x`) and ignores the second argument (`scalar_t`).

`std::log(x)` is a function from the C++ Standard Library (part of the `<cmath>` header) that computes the natural logarithm (base `e`) of the value `x`.

-   **x**: The value for which the natural logarithm is to be computed.
-   **Return Value**: The natural logarithm of `x`, i.e., the exponent to which `e` (Euler's number, approximately `2.71828`) must be raised to produce the value `x`.

```c++
// Element-wise exponential
void EwiseExp(const AlignedArray& a, AlignedArray* out) {
  ScalarOp(a, 0.0f, out, [](scalar_t x, scalar_t) { return std::exp(x); });
}
```
**Purpose**: This function computes the exponential (e^x) of each element in the array `a` and stores the result in the array `out`.

**Explanation**: The function uses `ScalarOp` (scalar operation template) and passes a lambda function `[](scalar_t x, scalar_t) { return std::exp(x); }`. Similar to the logarithm function, the second argument is not used, and the lambda computes the exponential of each element `x`. For each element `i`, it calculates `std::exp(a.ptr[i])` and stores the result in `out->ptr[i]`.

`std::exp(x)` is a function from the C++ Standard Library (part of the `<cmath>` header) that computes the exponential of `x`, i.e., it returns the value of $e^x$, where $e$ is Euler's number, approximately equal to `2.71828`.

-   **x**: The exponent to which the base eee is raised.
-   **Return Value**: The value of $e^x$, which is the result of raising Euler's number eee to the power of `x`.
- 
```c++
// Element-wise hyperbolic tangent
void EwiseTanh(const AlignedArray& a, AlignedArray* out) {
  ScalarOp(a, 0.0f, out, [](scalar_t x, scalar_t) { return std::tanh(x); });
}
```
**Purpose**: This function computes the hyperbolic tangent (tanh) of each element in the array `a` and stores the result in the array `out`.

**Explanation**: The function uses `ScalarOp` (scalar operation template) and passes a lambda function `[](scalar_t x, scalar_t) { return std::tanh(x); }`. As with the previous functions, the second argument is unused, and the lambda calculates the hyperbolic tangent for each element `x`. For each element `i`, it computes `std::tanh(a.ptr[i])` and stores the result in `out->ptr[i]`.

`std::tanh(x)` is a function from the C++ Standard Library (part of the `<cmath>` header) that computes the hyperbolic tangent of `x`. The hyperbolic tangent is a mathematical function that is similar to the ordinary tangent function but is based on hyperbolic geometry.

-   **x**: The value (in radians) for which the hyperbolic tangent is to be computed.
-   **Return Value**: The hyperbolic tangent of `x`, which is a value between `-1` and `1`.

### Explain of `m.def()`
`m.def()` is a function in **Pybind11**, a popular C++ library used to create Python bindings for C++ code. This function is used to **expose C++ functions to Python**, allowing them to be called as if they were regular Python functions.
```c++
m.def("python_function_name", cplusplus_function_pointer);
```
#### Components:

-   **`m`**: This refers to the module object that you define in the `PYBIND11_MODULE` block. It represents the Python module you're creating, and you're adding functions to this module.
    
-   **`def()`**: This is a method used to register a function with the Python module. It takes a few parameters:
    
    -   **`"python_function_name"`**: The name you want to use when calling the function from Python.
    -   **`cplusplus_function_pointer`**: The actual C++ function you are exposing to Python.

```c++
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

The code uses `m.def()` with Pybind11 to expose C++ functions to Python, enabling Python to call these functions for operations like element-wise multiplication, division, comparison, logarithms, exponentiation, and more on arrays or between arrays and scalars.

> The phrase **"register functions in the pybind module"** refers to the process of making the C++ functions available in Python by binding them to Python using `m.def()`. When you "register" a function, you're essentially telling Pybind11 to expose the C++ function so that Python can call it.

#### Example
```c++
   m.def("ewise_div", EwiseDiv);
```
In Python, you would call it as `ewise_div()`. Here's how Python would call `EwiseDiv`:
```python
import your_module_name  # Replace 'your_module_name' with the actual name of your module

# Assuming you have numpy arrays or other arrays of the appropriate type
a = your_module_name.Array()  # Create or load an array
b = your_module_name.Array()  # Another array

# Create an output array to store the result
out = your_module_name.Array()

# Perform element-wise division
your_module_name.ewise_div(a, b, out)

# The result of the division will be stored in 'out'
```
