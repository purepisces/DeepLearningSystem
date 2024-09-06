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
### Explaination of 
