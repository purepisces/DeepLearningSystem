## Part 4: CPU Backend - Reductions

Implement the following functions in `ndarray_backend_cpu.cc`:

* `ReduceMax()`

* `ReduceSum()`

In general, the reduction functions `.max()` and `.sum()` in NDArray take the max or sum across a specified axis specified by the `axis` argument (or across the entire array when `axis=None`); note that we don't support axis being a set of axes, though this wouldn't be too hard to add if you desired (but it's not in the interface you should implement for the homework).


Because summing over individual axes can be a bit tricky, even for compact arrays, these functions (in Python) in Python simplify things by permuting the last axis to the be the one reduced over (this is what the `reduce_view_out()` function in NDArray does), then compacting the array. So for your `ReduceMax()` and `ReduceSum()` functions you implement in C++, you can assume that both the input and output arrays are contiguous in memory, and you want to just reduce over contiguous elements of size `reduce_size` as passed to the C++ functions.

**Code Implementation**
```c++
void ReduceMax(const AlignedArray& a, AlignedArray* out, size_t reduce_size) {
  /**
  * Reduce by taking maximum over `reduce_size` contiguous blocks.
  *
  * Args:
  *   a: compact array of size a.size = out.size * reduce_size to reduce over
  *   out: compact array to write into
  *   reduce_size: size of the dimension to reduce over
  */

  /// BEGIN SOLUTION
  for (size_t i = 0; i < out->size; i++) {
    scalar_t max_val = a.ptr[i * reduce_size];  // Initialize max to the first element in the block
    for (size_t j = 1; j < reduce_size; j++) {
      max_val = std::max(max_val, a.ptr[i * reduce_size + j]);
    }
    out->ptr[i] = max_val;
  }
  /// END SOLUTION
}

void ReduceSum(const AlignedArray& a, AlignedArray* out, size_t reduce_size) {
  /**
  * Reduce by taking sum over `reduce_size` contiguous blocks.
  *
  * Args:
  *   a: compact array of size a.size = out.size * reduce_size to reduce over
  *   out: compact array to write into
  *   reduce_size: size of the dimension to reduce over
  */

  /// BEGIN SOLUTION
  for (size_t i = 0; i < out->size; i++) {
    scalar_t sum_val = 0.0;  // Initialize sum to 0
    for (size_t j = 0; j < reduce_size; j++) {
      sum_val += a.ptr[i * reduce_size + j];
    }
    out->ptr[i] = sum_val;
  }
  /// END SOLUTION
}
```
### Explain of  `ReduceMax`

-   **Loop Over Blocks**: We loop over each contiguous block in the input array `a`. The number of blocks corresponds to the size of the output array (`out->size`).
-   **Initial Maximum**: For each block of size `reduce_size`, we initialize `max_val` with the first element of the block.
-   **Find Maximum**: We loop over the remaining elements in the block and update `max_val` if a larger value is found.
-   **Store Result**: After processing the block, the maximum value is stored in the corresponding position in the output array `out`.

#### Example:

Given the input array `a = [2, 4, 1, 5, 3, 6]`, `reduce_size = 2`, and output array `out = [ ]`:

-   For the first block `[2, 4]`, the maximum is `4`.
-   For the second block `[1, 5]`, the maximum is `5`.
-   For the third block `[3, 6]`, the maximum is `6`.

The result in `out` will be `[4, 5, 6]`.

### Explain of  `reducesum`
#### **Explanation**:

-   **Loop Over Blocks**: Similar to `ReduceMax()`, this function loops over each contiguous block in the input array `a`. Each block has `reduce_size` elements.
-   **Initial Sum**: For each block, we initialize the sum `sum_val` to `0.0`.
-   **Calculate Sum**: We loop over each element in the block and accumulate the sum by adding each value to `sum_val`.
-   **Store Result**: After summing the elements in the block, we store the sum in the output array `out`.

#### Example:

Given the input array `a = [2, 4, 1, 5, 3, 6]`, `reduce_size = 2`, and output array `out = [ ]`:

-   For the first block `[2, 4]`, the sum is `6`.
-   For the second block `[1, 5]`, the sum is `6`.
-   For the third block `[3, 6]`, the sum is `9`.

The result in `out` will be `[6, 6, 9]`.


### General Process:

1.  **Block-Based Operation**: Both functions perform operations on contiguous blocks of elements in the input array `a`. The size of each block is determined by `reduce_size`, and there are as many blocks as there are elements in the output array `out`.
    
2.  **Reduction**:
    
    -   In `ReduceMax()`, the operation is finding the maximum value within each block.
    -   In `ReduceSum()`, the operation is summing all the values within each block.
3.  **Memory Layout**: The code assumes that both the input array `a` and output array `out` are contiguous in memory, which simplifies the implementation by using direct index calculations (`i * reduce_size + j`) to access the elements of `a`.
