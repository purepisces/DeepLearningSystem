## Part 5: CPU Backend - Matrix multiplication

Implement the following functions in `ndarray_backend_cpu.cc`:

* `Matmul()`

* `MatmulTiled()`

* `AlignedDot()`

The first implementation, `Matmul()` can use the naive three-nested-for-loops algorithm for matrix multiplication. However, the `MatmulTiled()` performs the same matrix multiplication on memory laid out in tiled form, i.e., as a contiguous 4D array

```c++

float[M/TILE][N/TILE][TILE][TILE];

```

Note that the Python `__matmul__` code already does the conversion to tiled form when all sizes of the matrix multiplication are divisible by `TILE`, so your code just needs to implement the multiplication in this form. In order to make the methods efficient, you will want to make use of (after you implement it), the `AlignedDot()` function, which will enable the compiler to efficiently make use of vector operations and proper caching. The output matrix will also be in the tiled form above, and the Python backend will take care of the conversion to a normal 2D array.

Note that in order to get the most speedup possible from you tiled version, you may want to use the clang compiler with colab instead of gcc. To do this, run the following command before building your code.

**Code Implementation**
```c++
void Matmul(const AlignedArray& a, const AlignedArray& b, AlignedArray* out, uint32_t m, uint32_t n,
            uint32_t p) {
  /**
  * Multiply two (compact) matrices into an output (also compact) matrix.  For this implementation
  * you can use the "naive" three-loop algorithm.
  *
  * Args:
  *   a: compact 2D array of size m x n
  *   b: compact 2D array of size n x p
  *   out: compact 2D array of size m x p to write the output to
  *   m: rows of a / out
  *   n: columns of a / rows of b
  *   p: columns of b / out
  */

  /// BEGIN SOLUTION
  // Initialize the output matrix to zero
  for (size_t i = 0; i < m * p; i++) {
    out->ptr[i] = 0;
  }

  // Naive three-loop matrix multiplication
  for (uint32_t i = 0; i < m; i++) {
    for (uint32_t j = 0; j < p; j++) {
      for (uint32_t k = 0; k < n; k++) {
        out->ptr[i * p + j] += a.ptr[i * n + k] * b.ptr[k * p + j];
      }
    }
  }
  /// END SOLUTION
}

inline void AlignedDot(const float* __restrict__ a,
                      const float* __restrict__ b,
                      float* __restrict__ out) {

  /**
  * Multiply together two TILE x TILE matrices, and _add _the result to out (it is important to add
  * the result to the existing out, which you should not set to zero beforehand).  We are including
  * the compiler flags here that enable the compile to properly use vector operators to implement
  * this function.  Specifically, the __restrict__ keyword indicates to the compile that a, b, and
  * out don't have any overlapping memory (which is necessary in order for vector operations to be
  * equivalent to their non-vectorized counterparts (imagine what could happen otherwise if a, b,
  * and out had overlapping memory).  Similarly the __builtin_assume_aligned keyword tells the
  * compiler that the input array will be aligned to the appropriate blocks in memory, which also
  * helps the compiler vectorize the code.
  *
  * Args:
  *   a: compact 2D array of size TILE x TILE
  *   b: compact 2D array of size TILE x TILE
  *   out: compact 2D array of size TILE x TILE to write to
  */

  a = (const float*)__builtin_assume_aligned(a, TILE * ELEM_SIZE);
  b = (const float*)__builtin_assume_aligned(b, TILE * ELEM_SIZE);
  out = (float*)__builtin_assume_aligned(out, TILE * ELEM_SIZE);

  /// BEGIN SOLUTION
  // Perform matrix multiplication and accumulate the result in out
  for (size_t i = 0; i < TILE; i++) {
    for (size_t j = 0; j < TILE; j++) {
      for (size_t k = 0; k < TILE; k++) {
        out[i * TILE + j] += a[i * TILE + k] * b[k * TILE + j]; // Accumulate the result in out
      }
    }
  }
  /// END SOLUTION
}

void MatmulTiled(const AlignedArray& a, const AlignedArray& b, AlignedArray* out, uint32_t m,
                 uint32_t n, uint32_t p) {
  /**
  * Matrix multiplication on tiled representations of array.  In this setting, a, b, and out
  * are all *4D* compact arrays of the appropriate size, e.g. a is an array of size
  *   a[m/TILE][n/TILE][TILE][TILE]
  * You should do the multiplication tile-by-tile to improve performance of the array (i.e., this
  * function should call `AlignedDot()` implemented above).
  *
  * Note that this function will only be called when m, n, p are all multiples of TILE, so you can
  * assume that this division happens without any remainder.
  *
  * Args:
  *   a: compact 4D array of size m/TILE x n/TILE x TILE x TILE
  *   b: compact 4D array of size n/TILE x p/TILE x TILE x TILE
  *   out: compact 4D array of size m/TILE x p/TILE x TILE x TILE to write to
  *   m: rows of a / out
  *   n: columns of a / rows of b
  *   p: columns of b / out
  *
  */
  /// BEGIN SOLUTION
  size_t mt = m / TILE;  // Number of tiles in the row dimension
  size_t nt = n / TILE;  // Number of tiles in the shared dimension
  size_t pt = p / TILE;  // Number of tiles in the column dimension

  // Initialize the output matrix to zero
  for (size_t i = 0; i < m * p; i++) {
    out->ptr[i] = 0;
  }

  // Iterate over tiles
  for (size_t i = 0; i < mt; i++) {
    for (size_t j = 0; j < pt; j++) {
      for (size_t k = 0; k < nt; k++) {
        // Get pointers to the tile blocks in matrices a, b, and out
        const float* a_tile = &a.ptr[(i * nt + k) * TILE * TILE];
        const float* b_tile = &b.ptr[(k * pt + j) * TILE * TILE];
        float* out_tile = &out->ptr[(i * pt + j) * TILE * TILE];

        // Perform multiplication on the TILE x TILE blocks
        AlignedDot(a_tile, b_tile, out_tile);
      }
    }
  }
  /// END SOLUTION
}
```

### Explaination of `matmul`
```c++
void Matmul(const AlignedArray& a, const AlignedArray& b, AlignedArray* out, uint32_t m, uint32_t n,
            uint32_t p) {
  /**
  * Multiply two (compact) matrices into an output (also compact) matrix.  For this implementation
  * you can use the "naive" three-loop algorithm.
  *
  * Args:
  *   a: compact 2D array of size m x n
  *   b: compact 2D array of size n x p
  *   out: compact 2D array of size m x p to write the output to
  *   m: rows of a / out
  *   n: columns of a / rows of b
  *   p: columns of b / out
  */

  /// BEGIN SOLUTION
  // Initialize the output matrix to zero
  for (size_t i = 0; i < m * p; i++) {
    out->ptr[i] = 0;
  }

  // Naive three-loop matrix multiplication
  for (uint32_t i = 0; i < m; i++) {
    for (uint32_t j = 0; j < p; j++) {
      for (uint32_t k = 0; k < n; k++) {
        out->ptr[i * p + j] += a.ptr[i * n + k] * b.ptr[k * p + j];
      }
    }
  }
  /// END SOLUTION
}
```
This function performs matrix multiplication on two 2D arrays `a` and `b` and stores the result in the output array `out`. It uses a "naive" three-loop algorithm to multiply the matrices.

#### Args:

-   `a`: Compact 2D array of size `m x n`.
-   `b`: Compact 2D array of size `n x p`.
-   `out`: Compact 2D array of size `m x p` where the result is written.
-   `m`: The number of rows in `a` and `out`.
-   `n`: The number of columns in `a` and rows in `b`.
-   `p`: The number of columns in `b` and `out`.

#### Matrix Multiplication Concept

Given two matrices `A` (of size `m x n`) and `B` (of size `n x p`), the product matrix `C = A * B` (of size `m x p`) is computed as:

$$C[i][j] = \sum_{k=0}^{n-1} A[i][k] \times B[k][j]$$

This means that the value at each position `C[i][j]` is the dot product of the `i`-th row of matrix `A` with the `j`-th column of matrix `B`.

### Explan of `__builtin_assume_aligned
`__builtin_assume_aligned` is a built-in function in GCC and Clang compilers that allows the programmer to inform the compiler that a given pointer is aligned to a specific boundary. It helps the compiler generate more efficient code, particularly for vectorized operations (such as SIMD), where memory alignment plays a crucial role in performance optimization.

**Syntax**
```c++
void* __builtin_assume_aligned(const void* ptr, size_t alignment);
```
-   **ptr**: The pointer to the memory that you are telling the compiler is aligned.
-   **alignment**: The alignment boundary in bytes. It specifies how the data should be aligned in memory (e.g., 16 bytes for 128-bit alignment, 32 bytes for 256-bit alignment, etc.).

The purpose of this function is to let the compiler assume that the memory pointer is aligned to the specified boundary, which enables it to generate optimized, aligned memory access instructions (especially useful in vectorized operations).
```c++
a = (const float*)__builtin_assume_aligned(a, TILE * ELEM_SIZE);
b = (const float*)__builtin_assume_aligned(b, TILE * ELEM_SIZE);
out = (float*)__builtin_assume_aligned(out, TILE * ELEM_SIZE);
```
-   **Purpose**: These lines tell the compiler that `a`, `b`, and `out` are aligned to a memory boundary defined by `TILE * ELEM_SIZE`. This alignment enables the compiler to use optimized vectorized instructions for faster memory access.
-   **Example**: If `TILE` is 4 and `ELEM_SIZE` is 4 bytes (for `float`), the arrays are aligned to 16-byte boundaries, ensuring more efficient memory access.

### Explan of `inline`

-   **Purpose**: The `inline` keyword in C++ is a suggestion to the compiler that this function should be expanded **in-place** wherever it is called, rather than using the typical function call mechanism. This means that the function code is inserted directly into the calling code to avoid the overhead of function calls (like pushing arguments to the stack, jumping to the function location, and then returning).

-   **When to use**:
    
    -   For small, frequently called functions, inlining can improve performance by eliminating the function call overhead.
    -   The compiler can choose to ignore the `inline` request if the function is too large or complex, so it is not a strict directive.
-   **Example**:
```c++
inline int add(int a, int b) {
    return a + b;
}
```
When you call `add(3, 5)`, instead of performing a function call, the compiled code will replace it with `3 + 5`.

- **Caution**:

  -   Excessive use of `inline` on large functions can increase code size, leading to issues with instruction cache, possibly reducing performance in some cases (this is called **code bloat**).

### Explain of `__restrict__`

-   **Purpose**: `__restrict__` (or `restrict` in standard C) is a hint to the compiler that tells it that the pointers do not **alias** or overlap in memory. This means that the memory regions pointed to by the pointers do not overlap, and the compiler can make aggressive optimizations based on this assumption.
    
-   **Benefits**: If the compiler knows that two pointers point to different areas of memory, it can perform optimizations like reordering instructions and vectorizing code, leading to faster execution. Without `__restrict__`, the compiler has to assume that pointers might point to the same memory region, which forces it to take extra precautions (like additional memory loads).

### Explaination of `AlignedDot`

```c++
inline void AlignedDot(const float* __restrict__ a,
                      const float* __restrict__ b,
                      float* __restrict__ out) {

  /**
  * Multiply together two TILE x TILE matrices, and _add _the result to out (it is important to add
  * the result to the existing out, which you should not set to zero beforehand).  We are including
  * the compiler flags here that enable the compile to properly use vector operators to implement
  * this function.  Specifically, the __restrict__ keyword indicates to the compile that a, b, and
  * out don't have any overlapping memory (which is necessary in order for vector operations to be
  * equivalent to their non-vectorized counterparts (imagine what could happen otherwise if a, b,
  * and out had overlapping memory).  Similarly the __builtin_assume_aligned keyword tells the
  * compiler that the input array will be aligned to the appropriate blocks in memory, which also
  * helps the compiler vectorize the code.
  *
  * Args:
  *   a: compact 2D array of size TILE x TILE
  *   b: compact 2D array of size TILE x TILE
  *   out: compact 2D array of size TILE x TILE to write to
  */

  a = (const float*)__builtin_assume_aligned(a, TILE * ELEM_SIZE);
  b = (const float*)__builtin_assume_aligned(b, TILE * ELEM_SIZE);
  out = (float*)__builtin_assume_aligned(out, TILE * ELEM_SIZE);

  /// BEGIN SOLUTION
  // Perform matrix multiplication and accumulate the result in out
  for (size_t i = 0; i < TILE; i++) {
    for (size_t j = 0; j < TILE; j++) {
      for (size_t k = 0; k < TILE; k++) {
        out[i * TILE + j] += a[i * TILE + k] * b[k * TILE + j]; // Accumulate the result in out
      }
    }
  }
  /// END SOLUTION
}
```
The `AlignedDot` function performs matrix multiplication for two **TILE x TILE** matrices and **adds** the result to the existing output matrix, without initializing the output matrix to zero. This function leverages certain compiler flags to optimize the operation for vectorization and memory alignment.

**1. Function Signature and `__restrict__` keyword**:
```c++
inline void AlignedDot(const float* __restrict__ a,
                       const float* __restrict__ b,
                       float* __restrict__ out)
```
-   **`inline`**: Suggests to the compiler that the function can be expanded inline where it is used, improving performance by reducing function call overhead.
-   **`const float* __restrict__ a`**: Indicates that the pointer `a` points to a **read-only** memory region of type `float`. The `__restrict__` keyword tells the compiler that the pointers `a`, `b`, and `out` do not overlap in memory, allowing more aggressive optimizations.
-   **`float* __restrict__ out`**: The result of the matrix multiplication is **added** to this output matrix. It is crucial that `out` is not set to zero before the function, as it accumulates results from multiple calls to `AlignedDot`.

**2. Memory Alignment with `__builtin_assume_aligned`**:

```c++
a = (const float*)__builtin_assume_aligned(a, TILE * ELEM_SIZE);
b = (const float*)__builtin_assume_aligned(b, TILE * ELEM_SIZE);
out = (float*)__builtin_assume_aligned(out, TILE * ELEM_SIZE);
```
-   **`__builtin_assume_aligned`** tells the compiler to assume that the memory pointers `a`, `b`, and `out` are aligned in memory to a specific block size (`TILE * ELEM_SIZE`). This allows the compiler to generate more efficient vectorized instructions.

**3. Matrix Multiplication and Accumulation**:
```c++
for (size_t i = 0; i < TILE; i++) {
  for (size_t j = 0; j < TILE; j++) {
    for (size_t k = 0; k < TILE; k++) {
      out[i * TILE + j] += a[i * TILE + k] * b[k * TILE + j]; // Accumulate the result in out
    }
  }
}
```
This part of the code performs a **three-nested-loop matrix multiplication** on the `TILE x TILE` sub-matrices:

-   The **outer loop (`i`)** iterates over the rows of the matrix `a`.
    
-   The **middle loop (`j`)** iterates over the columns of the matrix `b`.
    
-   The **inner loop (`k`)** iterates over the columns of `a` and the rows of `b`, performing the **dot product** of a row from `a` and a column from `b`.

### Explain of `MatmulTiled`

```c++
void MatmulTiled(const AlignedArray& a, const AlignedArray& b, AlignedArray* out, uint32_t m,
                 uint32_t n, uint32_t p) {
  /**
  * Matrix multiplication on tiled representations of array.  In this setting, a, b, and out
  * are all *4D* compact arrays of the appropriate size, e.g. a is an array of size
  *   a[m/TILE][n/TILE][TILE][TILE]
  * You should do the multiplication tile-by-tile to improve performance of the array (i.e., this
  * function should call `AlignedDot()` implemented above).
  *
  * Note that this function will only be called when m, n, p are all multiples of TILE, so you can
  * assume that this division happens without any remainder.
  *
  * Args:
  *   a: compact 4D array of size m/TILE x n/TILE x TILE x TILE
  *   b: compact 4D array of size n/TILE x p/TILE x TILE x TILE
  *   out: compact 4D array of size m/TILE x p/TILE x TILE x TILE to write to
  *   m: rows of a / out
  *   n: columns of a / rows of b
  *   p: columns of b / out
  *
  */
  /// BEGIN SOLUTION
  size_t mt = m / TILE;  // Number of tiles in the row dimension
  size_t nt = n / TILE;  // Number of tiles in the shared dimension
  size_t pt = p / TILE;  // Number of tiles in the column dimension

  // Initialize the output matrix to zero
  for (size_t i = 0; i < m * p; i++) {
    out->ptr[i] = 0;
  }

  // Iterate over tiles
  for (size_t i = 0; i < mt; i++) {
    for (size_t j = 0; j < pt; j++) {
      for (size_t k = 0; k < nt; k++) {
        // Get pointers to the tile blocks in matrices a, b, and out
        const float* a_tile = &a.ptr[(i * nt + k) * TILE * TILE];
        const float* b_tile = &b.ptr[(k * pt + j) * TILE * TILE];
        float* out_tile = &out->ptr[(i * pt + j) * TILE * TILE];

        // Perform multiplication on the TILE x TILE blocks
        AlignedDot(a_tile, b_tile, out_tile);
      }
    }
  }
  /// END SOLUTION
}
```
The function `MatmulTiled` performs matrix multiplication on **tiled representations** of matrices. The inputs `a`, `b`, and `out` are all represented as 4D arrays that correspond to tiled versions of 2D matrices. Each tile is a `TILE x TILE` matrix, and the multiplication is done tile-by-tile to maximize memory performance.

#### Key Points:

-   **Tiling**: The matrices are broken into smaller tiles (`TILE x TILE` blocks), which helps improve cache performance during matrix multiplication.
-   **Matrix Dimensions**:
    -   `a`: matrix of size `m x n`
    -   `b`: matrix of size `n x p`
    -   `out`: result matrix of size `m x p`

However, these matrices are stored as compact 4D arrays, where each `TILE x TILE` block is stored contiguously in memory.

#### Explanation of the Code:

1.  **Arguments**:
    
    -   `a`, `b`: Input 4D arrays (matrices) to be multiplied.
    -   `out`: Output 4D array to store the result of the multiplication.
    -   `m`, `n`, `p`: Dimensions of the matrices. These are multiple of `TILE` (because the matrices are tiled).

2. **Setup the Tile Dimensions**:

```c++
size_t mt = m / TILE;  // Number of tiles in the row dimension
size_t nt = n / TILE;  // Number of tiles in the shared dimension
size_t pt = p / TILE;  // Number of tiles in the column dimension
```
Here, the code computes the number of tiles (`mt`, `nt`, `pt`) along the row, shared, and column dimensions. Each matrix is divided into tiles, and these variables store the number of tiles in each direction.

3. **Initialize the Output Matrix**:
```c++
for (size_t i = 0; i < m * p; i++) {
  out->ptr[i] = 0;
}
```
-   This loop initializes the output matrix `out` to all zeros. The matrix `out` is treated as a 1D flattened array of size `m * p` (since the 4D array is flattened into a 1D array in memory).
    
  4.  **Iterate Over Tiles**: The function uses three nested loops to iterate over the tiles in the `out` matrix.

```c++
for (size_t i = 0; i < mt; i++) {  // Iterate over rows of tiles
  for (size_t j = 0; j < pt; j++) {  // Iterate over columns of tiles
    for (size_t k = 0; k < nt; k++) {  // Iterate over shared dimension tiles
```
-   -   `i`: index of the row tile in matrix `a`.
    -   `j`: index of the column tile in matrix `b`.
    -   `k`: index of the shared dimension tile between `a` and `b`.

5. **Accessing Tile Blocks**: In each iteration, the function retrieves the memory addresses of the specific `TILE x TILE` blocks (tiles) from matrices `a`, `b`, and `out`.
```c++
const float* a_tile = &a.ptr[(i * nt + k) * TILE * TILE];
const float* b_tile = &b.ptr[(k * pt + j) * TILE * TILE];
float* out_tile = &out->ptr[(i * pt + j) * TILE * TILE];
```
-   Here, the memory address of each tile block is calculated using pointer arithmetic:
    
    -   `a_tile`: The tile in matrix `a` at position `[i, k]` (row tile `i` and column tile `k`).
    -   `b_tile`: The tile in matrix `b` at position `[k, j]` (row tile `k` and column tile `j`).
    -   `out_tile`: The tile in matrix `out` at position `[i, j]` (row tile `i` and column tile `j`).
    
    These are computed based on the formula `(i * nt + k) * TILE * TILE`, which calculates the linear offset for each tile within the flattened 4D array.
    
6. **Performing Matrix Multiplication on Each Tile**: Once the tile blocks are identified, the `AlignedDot` function is called to perform matrix multiplication on the two `TILE x TILE` tiles.

```c++
AlignedDot(a_tile, b_tile, out_tile);
```
The `AlignedDot` function performs matrix multiplication on the two `TILE x TILE` blocks from `a` and `b`, and accumulates the result into the corresponding `TILE x TILE` block in `out`.
