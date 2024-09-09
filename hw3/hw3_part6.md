## Part 6: CUDA Backend - Compact and setitem

Implement the following functions in `ndarray_backend_cuda.cu`:

* `Compact()`

* `EwiseSetitem()`

* `ScalarSetitem()`

For this portion, you'll implement the compact and setitem calls in the CUDA backend. This is fairly similar to the C++ version, however, depending on how you implemented that function, there could also be some substantial differences. We specifically want to highlight a few differences between the C++ and the CUDA implementations, however.

First, as with the example functions implemented in the CUDA backend code, for all the functions above you will actually want to implement two functions: the basic functions listed above that you will call from Python, and the corresponding CUDA kernels that will actually perform the computation. For the most part, we only provide the prototype for the "base" function in the `ndarray_backend_cuda.cu` file, and you will need to define and implement the kernel function yourself. However, to see how these work, for the `Compact()` call we are providing you with the _complete_  `Compact()` call, and the function prototype for the `CompactKernel()` call.

  

One thing you may notice is the seemingly odd use of a `CudaVec` struct, which is a struct used to pass shape/stride parameters. In the C++ version we used the STL `std::vector` variables to store these inputs (and the same is done in the base `Compact()` call, but CUDA kernels cannot operation on STL vectors, so something else is needed). Furthermore, although we _could_ convert the vectors to normal CUDA arrays, this would be rather cumbersome, as we would need to call `cudaMalloc()`, pass the parameters as integer pointers, then free them after the calls. Of course such memory management is needed for the actual underlying data in the array, but it seems like overkill to do it for just passing a variable-sized small tuple of shape/stride values. The solution is to create a struct that has a "maximize" size for the number of dimensions an array can have, and then just store the actual shape/stride data in the first entries of these fields. This is all done by the included `CudaVec` struct and `VecToCuda()` function, and you can just use these as provided for all the CUDA kernels that require passing shape/strides to the kernel itself.

  

The other (more conceptual) big difference between the C++ and CUDA implementations of `Compact()` is that in C++ you will typically loop over the elements of the non-compact array sequentially, which allows you to perform some optimizations with respect to computing the corresponding indices between the compact and non-compact arrays. In CUDA, you cannot do this, and will need to implement code that can directly map from an index in the compact array to one in the strided array.

  

As before, we recommend you implement your code in such as way that it can easily be re-used between the `Compact()`, and `Setitem()` calls. As a short note, remember that if you want to call a (separate, non-kernel) function from kernel code, you need to define it as a `__device__` function.

**Code Implementation**
```c++
__device__ size_t index_to_offset_cuda(const CudaVec& strides, const CudaVec& shape, size_t index, size_t base_offset) {
    size_t offset = base_offset;  // Start with the base offset

    // Loop over the dimensions in reverse
    for (int dim = shape.size - 1; dim >= 0; dim--) {
        offset += (index % shape.data[dim]) * strides.data[dim];  //Calculate offset using strides, where (index % shape.data[dim]) gives the index within this dimension.
        index /= shape.data[dim];  // Move to the next dimension by dividing by the current shape
    }

    return offset;  // Return the final offset
}

__global__ void CompactKernel(const scalar_t* a, scalar_t* out, size_t size, CudaVec shape,
                              CudaVec strides, size_t offset) {
  /**
   * The CUDA kernel for the compact opeation.  This should effectively map a single entry in the 
   * non-compact input a, to the corresponding item (at location gid) in the compact array out.
   * 
   * Args:
   *   a: CUDA pointer to a array
   *   out: CUDA point to out array
   *   size: size of out array
   *   shape: vector of shapes of a and out arrays (of type CudaVec, for past passing to CUDA kernel)
   *   strides: vector of strides of out array
   *   offset: offset of out array
   */
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;

  /// BEGIN SOLUTION
  if (gid >= size) return;  // Check for out-of-bounds

    // Compute the offset in the non-compact array `a` and copy to the compact array `out`
    size_t a_offset = index_to_offset_cuda(strides, shape, gid, offset);
    out[gid] = a[a_offset];
  /// END SOLUTION
}

void Compact(const CudaArray& a, CudaArray* out, std::vector<int32_t> shape,
             std::vector<int32_t> strides, size_t offset) {
  /**
   * Compact an array in memory.  Unlike the C++ version, in CUDA this will primarily call the 
   * relevant CUDA kernel.  In this case, we illustrate how you should set this up (i.e., we give 
   * you the code for this fuction, and also the prototype for the CompactKernel() function).  For
   * the functions after this, however, you'll need to define these kernels as you see fit to 
   * execute the underlying function.
   * 
   * Args:
   *   a: non-compact represntation of the array, given as input
   *   out: compact version of the array to be written
   *   shape: shapes of each dimension for a and out
   *   strides: strides of the *a* array (not out, which has compact strides)
   *   offset: offset of the *a* array (not out, which has zero offset, being compact)
   */

  // Nothing needs to be added here
  CudaDims dim = CudaOneDim(out->size);
  CompactKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, out->size, VecToCuda(shape),
                                         VecToCuda(strides), offset);
}
```

In CUDA programming, there is a key distinction between **host** code (which runs on the CPU) and **device** code (which runs on the GPU). In your case:

-   **`Compact()`**, **`EwiseSetitem()`**, and **`ScalarSetitem()`** are **host-side** functions. They prepare and launch GPU kernels and are callable from Python through Pybind11.
-   **`CompactKernel()`** (and other kernels you will define, such as for `Setitem`) are **device-side** functions. These kernels are executed in parallel on the GPU by many threads.

### Explain Thread, Block, Block Dimension and Grid

#### 1. **Thread (`threadIdx`)**

A **thread** is the basic unit of execution in CUDA. Each thread executes the same code but can operate on different data based on its unique ID.

-   **`threadIdx`** refers to the **index of the thread within a block**. It’s a built-in variable provided by CUDA to give each thread a unique identifier **within its block**.
    
    -   **Example**: If a block has 40 threads (i.e., `blockDim.x = 5` and `blockDim.y = 8`), the **thread indices** are identified by their `x` and `y` coordinates:
        
        -   **`threadIdx.x`** will range from `0` to `4` (5 columns in each block).
        -   **`threadIdx.y`** will range from `0` to `7` (8 rows in each block).
        
        In this case, each thread is assigned a unique 2D coordinate `(threadIdx.x, threadIdx.y)` within the block.
        
        For instance:
        
        -   `(threadIdx.x = 0, threadIdx.y = 0)` is the top-left thread of the block.
        -   `(threadIdx.x = 4, threadIdx.y = 7)` is the bottom-right thread of the block.
```css
blockDim.x = 5 (columns)
blockDim.y = 8 (rows)

(threadIdx.x, threadIdx.y)

(0, 0)  (1, 0)  (2, 0)  (3, 0)  (4, 0)  // Row 0
(0, 1)  (1, 1)  (2, 1)  (3, 1)  (4, 1)  // Row 1
(0, 2)  (1, 2)  (2, 2)  (3, 2)  (4, 2)  // Row 2
(0, 3)  (1, 3)  (2, 3)  (3, 3)  (4, 3)  // Row 3
(0, 4)  (1, 4)  (2, 4)  (3, 4)  (4, 4)  // Row 4
(0, 5)  (1, 5)  (2, 5)  (3, 5)  (4, 5)  // Row 5
(0, 6)  (1, 6)  (2, 6)  (3, 6)  (4, 6)  // Row 6
(0, 7)  (1, 7)  (2, 7)  (3, 7)  (4, 7)  // Row 7
```
> In CUDA, the **x-dimension** represents the columns, with `threadIdx.x` indicating the thread's position within those columns, while the **y-dimension** represents the rows, with `threadIdx.y` specifying the thread's position within those rows. Unlike traditional matrix indexing conventions, `threadIdx.x` and `threadIdx.y` are part of a thread coordinate system within a block. This mapping of `threadIdx.x` to the horizontal (column) axis and `threadIdx.y` to the vertical (row) axis is based on visualizing the 2D structure of threads in a block rather than matrix conventions. CUDA follows this convention because it aligns with a natural 2D Cartesian coordinate system, where `threadIdx.x` corresponds to horizontal movement and `threadIdx.y` to vertical movement, making it easier to conceptualize thread organization and execution.
> 
> **Visualizing a grid**:
>    - **threadIdx.x** moves **horizontally** (left to right), similar to moving across columns.
>     -   **threadIdx.y** moves **vertically** (top to bottom), similar to moving across rows.
#### 2. **Block (`blockIdx`)**

CUDA threads are organized into **blocks**. A block is a group of threads that execute together, and each block has a unique index within the **grid**.

-   **`blockIdx`** refers to the **index of the block within the grid**. Like `threadIdx`, it’s a built-in variable used to identify the block's position **in the grid**.
    
    -   **Example**: If there are 10 blocks in the grid (i.e., `gridDim.x = 10`), the **block indices** (`blockIdx.x`) will range from `0` to `9`.

#### 3. **Block Dimension (`blockDim`)**

**`blockDim`** refers to the number of threads **within each block** along a specific dimension. CUDA blocks can be 1D, 2D, or 3D, and `blockDim.x` is the **number of threads per block in the x-dimension**.

-   **Example**: If you launch a grid of blocks with `blockDim.x = 256`, then each block contains 256 threads.

#### 4. **Grid**

A **grid** is a collection of blocks. It is organized into **blocks** that execute independently, and the blocks are indexed within the grid. You can launch a grid with blocks that are 1D, 2D, or 3D.

-   **Grid of threads**: When you launch a kernel, you define a grid of blocks. Each block has its threads, and together these threads form a **grid of threads** that work in parallel.
    
    -   **gridDim**: The number of blocks in each dimension of the grid. For example, `gridDim.x` is the number of blocks in the x-dimension.

#### Putting It All Together:

-   **Threads are organized into blocks**.
    -   Each block contains multiple threads.
    -   The threads within a block are identified by `threadIdx`.
-   **Blocks are organized into a grid**.
    -   The blocks are identified by `blockIdx`.
    -   The blocks within a grid can contain many threads.

#### Example of a Grid with Blocks and Threads:

Suppose you launch a CUDA kernel with:
```cpp
dim3 blockDim(4, 4);  // 16 threads per block (4x4 threads)
dim3 gridDim(2, 2);   // 4 blocks in the grid (2x2 blocks)
```
Here’s how it looks:

-   The **grid** contains `2 x 2 = 4 blocks` (i.e., `gridDim.x = 2`, `gridDim.y = 2`).
-   Each **block** contains `4 x 4 = 16 threads` (i.e., `blockDim.x = 4`, `blockDim.y = 4`).
-   **Threads within a block** are identified by `threadIdx` (e.g., `(threadIdx.x, threadIdx.y)`).
-   **Blocks within the grid** are identified by `blockIdx` (e.g., `(blockIdx.x, blockIdx.y)`).

**Example of threadIdx.x, threadIdx.y**
```css
Block (4x4 threads):

(threadIdx.x, threadIdx.y)

(0, 0)  (1, 0)  (2, 0)  (3, 0)  // Row 0
(0, 1)  (1, 1)  (2, 1)  (3, 1)  // Row 1
(0, 2)  (1, 2)  (2, 2)  (3, 2)  // Row 2
(0, 3)  (1, 3)  (2, 3)  (3, 3)  // Row 3
```
For a block with `4x4` threads (`blockDim.x = 4`, `blockDim.y = 4`), `threadIdx.x` and `threadIdx.y` range from `0` to `3` because:
-   `threadIdx.x`: Column index of the thread (ranging from `0` to `blockDim.x - 1`).
-   `threadIdx.y`: Row index of the thread (ranging from `0` to `blockDim.y - 1`).


#### Total Number of Threads

The total number of threads in the grid is the product of the number of blocks and the number of threads per block:
```cpp
Total Threads = gridDim.x * gridDim.y * blockDim.x * blockDim.y
```
In our example, the total number of threads is:
```cpp
Total Threads = (2 * 2) * (4 * 4) = 4 * 16 = 64 threads
```

#### How Threads Are Indexed:

Each thread is uniquely identified by both its `blockIdx` (its block within the grid) and `threadIdx` (its position within the block). CUDA uses these indices to assign work to each thread.

In the case of a **1D grid of blocks**, you can compute the **global thread index** as follows:
```cpp
size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
```
This calculates the **global ID** of the thread across all blocks in a 1D grid. It combines the block index and the thread index within the block to determine the global position of the thread.

#### Summary:

-   **`threadIdx`**: Index of the thread **within a block**.
-   **`blockIdx`**: Index of the block **within the grid**.
-   **`blockDim`**: Number of threads **per block**.
-   **Grid**: A collection of blocks, each containing multiple threads, forming a **grid of threads**.

These concepts are essential for efficiently organizing and running parallel computations on the GPU using CUDA.

### Explain SIMD (Single Instruction, Multiple Data) and SIMT (Single Instruction, Multiple Threads)

#### 1. **SIMD (Single Instruction, Multiple Data)**

-   **How it works**: In SIMD, a single instruction is applied to multiple **data points** simultaneously by using **one unit** or **one core** with vector registers that can store multiple data points. All data points in the vector are processed **in parallel**, but the operation (instruction) is the same for all data points.
    
-   **Key point**: In SIMD, a single processing unit (like a CPU core) applies the same operation to multiple data points simultaneously.
    
-   **Example**: Suppose we want to add two vectors:

```css
A = [1, 2, 3, 4]
B = [5, 6, 7, 8]
```
In SIMD, a single instruction would perform the addition on **all elements at once**:
```css
C = A + B
C = [6, 8, 10, 12]
```
-   The CPU’s SIMD unit would load the vectors `A` and `B` into vector registers, and then perform the addition on the entire vector in one go, processing the elements in parallel.
    
-   **Hardware**: CPUs typically have SIMD instructions like **SSE**, **AVX**, and **NEON** that are optimized for this type of operation. These instructions can operate on vectors (arrays of values) directly.

#### 2. **SIMT (Single Instruction, Multiple Threads)**

-   **How it works**: In SIMT, a single instruction is executed across **multiple threads**, and each thread can operate on different data. Unlike SIMD, each thread is a separate entity that can have its own flow of execution, although they often perform the same operations on different data. The main difference is that **each thread is an independent processing unit** and can theoretically follow its own control flow (though this causes performance issues if threads diverge too much).
    
-   **Key point**: In SIMT, **multiple threads** (each its own execution unit) execute the same instruction but on different data, and each thread can operate independently.
    
-   **Example**: Suppose we want to add two vectors again:

```css
A = [1, 2, 3, 4]
B = [5, 6, 7, 8]
```
-   In SIMT (like on a GPU), each **thread** would handle one element of the vectors. If there are 4 threads, one thread might handle:
    
    -   Thread 0: `C[0] = A[0] + B[0]` → `C[0] = 1 + 5 = 6`
    -   Thread 1: `C[1] = A[1] + B[1]` → `C[1] = 2 + 6 = 8`
    -   Thread 2: `C[2] = A[2] + B[2]` → `C[2] = 3 + 7 = 10`
    -   Thread 3: `C[3] = A[3] + B[3]` → `C[3] = 4 + 8 = 12`
    
    Here, the GPU uses multiple threads, and each thread operates on its own element from `A` and `B`. All threads execute the same instruction (addition), but they work independently on different data points.
    
-   **Hardware**: GPUs use SIMT architecture. In **NVIDIA CUDA**, the GPU uses warps (groups of 32 threads) to perform operations in parallel. Each thread in a warp executes the same instruction, but they operate on different data elements.

### Explain of `dim3`
In CUDA, `dim3` is a built-in data structure that represents a 3D vector used to specify the dimensions of thread blocks and grids. It can hold three values, corresponding to the **x**, **y**, and **z** dimensions.

When you define `blockDim` and `gridDim` using `dim3`, you're specifying the size of each block and grid in terms of threads (for blocks) and blocks (for grids).
```cpp
dim3 blockDim(4, 4);  // Defines a block with 4 threads in the x-dimension and 4 threads in the y-dimension
dim3 gridDim(2, 2);   // Defines a grid with 2 blocks in the x-dimension and 2 blocks in the y-dimension
```
In this example:

-   **blockDim(4, 4)** specifies a 2D block with 4 threads along the x-axis and 4 threads along the y-axis, resulting in 16 threads per block.
-   **gridDim(2, 2)** specifies a 2D grid with 2 blocks along both the x and y axes, so there are a total of 4 blocks in the grid.

The **z-dimension** is optional and defaults to 1 if not specified. So, `dim3(4, 4)` is equivalent to `dim3(4, 4, 1)`.


### Explain of `size_t gid = blockIdx.x * blockDim.x + threadIdx.x`

```cpp
size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
```
is for a **1D grid** and **1D blocks** setup. In this case:

-   **`blockIdx.x`** is the index of the block in the grid (1D grid).
-   **`blockDim.x`** is the number of threads per block (1D block).
-   **`threadIdx.x`** is the index of the thread within its block (1D thread indexing).

**Visualization**

Consider the following setup:

-   **Grid**: A 1D grid of **3 blocks** (`gridDim.x = 3`).
-   **Block**: Each block has **4 threads** (`blockDim.x = 4`).

This will create a total of **12 threads** (3 blocks × 4 threads).

#### Threads within each block:
```css
Block 0: [Thread 0, Thread 1, Thread 2, Thread 3]
Block 1: [Thread 0, Thread 1, Thread 2, Thread 3]
Block 2: [Thread 0, Thread 1, Thread 2, Thread 3]
```
#### Global Thread Index (`gid`):

The **global thread index (gid)** in CUDA is the unique identifier for each thread running in a CUDA kernel, allowing each thread to work on a different piece of data independently. This index is computed based on the grid and block structure of CUDA, which organizes threads into blocks, and blocks into grids.

The global index (`gid`) is calculated using:

```cpp
size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
```
-   `blockIdx.x`: The index of the current block within the grid.
-   `blockDim.x`: The number of threads in each block.
-   `threadIdx.x`: The index of the thread within the current block.

Now, let's compute `gid` for each thread:

| Block   | `threadIdx.x` | `blockIdx.x` | Global ID (`gid`)  |
|---------|---------------|--------------|--------------------|
| Block 0 | 0             | 0            | 0 * 4 + 0 = 0      |
| Block 0 | 1             | 0            | 0 * 4 + 1 = 1      |
| Block 0 | 2             | 0            | 0 * 4 + 2 = 2      |
| Block 0 | 3             | 0            | 0 * 4 + 3 = 3      |
| Block 1 | 0             | 1            | 1 * 4 + 0 = 4      |
| Block 1 | 1             | 1            | 1 * 4 + 1 = 5      |
| Block 1 | 2             | 1            | 1 * 4 + 2 = 6      |
| Block 1 | 3             | 1            | 1 * 4 + 3 = 7      |
| Block 2 | 0             | 2            | 2 * 4 + 0 = 8      |
| Block 2 | 1             | 2            | 2 * 4 + 1 = 9      |
| Block 2 | 2             | 2            | 2 * 4 + 2 = 10     |
| Block 2 | 3             | 2            | 2 * 4 + 3 = 11     |

**Diagram:**
```css
Grid (3 blocks)
|
|-- Block 0 (4 threads) -> Global IDs: 0, 1, 2, 3
|
|-- Block 1 (4 threads) -> Global IDs: 4, 5, 6, 7
|
|-- Block 2 (4 threads) -> Global IDs: 8, 9, 10, 11
```
Each thread has a unique **global ID (`gid`)** that allows it to operate on specific data.

### Explain of `index_to_offset_cuda`

```cpp
__device__ size_t index_to_offset_cuda(const CudaVec& strides, const CudaVec& shape, size_t index, size_t base_offset) {
    size_t offset = base_offset;  // Start with the base offset

    // Loop over the dimensions in reverse
    for (int dim = shape.size - 1; dim >= 0; dim--) {
        offset += (index % shape.data[dim]) * strides.data[dim];  //Calculate offset using strides, where (index % shape.data[dim]) gives the index within this dimension.
        index /= shape.data[dim];  // Move to the next dimension by dividing by the current shape
    }

    return offset;  // Return the final offset
}
```
This function, `index_to_offset_cuda`, is designed to compute the linear memory offset in a non-compact array (where data is stored with strides) from a flat (1D) index in a compact array. This is useful when you need to access a multi-dimensional array stored in linear memory, but with irregular spacing between elements (due to strides). 

The **`__device__`** keyword in CUDA is used to indicate that a function or variable is executed or resides on the **GPU (device)** and can only be called or accessed by other code running on the GPU.

#### Arguments:

-   **`strides`**: A `CudaVec` representing how far apart elements in each dimension are in memory. Strides tell us how much to jump in memory to get from one element to the next in each dimension.
-   **`shape`**: A `CudaVec` that gives the size of each dimension. It represents how many elements there are in each dimension.
-   **`index`**: A flat (1D) index that corresponds to a position in a compact (contiguous) array.
-   **`base_offset`**: The starting point or offset from which the computed offset will be added. It is often 0 but can be non-zero in certain cases.

#### Explain with example

Let's assume we have a 2D array of size `3x3` (3 rows, 3 columns):

```cpp
array = [
    [a, b, c],
    [d, e, f],
    [g, h, i]
]
```

-   **index 0** -> position **(0, 0)** -> value **a**
-   **index 1** -> position **(0, 1)** -> value **b**
-   **index 2** -> position **(0, 2)** -> value **c**
-   **index 3** -> position **(1, 0)** -> value **d**
-   **index 4** -> position **(1, 1)** -> value **e**
-   **index 5** -> position **(1, 2)** -> value **f**
-   **index 6** -> position **(2, 0)** -> value **g**
-   **index 7** -> position **(2, 1)** -> value **h**
-   **index 8** -> position **(2, 2)** -> value **i**

Let's say we have a 2D array (flattened into 1D), and its shape is `shape = [4, 5]` (i.e., 4 rows and 5 columns).

-   If the linear `index` is `17`, the task is to find which position it corresponds to in the 2D array.

Steps:

1.  **First dimension (columns)**:  
    We need to know how many columns there are, so we use `%` with `shape[1] = 5` to get the position in the column.
    
```cpp
17 % 5 = 2  // This tells us the position in the second dimension (columns)
```

2. **Move to the next dimension (rows)**:  
Now, we need to adjust the `index` to focus on the first dimension. We use `/=` with `shape[1] = 5` to reduce the index:

```cpp
17 /= 5 = 3  // Now the index corresponds to the first dimension (rows)
```
Now, we know that the linear index `17` corresponds to position `(3, 2)` in the 2D array (row 3, column 2).

-   **`%`** extracts the position in the current dimension.
-   **`/=`** moves the index to the next higher dimension by adjusting for the current dimension's size.


#### Why Start from the Last Dimension?
```cpp
for (int dim = shape.size - 1; dim >= 0; dim--) {
    offset += (index % shape.data[dim]) * strides.data[dim];  // Calculate offset using strides
    index /= shape.data[dim];  // Move to the next dimension
}
```
We start from the last dimension because, in **row-major** (or **contiguous** memory layout), elements in the last dimension are stored consecutively. Starting from the last dimension allows us to handle the smaller memory offsets first and move toward larger dimensions. This is similar to how a number is broken down from the least significant digit to the most significant.

For example, if we have a 2D array (a matrix), the elements of each row are stored contiguously in memory. So, when calculating the memory offset for a specific element, we first calculate how far along the row we are (using the last dimension) before moving on to the larger offset for the entire row (using the earlier dimensions).

### Explain `CudaOneDim`

```cpp
#define BASE_THREAD_NUM 256

struct CudaDims {
  dim3 block, grid;
};

CudaDims CudaOneDim(size_t size) {
  /**
   * Utility function to get cuda dimensions for 1D call
   */
  CudaDims dim;
  size_t num_blocks = (size + BASE_THREAD_NUM - 1) / BASE_THREAD_NUM;
  dim.block = dim3(BASE_THREAD_NUM, 1, 1);
  dim.grid = dim3(num_blocks, 1, 1);
  return dim;
}
```
The function `CudaOneDim` is a utility function used to calculate the **grid** and **block** dimensions for a 1D CUDA kernel launch. CUDA launches kernels using a grid of blocks, and each block contains a number of threads. The purpose of this function is to set up the appropriate configuration for a one-dimensional computation based on the size of the problem (number of elements to process).

1. **Function Parameters:**
```cpp
CudaDims CudaOneDim(size_t size)
```
**`size`**: The total number of elements or the workload size. This could represent the number of elements in an array that you want to process.

2. **Return Value:**

The function returns a structure of type `CudaDims`, which contains two members:

-   **`dim.block`**: The block dimensions (how many threads per block).
-   **`dim.grid`**: The grid dimensions (how many blocks in the grid).

This structure helps configure the CUDA kernel launch with appropriate values for the number of blocks and threads.

3. **Calculation of Number of Blocks:**
```cpp
size_t num_blocks = (size + BASE_THREAD_NUM - 1) / BASE_THREAD_NUM;
```

-   **`BASE_THREAD_NUM`**: This constant represents the number of threads per block. Typically, a value like 256 or 512 is chosen to balance the workload across GPU cores effectively.
    
-   **`(size + BASE_THREAD_NUM - 1) / BASE_THREAD_NUM`**: This formula calculates the number of blocks required to process all elements:
    
    -   **`size`** is the total number of elements.
    -   **`BASE_THREAD_NUM`** is the number of threads per block.
    -   By adding `BASE_THREAD_NUM - 1` before dividing, the formula ensures that any remaining elements (if `size` is not a perfect multiple of `BASE_THREAD_NUM`) will be handled by an extra block. This ensures that the total number of blocks is sufficient to cover the entire dataset.
    
    Example:
    
    -   If `size = 1000` and `BASE_THREAD_NUM = 256`:
        -   The number of blocks would be `(1000 + 256 - 1) / 256 = 4`.
        -   This ensures that all 1000 elements can be processed across 4 blocks.

4. **Block Dimensions:**

```cpp
dim.block = dim3(BASE_THREAD_NUM, 1, 1);
```

-   This sets the block dimensions to a 1D block, where each block contains `BASE_THREAD_NUM` threads.
-   **`dim3(BASE_THREAD_NUM, 1, 1)`**: The `dim3` function constructs a 3D grid/block structure, but in this case, only the x-dimension is used, meaning that each block will contain `BASE_THREAD_NUM` threads along the x-axis. The other two dimensions (y and z) are set to 1.

5. **Grid Dimensions:**
```cpp
dim.grid = dim3(num_blocks, 1, 1);
```
-   This sets the grid dimensions to a 1D grid, where there are `num_blocks` blocks in the x-dimension.
-   **`dim3(num_blocks, 1, 1)`**: Similarly, the `dim3` constructor creates a 3D grid structure, but only the x-dimension is used. The number of blocks in the grid is set to `num_blocks`, calculated earlier.

6. **Returning the Structure:**
```cpp
return dim;
```
- The function returns the `CudaDims` structure containing the block and grid configurations.

#### Summary:

-   **`CudaOneDim(size_t size)`** calculates the number of blocks and threads required to process a 1D workload in CUDA.
-   It ensures that all elements (of size `size`) are divided across the appropriate number of threads (`BASE_THREAD_NUM` threads per block).
-   The result is a **1D grid** and **1D block** configuration (`dim3`), which is returned as a `CudaDims` structure and used to launch the CUDA kernel.

### Explain `if (gid >= size) return`

Let's say we have an array `a` with 100 elements and we want to copy it into a compact array `out`. The kernel launch could look something like this:

```cpp
int size = 100;  // Number of elements in the array
int threadsPerBlock = 32;
int blocks = (size + threadsPerBlock - 1) / threadsPerBlock; // Calculates the number of blocks needed to cover the total work (size)
CompactKernel<<<blocks, threadsPerBlock>>>(a, out, size, shape, strides, offset);
```
Here, `threadsPerBlock` is 32, and `blocks` would be `(100 + 32 - 1) / 32 = 4`. This means the total number of threads launched is `4 * 32 = 128`.

> `int blocks = (size + threadsPerBlock - 1) / threadsPerBlock;` is used to calculate the number of blocks needed to cover a given amount of work (`size`), where `size` is the total number of elements or tasks that need to be processed.

#### Why the `if (gid >= size) return;` check is necessary:

The kernel would spawn 128 threads in total, but we only have 100 elements to process. Each thread is assigned a unique global ID (`gid`), and each thread is responsible for processing one element of the array.

-   Threads with `gid` from 0 to 99 will have valid data to process since the array has 100 elements.
-   However, threads with `gid` from 100 to 127 exceed the array size. These threads do not correspond to any valid data in the array.

Without the `if (gid >= size) return;` check, threads with `gid` in the range [100, 127] would still try to access memory. Since these threads attempt to access indices beyond the bounds of the array, they would likely cause **out-of-bounds memory access**, leading to crashes or undefined behavior.

### Explaination of `CompactKernel`

```cpp
__global__ void CompactKernel(const scalar_t* a, scalar_t* out, size_t size, CudaVec shape,
                              CudaVec strides, size_t offset) {
  /**
   * The CUDA kernel for the compact opeation.  This should effectively map a single entry in the 
   * non-compact input a, to the corresponding item (at location gid) in the compact array out.
   * 
   * Args:
   *   a: CUDA pointer to a array
   *   out: CUDA point to out array
   *   size: size of out array
   *   shape: vector of shapes of a and out arrays (of type CudaVec, for past passing to CUDA kernel)
   *   strides: vector of strides of out array
   *   offset: offset of out array
   */
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;

  /// BEGIN SOLUTION
  if (gid >= size) return;  // Check for out-of-bounds

    // Compute the offset in the non-compact array `a` and copy to the compact array `out`
    size_t a_offset = index_to_offset_cuda(strides, shape, gid, offset);
    out[gid] = a[a_offset];
  /// END SOLUTION
```
This CUDA kernel function `CompactKernel` is designed to take a non-compact input array `a`, compute its corresponding compact memory layout, and store the result in the output array `out`. The operation aims to map each element from the non-compact array `a` into a compact array `out` by calculating the appropriate memory offset based on the shape and strides.

#### 1. **`size_t gid = blockIdx.x * blockDim.x + threadIdx.x;`**

This line calculates the **global thread index (gid)**. CUDA threads are launched in blocks, and each block contains multiple threads. This formula computes a unique identifier for each thread based on:

-   **`blockIdx.x`**: Refers to the index of the block in a 1D grid (starting from 0).
-   **`blockDim.x`**: Refers to the number of threads per block in the 1D block.
-   **`threadIdx.x`**: Refers to the index of the thread within its block in a 1D block (also starting from 0).

So, `gid` is the unique index representing each thread across all blocks.

#### 2. **`if (gid >= size) return;`**

This is an **out-of-bounds check**. Each thread is responsible for processing one element, and `size` represents the total number of elements in the `out` array. If the `gid` (thread index) is greater than or equal to `size`, the thread is trying to access elements beyond the bounds of the array. To prevent invalid memory access, this thread immediately exits the kernel using `return`.

This prevents out-of-bounds memory access, ensuring only threads with a valid `gid` process data.

#### 3. **`size_t a_offset = index_to_offset_cuda(strides, shape, gid, offset);`**

Here, we calculate the **memory offset** in the non-compact array `a` that corresponds to the element the current thread (`gid`) is responsible for.

-   `index_to_offset_cuda()` is a function that calculates the position of an element in the non-compact array `a` using:
    -   **`strides`**: A vector that tells us how far apart elements in different dimensions are located in memory. This helps map multi-dimensional indices to one-dimensional offsets.
    -   **`shape`**: The shape of the array, which tells us how many elements exist in each dimension.
    -   **`gid`**: The global thread index, which acts as the linear index.
    -   **`offset`**: The base offset from where the data starts in the non-compact array.

This step translates the linear `gid` into a proper memory offset in the non-compact array using its multi-dimensional shape and strides.

#### 4. **`out[gid] = a[a_offset];`**

Finally, the thread retrieves the value from the non-compact array `a` at the calculated offset (`a_offset`) and stores it in the corresponding compact array `out` at the index `gid`. This is the main operation where data is copied from the non-compact format to the compact format.

### Explain parallelized manner 🌟🌟🌟🌟🌟

This kernel ensures that data is correctly transferred from a non-compact memory layout to a compact one using multi-dimensional indexing in a parallelized manner on the GPU. 

1. **Multiple Threads Working Simultaneously**:

-   In this kernel, many threads are launched simultaneously. Each thread works on a different element of the array, copying the data from the non-compact array `a` to the compact array `out`. This way, rather than copying elements one by one in a sequential manner (which would be slow), many elements are copied at the same time.
-   For example, if you have 1000 elements to process and 1000 threads are launched, each thread will handle exactly one element, allowing all elements to be processed in parallel.



2.  **Efficient Resource Utilization**:
    
    -   GPUs have thousands of cores capable of performing simple operations simultaneously. By launching many threads to handle different parts of the array, the kernel can efficiently use the GPU's computational power. Each thread works independently without waiting for others to finish, maximizing throughput.

#### In This Kernel:

-   **Non-Compact to Compact Layout Transfer**: The operation of copying data from a non-compact memory layout to a compact layout can be done in parallel. Each thread computes the offset for its assigned element, accesses the correct data in the non-compact array, and writes the result to the corresponding location in the compact array.
-   **Parallel Workload**: If the array has, for example, 1000 elements, you could launch 1000 threads, and each thread will compute its own global thread index (`gid`) and process exactly one element of the array. This allows the entire array to be processed in parallel, instead of processing the elements one by one sequentially.

#### Benefits of Parallelization:

-   **Speed**: Since the GPU can handle thousands of threads in parallel, operations that involve large datasets (like transferring data from one memory layout to another) can be done much faster than a traditional CPU approach, where operations are typically done sequentially.
-   **Scalability**: As the size of the dataset grows, the benefits of parallelization increase. With more threads handling the workload, larger datasets can still be processed efficiently.

#### Example:

Consider an array of size 10,000. In a **sequential** (non-parallelized) approach, a CPU would process one element at a time, resulting in 10,000 iterations.

In the **parallelized manner** on a GPU, you might launch 1,000 threads, each responsible for handling 10 elements. Each thread works simultaneously, drastically reducing the overall time required to complete the task, since multiple elements are being processed at once.

#### Summary:

In this kernel, **parallelized manner** means that instead of copying data sequentially (one element at a time), many threads are used to process multiple elements simultaneously. This enables the GPU to leverage its highly parallel architecture, allowing for much faster data transfers and computations.

#### Compare CPU vs GPU

-   **CPU**: A CPU has a small number of cores (often 4 to 16 in consumer devices, more in high-performance systems), each capable of handling multiple threads using techniques like **hyper-threading**. CPUs are optimized for sequential tasks and complex operations, where each core handles a few threads with high individual performance. They can do parallel work but are primarily designed for versatility and handling different types of tasks efficiently.
    
-   **GPU**: A GPU, on the other hand, has **thousands** of smaller, simpler cores designed for massive parallelism. GPUs excel at handling many threads simultaneously, making them ideal for tasks where a large number of simple operations need to be performed in parallel, such as processing large datasets or running machine learning models.

Let’s consider a simple task: **adding two arrays element-wise**. You have two arrays, `A` and `B`, each with 8 elements, and you want to compute a new array `C` where `C[i] = A[i] + B[i]`.

##### Example 1: CPU (Sequential Execution)

On a CPU, the task is typically performed **sequentially**. This means that each addition is performed one after another, using a single processing core or thread.

```cpp
// CPU version: Sequential execution
void add_arrays_cpu(const int* A, const int* B, int* C, size_t size) {
    for (size_t i = 0; i < size; i++) {
        C[i] = A[i] + B[i];  // Perform one addition at a time
    }
}
```
Here’s what happens:

1.  The CPU picks the first element (`A[0] + B[0]`) and stores the result in `C[0]`.
2.  Then it moves to the second element (`A[1] + B[1]`) and stores the result in `C[1]`.
3.  This continues until all 8 additions are performed **sequentially**.

So, if you have 8 elements, the CPU does 8 steps, one at a time.

##### Example 2: GPU (Parallel Execution)

On a GPU, tasks can be performed **in parallel** by using multiple threads to handle multiple operations at once. In this case, each thread can be responsible for adding one element of `A` and `B` and storing the result in `C`.
```cpp
// GPU version: Parallel execution
__global__ void add_arrays_gpu(const int* A, const int* B, int* C, size_t size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        C[idx] = A[idx] + B[idx];  // Each thread handles one addition
    }
}
```
Here’s what happens on the GPU:

1.  The GPU launches **8 threads** in parallel, each one responsible for adding a single pair of elements.
2.  Each thread independently computes `C[i] = A[i] + B[i]` for a different index `i` at the same time.

So, if you have 8 elements and launch 8 threads, the GPU can compute all 8 additions **simultaneously** in a single step. This is parallelism.

 **Visualization:**

##### CPU (Sequential):
```css
Time 1: A[0] + B[0] -> C[0]
Time 2: A[1] + B[1] -> C[1]
Time 3: A[2] + B[2] -> C[2]
...
Time 8: A[7] + B[7] -> C[7]
```
##### GPU (Parallel):
```css
Time 1: A[0] + B[0] -> C[0]
         A[1] + B[1] -> C[1]
         A[2] + B[2] -> C[2]
         ...
         A[7] + B[7] -> C[7] (all happen at the same time)
```
On a **CPU**, tasks are typically performed one at a time in sequence, while on a **GPU**, tasks can be broken down into smaller units of work (such as array element addition) and performed in parallel across many threads, significantly speeding up the process. This parallelism is what gives GPUs their massive performance advantage in handling large datasets.

#### Explain In CUDA CPU  as host and GPU as device
In CUDA and parallel computing terminology, the **CPU is referred to as the "host"** and the **GPU as the "device"** because of the following reasons:

#### 1. **Role in the System Architecture**:

-   **CPU (Host)**:
    
    -   The **CPU** controls and manages the overall execution of programs. It handles tasks like memory allocation, launching GPU kernels, and coordinating data transfers between the CPU and GPU. In this sense, the CPU is responsible for **"hosting"** the entire operation.
    -   It is also where the operating system and most of the system's primary functions run.
    -   The CPU is good at handling complex, general-purpose tasks with low levels of parallelism.
-   **GPU (Device)**:
    
    -   The **GPU** is designed specifically for **parallel processing**. It excels at executing the same operation on multiple data points simultaneously. In CUDA, the GPU is referred to as the **"device"** because it is a specialized hardware resource that performs specific computational tasks **as instructed by the host (CPU)**.
    -   The GPU relies on the CPU to initiate and manage the computation, including allocating data and launching kernels.
    -   The GPU handles massively parallel operations, such as matrix computations, image processing, or machine learning tasks.

#### 2. **CPU as the Master Controller**:

In CUDA programming, the **CPU (host)** initiates and controls the flow of the program. It is the master that controls:

-   Allocating memory on both the CPU and GPU.
-   Copying data from the host to the device and vice versa.
-   Launching GPU kernels to run on the GPU.
-   Managing the synchronization between the CPU and GPU.

The GPU (**device**) performs the heavy lifting of the actual computations in parallel, but it depends on the CPU to initiate these tasks.

#### 3. **Specialization of Roles**:

-   **Host (CPU)**: Acts as the controller, managing memory, data transfers, and kernel launches.
-   **Device (GPU)**: Acts as the processor, handling large-scale parallel computations as assigned by the host.

#### Could the CPU Be a Device?

Yes, the CPU can also be considered a **device** in other contexts, but not in the same sense as the GPU in CUDA programming.

For example:

-   In traditional multi-core CPU programs (using threads or multiprocessing libraries), one CPU core may be assigned tasks by a **controller thread**. In this case, the CPU core could be thought of as a **device** in a broader sense, since it's executing specific tasks.
-   In distributed systems, one machine can act as the **host** while treating other machines (which could be CPUs) as **devices**.

However, in CUDA programming:

-   **Host** refers to the **CPU** (as the controller).
-   **Device** refers to the **GPU** (as the worker that performs the computation).

### Explain `VecToCuda`
```cpp
#define MAX_VEC_SIZE 8
struct CudaVec {
  uint32_t size;
  int32_t data[MAX_VEC_SIZE];
};

CudaVec VecToCuda(const std::vector<int32_t>& x) {
  CudaVec shape;
  if (x.size() > MAX_VEC_SIZE) throw std::runtime_error("Exceeded CUDA supported max dimesions");
  shape.size = x.size();
  for (size_t i = 0; i < x.size(); i++) {
    shape.data[i] = x[i];
  }
  return shape;
}
```
The code defines a structure `CudaVec` and a function `VecToCuda` to convert a standard C++ `std::vector<int32_t>` into a `CudaVec` format, which is suitable for passing between the host (CPU) and device (GPU) in CUDA programming. This function is useful when you need to pass small arrays (like dimensions or strides) from the CPU to the GPU in a CUDA kernel. Since CUDA kernels cannot directly handle `std::vector`, this conversion to `CudaVec` makes it easier to pass and manage fixed-size arrays on the device (GPU) side. 

> In CUDA programming, data typically originates in the host (CPU) memory, where it is prepared and structured before being transferred to the device (GPU) memory for parallel processing. The reason why the function is designed to convert data from CPU to GPU is that CUDA kernels run on the GPU but are launched from the CPU, meaning the data must be transferred from the CPU's memory to the GPU's memory for the kernel to process it.

Here’s a detailed explanation of each component:

**#define MAX_VEC_SIZE 8**
```cpp
#define MAX_VEC_SIZE 8
```
- This line defines a macro `MAX_VEC_SIZE` with a value of 8, meaning the maximum size of the `CudaVec` array is 8. This is a constraint for the number of dimensions the `CudaVec` can support.

**struct CudaVec**
```cpp
struct CudaVec {
  uint32_t size;
  int32_t data[MAX_VEC_SIZE];
};
```
-   **`CudaVec`** is a structure that represents a vector with a limited number of dimensions, specifically up to `MAX_VEC_SIZE` (8 dimensions in this case). The `CudaVec` structure is a fixed-size container for small arrays (up to 8 elements), often representing things like array shapes or strides in CUDA programming.
    
-   **Fields**:
    
    -   **`size`**: A `uint32_t` integer (an unsigned 32-bit integer) representing the number of dimensions (or the size) of the vector. This indicates how many elements in `data[]` are valid. For example, if the vector represents a 3D array, this field would store the number `3`.
    -   **`data[MAX_VEC_SIZE]`**: An array of size `MAX_VEC_SIZE` that stores up to 8 integers (each of type `int32_t`). These integers represent the values or dimensions of the vector. For instance, in a 3D vector with dimensions 3x4x5, this array would store the values `[3, 4, 5]`.

#### Explain Fields

In programming, **fields** refer to the variables that are defined inside a structure, class, or object. They hold data or attributes that describe the state or properties of the structure or object.

In the context of the structure `CudaVec`, **fields** are the member variables **`size`** and **`data[MAX_VEC_SIZE]**. These fields store specific pieces of information about the vector, such as how many dimensions it has and the actual dimension values.

### Explain CudaArray
```cpp
struct CudaArray {
  CudaArray(const size_t size) {
    cudaError_t err = cudaMalloc(&ptr, size * ELEM_SIZE);
    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
    this->size = size;
  }
  ~CudaArray() { cudaFree(ptr); }
  size_t ptr_as_int() { return (size_t)ptr; }
  
  scalar_t* ptr;
  size_t size;
};
```


This struct handles memory allocation on the **GPU (device)** for GPU arrays, when you create an instance of `CudaArray`, it allocates memory on the GPU using `cudaMalloc`, which makes the GPU memory accessible for computation within CUDA kernels. It automatically manages memory with CUDA's `cudaMalloc` and `cudaFree` functions. It also defines a helper method `ptr_as_int` to return the pointer as an integer, which can be useful when debugging or interfacing with other systems.

-   The memory in the **GPU** is allocated when the `CudaArray` is created, and it is freed when the destructor is called (when the `CudaArray` goes out of scope).
    
-   The **`ptr`** in the `CudaArray` struct points to memory on the **GPU**, not on the CPU.

### Explain `Compact`
```cpp
void Compact(const CudaArray& a, CudaArray* out, std::vector<int32_t> shape,
             std::vector<int32_t> strides, size_t offset) {
  /**
   * Compact an array in memory.  Unlike the C++ version, in CUDA this will primarily call the 
   * relevant CUDA kernel.  In this case, we illustrate how you should set this up (i.e., we give 
   * you the code for this fuction, and also the prototype for the CompactKernel() function).  For
   * the functions after this, however, you'll need to define these kernels as you see fit to 
   * execute the underlying function.
   * 
   * Args:
   *   a: non-compact represntation of the array, given as input
   *   out: compact version of the array to be written
   *   shape: shapes of each dimension for a and out
   *   strides: strides of the *a* array (not out, which has compact strides)
   *   offset: offset of the *a* array (not out, which has zero offset, being compact)
   */

  // Nothing needs to be added here
  CudaDims dim = CudaOneDim(out->size);
  CompactKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, out->size, VecToCuda(shape),
                                         VecToCuda(strides), offset);
}
}
```
The `Compact` function is a **host-side function** that manages the configuration and launch of the `CompactKernel` CUDA kernel. Its purpose is to convert a **non-contiguous (non-compact) array** into a **contiguous (compact) layout** in memory using parallelism on the GPU.

#### Arguments:

1.  **`a`**: A `CudaArray` object that represents the non-compact input array on the device (GPU memory). It contains a pointer to the GPU memory where the non-contiguous data resides.
    
2.  **`out`**: A pointer to the `CudaArray` object that will store the compact version of the array. The data will be written to this array in contiguous form.
    
3.  **`shape`**: A `std::vector<int32_t>` containing the shapes of the array across each dimension. It describes the size of the array in each dimension (multi-dimensional array).
    
4.  **`strides`**: A `std::vector<int32_t>` representing the strides of the non-compact array `a`. Strides define how far apart elements are in memory for each dimension. Non-compact arrays often have gaps between elements due to the stride being larger than the next immediate element's memory address.
    
5.  **`offset`**: The offset in memory where the non-compact array `a` starts. The `offset` parameter indicates how far from the base memory address the actual data begins.

#### CUDA Kernel Launch:
```cpp
  CudaDims dim = CudaOneDim(out->size);
  CompactKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, out->size, VecToCuda(shape),
                                         VecToCuda(strides), offset);
}
```
-   **`CudaDims dim = CudaOneDim(out->size);`**: This line defines the CUDA grid and block dimensions for the kernel launch. `CudaOneDim` calculates the number of blocks and threads needed to process the entire array. The total number of threads is based on the size of the output array (`out->size`). It ensures that there are enough threads to handle each element of the array in parallel.
    
-   **`CompactKernel<<<dim.grid, dim.block>>>`**: This launches the CUDA kernel `CompactKernel` with the grid and block configuration determined by `dim.grid` and `dim.block`. This is where the actual data transfer from non-compact to compact memory happens.

**Kernel Arguments**:

-   **`a.ptr`**: The pointer to the non-compact input array `a` on the GPU.
-   **`out->ptr`**: The pointer to the compact output array `out` on the GPU.
-   **`out->size`**: The total number of elements in the compact output array.
-   **`VecToCuda(shape)`**: Converts the `shape` vector to a CUDA-compatible format (e.g., `CudaVec`).
-   **`VecToCuda(strides)`**: Converts the `strides` vector to a CUDA-compatible format.
-   **`offset`**: The memory offset where the non-compact array `a` starts.

#### Summary:

The `Compact` function is designed to transfer data from a non-contiguous array (`a`) to a compact, contiguous array (`out`) on the GPU. It:

1.  Sets up the necessary parameters like shape, strides, and offset.
2.  Computes the grid and block dimensions for parallel execution.
3.  Launches the `CompactKernel` to copy each element of the non-compact array to its corresponding location in the compact array, using parallel threads on the GPU.

The actual copying is done in the `CompactKernel`, which computes the correct offset for each element based on the strides and shape, ensuring the data is transferred in a compact layout.

#### **Explain Data Transfer from CPU to GPU**

In CUDA programming, any data that originates on the CPU (host) must be explicitly copied to the GPU (device) to be used in CUDA kernels. In the code, most of the memory is handled directly on the GPU using CUDA functions like `cudaMalloc`, but let's look at how the function works.

 **VecToCuda**:

This function converts a `std::vector<int32_t>` from the **CPU** into a `CudaVec` structure, which is suitable for GPU kernels. While the function prepares the data for the GPU, it **does not actually transfer** data to the GPU directly.

```cpp
define MAX_VEC_SIZE 8
struct CudaVec {
  uint32_t size;
  int32_t data[MAX_VEC_SIZE];
};

CudaVec VecToCuda(const std::vector<int32_t>& x) {
  CudaVec shape;
  if (x.size() > MAX_VEC_SIZE) throw std::runtime_error("Exceeded CUDA supported max dimesions");
  shape.size = x.size();
  for (size_t i = 0; i < x.size(); i++) {
    shape.data[i] = x[i];
  }
  return shape;
}
```

-   **What happens here?**: The `VecToCuda` function prepares a `CudaVec` structure that can be passed as an argument to a CUDA kernel. This structure resides in **CPU memory**.
-   **Data Transfer**: When the `CompactKernel` is launched, the `VecToCuda` result is passed as an argument, and the CUDA runtime automatically transfers this small structure from the **CPU to the GPU**.

 **Summary of Data Transfer**:

-   Data that originates on the **CPU** (e.g., shapes, strides, scalar values) must be transferred to the **GPU** for the kernel to use it. This is achieved by:
    -   Preparing the data (e.g., `VecToCuda` for shape/strides).
    -   Launching the CUDA kernel, where the prepared data is passed to the GPU.
-   The **actual operations** (e.g., filling an array, compacting data) happen **on the GPU**.
-   Any **non-GPU data (CPU)** is transferred to the **GPU** when the kernel is launched, allowing the GPU to process the data in parallel.


			
#### Why This Is Not Direct CPU-to-GPU Transfer:
```cpp
#define MAX_VEC_SIZE 8
struct CudaVec {
  uint32_t size;
  int32_t data[MAX_VEC_SIZE];
};

CudaVec VecToCuda(const std::vector<int32_t>& x) {
  CudaVec shape;
  if (x.size() > MAX_VEC_SIZE) throw std::runtime_error("Exceeded CUDA supported max dimesions");
  shape.size = x.size();
  for (size_t i = 0; i < x.size(); i++) {
    shape.data[i] = x[i];
  }
  return shape;
}
```
`VecToCuda function` is **not transferring data from the CPU to the GPU**. Instead, it converts a **`std::vector<int32_t>`** (which resides on the **CPU**) into a **`CudaVec`** structure, but this structure is still stored in CPU memory.
    
-   The line `shape.data[i] = x[i];` is just copying data from one CPU memory structure (`std::vector<int32_t> x`) to another CPU memory structure (`CudaVec shape`). This **does not involve the GPU at all**.
    
-   The `shape` that is returned by this function is still a CPU-side data structure. Even though this `CudaVec` can be used in the arguments for a CUDA kernel, it is **not stored in GPU memory**.

**Summary of Data Flow**

-   **`CompactKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, out->size, VecToCuda(shape), VecToCuda(strides), offset);`**:
    -   **`a.ptr`** and **`out->ptr`**: These are **pointers to memory already allocated on the GPU** (by `CudaArray`). The kernel (`CompactKernel`) will access these pointers to read/write data from/to GPU memory. Memory for `a` and `out` is allocated on the **GPU** using `cudaMalloc()` inside the `CudaArray` structure.
    -   **`VecToCuda(shape)`** and **`VecToCuda(strides)`**: These are CPU-side structures converted into `CudaVec`. Even though they are prepared on the CPU, when passed to the kernel during the launch, they are transferred to the GPU as arguments to the kernel.
    
**Kernel Launch**:

When a CUDA kernel is launched using the `<<<grid, block>>>` syntax, CUDA automatically transfers small arguments (such as pointers, integers, and small structures) from the CPU (host) to the GPU (device) memory. In the call `CompactKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, out->size, VecToCuda(shape), VecToCuda(strides), offset);`, these arguments (`a.ptr`, `out->ptr`, `shape`, `strides`, and `offset`) are transferred to the GPU. Once transferred, the GPU uses these arguments to access and manipulate data directly in GPU memory.

In this case:

-   **`a.ptr`** and **`out->ptr`**: These are already pointing to GPU memory.
-   **Other arguments** (like `shape`, `strides`, and `offset`): These are transferred from the CPU to the GPU automatically by the CUDA runtime.
    
The actual **compaction operation** happens on the **GPU**, where each thread processes part of the data, reading from the non-contiguous `a` array and writing the result into the contiguous `out` array.

In summary, **the kernel launch** is the main point where data is passed from the CPU (host) to the GPU (device). Small arguments (like shape and strides) are copied automatically when the kernel is launched, and larger data (like `a` and `out`) is already allocated on the GPU and accessed directly by the kernel.

### Explain kernel 🌟

A **kernel** in the context of CUDA programming and GPU computing is a function that runs on the **GPU** and is executed in parallel by multiple threads. Kernels are marked by the **`__global__`** keyword in CUDA, which indicates that they are to be launched by the host (CPU) and executed on the device (GPU).

**Kernel Launch** 
Kernels are launched from the **host (CPU)** using the syntax `<<<grid, block>>>`. The grid specifies how many blocks of threads will be launched, and the block specifies how many threads will be within each block. For example:
```cpp
MyKernel<<<gridDim, blockDim>>>(args);
```
This triggers the execution of the kernel function on the GPU, where it runs in parallel across the grid of threads.

```cpp
__global__ void CompactKernel(const scalar_t* a, scalar_t* out, size_t size, CudaVec shape,
                              CudaVec strides, size_t offset)
```

**`__global__` functions**, also known as **kernels**, are executed in parallel by multiple GPU threads.

-   **`__global__`**: This keyword designates the function as a **CUDA kernel**, which runs on the GPU and is invoked from the host (CPU). Kernels are used for parallel computation, leveraging the GPU's ability to run thousands of threads concurrently.

**Kernels Execute on the GPU**: CUDA kernels marked with `__global__` run directly on the GPU. For these kernels to access data, it must first be transferred from the CPU's memory (host) to the GPU's memory (device).


**`Compact` function (CPU function)**:

-   The **`Compact` function** is a **host-side** (CPU) function. This function is executed by the CPU and is responsible for managing and launching the **CUDA kernel** on the GPU.
- Runs on the **CPU** (host).
- Launches the CUDA kernel `CompactKernel` to perform the actual computation on the GPU.

**`CompactKernel` (GPU function)**:

-   The **`CompactKernel`** is a **GPU-side** function, also known as a **CUDA kernel**. It is marked with the `__global__` keyword, which tells the CUDA runtime that this function will run on the GPU, and it is launched from the host (CPU).
- Runs on the **GPU** (device).
-   **`CompactKernel`** is the actual GPU function (kernel) that performs the computation in parallel on the GPU.

> **Performance**: On a CPU, compaction would be done sequentially, processing one element after another. This would be much slower, especially for large arrays. By using a CUDA kernel on the GPU, we can handle hundreds or thousands of elements simultaneously, drastically speeding up the process.

The **kernel** is used in the `Compact` function because it allows the task of compacting a non-contiguous array into a contiguous layout to be executed **in parallel** on the **GPU**, taking advantage of the GPU's high parallelism and memory bandwidth. This leads to **faster processing** of large datasets and efficient use of hardware resources compared to sequential processing on the CPU.

---
**Code Implementation**
```cpp
__global__ void EwiseSetitemKernel(const scalar_t* a, scalar_t* out, size_t size, CudaVec shape,
                              CudaVec strides, size_t offset) {
  
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;

  if (gid >= size) return;  // Check for out-of-bounds
    size_t out_offset = index_to_offset_cuda(strides, shape, gid, offset);
    out[out_offset] = a[gid];
}

void EwiseSetitem(const CudaArray& a, CudaArray* out, std::vector<int32_t> shape,
                  std::vector<int32_t> strides, size_t offset) {
  /**
   * Set items in a (non-compact) array using CUDA.  Yyou will most likely want to implement a
   * EwiseSetitemKernel() function, similar to those above, that will do the actual work.
   * 
   * Args:
   *   a: _compact_ array whose items will be written to out
   *   out: non-compact array whose items are to be written
   *   shape: shapes of each dimension for a and out
   *   strides: strides of the *out* array (not a, which has compact strides)
   *   offset: offset of the *out* array (not a, which has zero offset, being compact)
   */
  /// BEGIN SOLUTION
  CudaDims dim = CudaOneDim(out->size);
  EwiseSetitemKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, a.size, VecToCuda(shape),
                                         VecToCuda(strides), offset);
  /// END SOLUTION
}
```
### Explaination of `EwiseSetitem`
1.  **`EwiseSetitemKernel` (GPU kernel function)**: This is the CUDA kernel function that performs the element-wise set operation. It runs on the GPU and is responsible for copying elements from a compact input array `a` to a non-compact output array `out`.
    
    Key points:
    
    -   **`gid = blockIdx.x * blockDim.x + threadIdx.x;`**: This calculates the global thread index (`gid`) for each thread based on the current block and thread within the block. CUDA runs multiple threads in parallel, and each thread is responsible for processing one element.
    -   **`if (gid >= size) return;`**: If the thread index is greater than or equal to the size of the array, the thread exits early, ensuring no out-of-bounds memory access occurs.
    -   **`size_t out_offset = index_to_offset_cuda(strides, shape, gid, offset);`**: This computes the memory offset within the non-compact output array using the provided strides, shape, and offset. This is necessary because the output array is non-compact, meaning its elements may not be stored in consecutive memory locations.
    -   **`out[out_offset] = a[gid];`**: The value from the compact array `a` at index `gid` is copied to the calculated location `out_offset` in the non-compact output array `out`.
    
2.  **`EwiseSetitem` (Host-side function)**: This is the host (CPU-side) function that prepares and launches the CUDA kernel (`EwiseSetitemKernel`). It runs on the CPU and is responsible for setting up the grid and block dimensions and then calling the kernel to execute on the GPU.
    
    Key points:
    
    -   **`CudaDims dim = CudaOneDim(out->size);`**: This utility function calculates the number of blocks and threads needed to process all elements in the output array. The number of threads per block is usually a constant (e.g., 256), and the number of blocks is calculated based on the size of the output array.
    -   **`EwiseSetitemKernel<<<dim.grid, dim.block>>>(...);`**: This launches the kernel on the GPU, where each thread will process one element. The kernel is called with `dim.grid` blocks and `dim.block` threads per block.
    -   **Arguments to the kernel**:
        -   `a.ptr`: Pointer to the compact input array `a` on the GPU.
        -   `out->ptr`: Pointer to the non-compact output array `out` on the GPU.
        -   `a.size`: Total size of the input array `a`.
        -   `VecToCuda(shape)`: Converts the CPU-side shape vector to a `CudaVec` structure that can be passed to the GPU.
        -   `VecToCuda(strides)`: Converts the CPU-side strides vector to a `CudaVec` structure.
        -   `offset`: Offset for the non-compact output array `out`.

### Summary:

-   **Kernel function**: `EwiseSetitemKernel` runs on the GPU and copies values from a compact array `a` to a non-compact array `out`. It calculates the correct memory location for the output using strides and shape information.
-   **Host function**: `EwiseSetitem` runs on the CPU and sets up the kernel's execution by calculating the appropriate grid and block sizes, then launches the `EwiseSetitemKernel` on the GPU.

---
**Code Implementation**
```cpp
__global__ void ScalarSetitemKernel(scalar_t val, scalar_t* out, size_t size, CudaVec shape,
                              CudaVec strides, size_t offset) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;

  if (gid >= size) return;  // Check for out-of-bounds

   size_t out_offset = index_to_offset_cuda(strides, shape, gid, offset);
out[out_offset] = val;
}

void ScalarSetitem(size_t size, scalar_t val, CudaArray* out, std::vector<int32_t> shape,
                   std::vector<int32_t> strides, size_t offset) {
  /**
   * Set items is a (non-compact) array
   * 
   * Args:
   *   size: number of elements to write in out array (note that this will note be the same as
   *         out.size, because out is a non-compact subset array);  it _will_ be the same as the 
   *         product of items in shape, but covenient to just pass it here.
   *   val: scalar value to write to
   *   out: non-compact array whose items are to be written
   *   shape: shapes of each dimension of out
   *   strides: strides of the out array
   *   offset: offset of the out array
   */
  /// BEGIN SOLUTION
  CudaDims dim = CudaOneDim(size);
  ScalarSetitemKernel<<<dim.grid, dim.block>>>(val, out->ptr, size, VecToCuda(shape),
                                                 VecToCuda(strides), offset);
  /// END SOLUTION
}
```
### Explanation of `ScalarSetitem`

1.  **`ScalarSetitemKernel` (GPU Kernel Function)**: This CUDA kernel function is responsible for setting a scalar value (`val`) to specific locations in a non-compact output array `out`. The key challenge is that the output array `out` is non-compact, meaning the elements are not necessarily stored contiguously in memory. The kernel handles each element in parallel using GPU threads.
    
    **Key steps**:
    
    -   **Thread index (`gid`) calculation**:
        -   Each thread on the GPU computes its global index (`gid`) using `blockIdx.x * blockDim.x + threadIdx.x`. This index tells the thread which element it should process in parallel.
    -   **Out-of-bounds check**:
        -   The `if (gid >= size) return;` check ensures that if the thread index exceeds the array's size, it exits early to avoid accessing memory outside the bounds of the array.
    -   **Offset computation**:
        -   `size_t out_offset = index_to_offset_cuda(...)` calculates the exact memory location (offset) within the non-compact array `out` where the scalar value `val` should be written. This calculation is necessary because non-compact arrays have elements stored in non-consecutive memory locations, determined by their strides and shape.
    -   **Assigning the scalar value**:
        -   `out[out_offset] = val;` sets the scalar value `val` at the computed memory location `out_offset`.
2.  **`ScalarSetitem` (Host-Side Function)**: This is the host (CPU-side) function that prepares and launches the CUDA kernel. It handles the setup of grid and block sizes for the GPU threads and invokes the `ScalarSetitemKernel`.
    
    **Key steps**:
    
    -   **Grid and block dimensions**:
        -   `CudaDims dim = CudaOneDim(size);` computes the number of blocks and threads needed to process all elements in the output array. This helps maximize the parallelism by ensuring there are enough threads to handle the size of the output array.
    -   **Launching the kernel**:
        -   `ScalarSetitemKernel<<<dim.grid, dim.block>>>(...);` launches the GPU kernel with the specified grid and block sizes. The kernel then runs on the GPU, where each thread processes one element by setting the scalar value.
    -   **Arguments**:
        -   **`val`**: The scalar value that will be written to all specified locations in the array.
        -   **`out->ptr`**: A pointer to the non-compact output array `out` on the GPU.
        -   **`size`**: The total number of elements to write, typically calculated based on the product of the array's shape dimensions.
        -   **`VecToCuda(shape)` and `VecToCuda(strides)`**: These utility functions convert the CPU-side vectors for shape and strides into a format (`CudaVec`) that can be used by the GPU kernel.
        -   **`offset`**: The starting memory offset for the output array `out`.

### Summary:

-   The **kernel function** (`ScalarSetitemKernel`) runs on the GPU and assigns a scalar value to elements of a non-compact output array, calculating the correct memory location for each element based on the array's shape and strides.
-   The **host function** (`ScalarSetitem`) runs on the CPU and sets up the execution environment for the CUDA kernel by determining the appropriate grid and block sizes, and then launches the kernel to execute in parallel on the GPU.

This setup allows for efficient setting of values in non-contiguous memory regions of an array, leveraging the power of parallel computation on the GPU.
