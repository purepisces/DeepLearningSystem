## Part 8: CUDA Backend - Reductions

Implement the following functions in `ndarray_backend_cuda.cu`:


* `ReduceMax()`

* `ReduceSum()`


You can take a fairly simplistic approach here, and just use a separate CUDA thread for each individual reduction item: i.e., if there is a 100 x 20 array you are reducing over the second dimension, you could have 100 threads, each of which individually processed its own 20-dimensional array.. This is particularly inefficient for the `.max(axis=None)` calls, but we won't worry about this for the time being. If you want a more industrial-grade implementation, you use a hierarchical mechanism that first aggregated across some smaller span, then had a secondary function that aggregated across _these_ reduced arrays, etc. But this is not needed to pass the tests.

**Code Implementation**
```cpp
__global__ void ReduceMaxKernel(const scalar_t* a, scalar_t* out, size_t reduce_size, size_t out_size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  
  if (gid < out_size) {
    scalar_t max_val = a[gid * reduce_size];
    for (size_t j = 1; j < reduce_size; j++) {
      max_val = max(max_val, a[gid * reduce_size + j]);
    }
    out[gid] = max_val;
  }
}

void ReduceMax(const CudaArray& a, CudaArray* out, size_t reduce_size) {
  /**
   * Reduce by taking maximum over `reduce_size` contiguous blocks.  Even though it is inefficient,
   * for simplicity you can perform each reduction in a single CUDA thread.
   * 
   * Args:
   *   a: compact array of size a.size = out.size * reduce_size to reduce over
   *   out: compact array to write into
   *   redice_size: size of the dimension to reduce over
   */
  /// BEGIN SOLUTION
  CudaDims dim = CudaOneDim(out->size);
  ReduceMaxKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, reduce_size, out->size);
  /// END SOLUTION
}

__global__ void ReduceSumKernel(const scalar_t* a, scalar_t* out, size_t reduce_size, size_t out_size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  
  if (gid < out_size) {
    scalar_t sum_val = 0;
    for (size_t j = 0; j < reduce_size; j++) {
      sum_val += a[gid * reduce_size + j];
    }
    out[gid] = sum_val;
  }
}

void ReduceSum(const CudaArray& a, CudaArray* out, size_t reduce_size) {
  /**
   * Reduce by taking summation over `reduce_size` contiguous blocks.  Again, for simplicity you 
   * can perform each reduction in a single CUDA thread.
   * 
   * Args:
   *   a: compact array of size a.size = out.size * reduce_size to reduce over
   *   out: compact array to write into
   *   redice_size: size of the dimension to reduce over
   */
  /// BEGIN SOLUTION
  CudaDims dim = CudaOneDim(out->size);
  ReduceSumKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, reduce_size, out->size);
  /// END SOLUTION
}
```
### Explain 100 x 20 Scenario in question
In the statement, the dimensions of the array are referred to as `100 x 20`, meaning the array has 100 rows and 20 columns. The reference to reducing over the "second dimension" means reducing along the columns (in this case, over the 20 elements in each row).

-   **Array Shape**: The `100 x 20` array means:
    -   There are 100 rows (the first dimension).
    -   There are 20 columns (the second dimension).
-   **Reduction Over the Second Dimension**: When the statement refers to reducing over the second dimension, it means that each of the 100 rows is being reduced individually by performing an operation (like sum or max) over the 20 elements in that row. This results in reducing the 20 elements in each row to a single element.

#### What This Means

For a `100 x 20` array:

-   **Input Array**: Size is `100 x 20 = 2000` elements.
-   **Output Array**: Size is `100` (because the reduction collapses the 20 elements in each row into a single value). The output would have one value for each row.

To implement this in CUDA, the approach described in the prompt suggests:

-   You use **one thread per row** of the array.
-   Each thread processes all 20 elements in the corresponding row (performing the sum, max, or whatever operation is required).

#### Why This Approach Might Be Inefficient

This approach could be inefficient for several reasons:

1.  **Single-threaded per Row**: Each thread processes all elements of the row by itself. In more optimized approaches, you would use multiple threads (or even multiple blocks) to process different parts of a row, reducing the time required for the reduction operation.
    
2.  **Sequential Processing**: Since each thread handles the entire row sequentially, there's no parallelism within the row itself, which CUDA can handle efficiently if designed properly.
    
3.  **Unused Parallelism**: CUDA's strength comes from using many threads simultaneously to perform calculations. By using just one thread per row, you're not fully utilizing the available parallelism of the GPU.
    

#### Better Approaches

In a more optimized approach:

-   **Hierarchical Reduction**: You would divide the work within each row across multiple threads. Threads could process smaller chunks of the row (e.g., 5 threads each processing 4 elements), and then combine their results in a final step.
-   **Shared Memory**: You could use CUDA’s shared memory to store intermediate results, further speeding up the reduction process within each block of threads.

#### In Summary

-   The **"100 x 20"** refers to an array with 100 rows and 20 columns.
-   The **"reduction over the second dimension"** means reducing the 20 elements in each row to a single value, leading to an output array of size 100.
-   The prompt suggests a simple approach where one CUDA thread processes one row, which works but is inefficient because it doesn’t fully utilize the parallel processing capabilities of the GPU. More optimized solutions would use multiple threads per row.

### Explaination of `ReduceMax`

This code implements a CUDA kernel and a function for performing **max reduction** over contiguous blocks of an array. Here's a detailed explanation of both the kernel and the wrapper function.

1. **CUDA Kernel: `ReduceMaxKernel`**

The kernel performs the actual reduction operation. It calculates the maximum value over `reduce_size` elements in each block and stores the result in the output array.

#### Key Elements:
```cpp
__global__ void ReduceMaxKernel(const scalar_t* a, scalar_t* out, size_t reduce_size, size_t out_size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  
  if (gid < out_size) {
    scalar_t max_val = a[gid * reduce_size];
    for (size_t j = 1; j < reduce_size; j++) {
      max_val = max(max_val, a[gid * reduce_size + j]);
    }
    out[gid] = max_val;
  }
}
```

#### Explanation:

-   **Input Parameters**:
    
    -   `a`: A pointer to the input array stored on the GPU.
    -   `out`: A pointer to the output array where results are stored.
    -   `reduce_size`: The size of the block of elements to reduce over (i.e., how many elements per block the kernel should compute the max for).
    -   `out_size`: The size of the output array (number of blocks that the kernel will process).
-   **Thread Indexing**:
    
    -   `size_t gid = blockIdx.x * blockDim.x + threadIdx.x;`: This computes the global thread index (`gid`), which uniquely identifies each thread in the grid of threads launched.
-   **Reduction Logic**:
    
    -   Each thread is responsible for reducing a contiguous block of `reduce_size` elements.
    -   `scalar_t max_val = a[gid * reduce_size];`: Each thread initializes the maximum value to the first element in the block assigned to it.
    -   The loop `for (size_t j = 1; j < reduce_size; j++)` goes through the rest of the elements in the block and updates `max_val` if a larger value is found using `max()`.
    -   After the loop, the result is written to the output array: `out[gid] = max_val;`.
-   **Thread Bounds Check**:
    
    -   `if (gid < out_size)`: Ensures that only valid threads (those within the range of the output size) perform computations. Threads with a `gid` greater than `out_size` do nothing.


2. **Host Function: `ReduceMax`**

This function is called from the host (CPU side) and launches the CUDA kernel. It calculates how many threads and blocks are needed and then calls the `ReduceMaxKernel`.

#### Key Elements:
```cpp
void ReduceMax(const CudaArray& a, CudaArray* out, size_t reduce_size) {
  CudaDims dim = CudaOneDim(out->size);
  ReduceMaxKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, reduce_size, out->size);
}
```
#### Explanation:

-   **Input Parameters**:
    
    -   `a`: The input array (`CudaArray`) containing the data to reduce. This array is in compact form, meaning all data is stored contiguously.
    -   `out`: The output array (`CudaArray`) where the results will be stored. The size of `out` is smaller than `a`, and it has one value per block after reduction.
    -   `reduce_size`: The number of elements to reduce in each block (i.e., the size of the contiguous block).
-   **Compute Thread Layout**:
    
    -   `CudaDims dim = CudaOneDim(out->size);`: This function calculates the grid and block size (number of threads and blocks) based on the size of the output array (`out->size`). This ensures that one thread is assigned to compute the maximum for each block of `reduce_size` elements.
    -   `CudaOneDim(out->size)` sets the number of blocks and threads so that each thread handles one block of `reduce_size` elements.
-   **Kernel Launch**:
    
    -   `ReduceMaxKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, reduce_size, out->size);`: This line launches the `ReduceMaxKernel` with the appropriate grid and block dimensions. It passes the input and output array pointers, the `reduce_size`, and the `out_size` to the kernel.

#### How It Works:

1.  **Input Array Structure**:
    
    -   The input array `a` is divided into blocks of size `reduce_size`. Each thread processes one block and reduces it to a single value by taking the maximum.
2.  **Output Array**:
    
    -   The output array `out` has a size of `a.size / reduce_size` because each block is reduced to a single value.
3.  **Example**:
    
    -   If the input array `a` has 1000 elements and `reduce_size = 10`, the kernel reduces each block of 10 elements to a single maximum value. The output array `out` will contain 100 elements (one for each block).

#### Inefficiency:

-   **Single-threaded per block**: As mentioned in the prompt, each thread processes an entire block sequentially. This can be inefficient because CUDA is optimized for parallel computation across many threads, and this approach doesn't fully exploit the GPU's parallel capabilities within each block. More optimized implementations would involve using multiple threads per block and shared memory for better performance.

### Explaination of `ReduceSum`

This code implements a CUDA kernel and a corresponding function to perform **sum reduction** over contiguous blocks of an array, similar to how the max reduction was implemented. The kernel sums up values within blocks of a given size (`reduce_size`), and each CUDA thread handles one block of elements.

1. **CUDA Kernel: `ReduceSumKernel`**

The kernel performs the actual sum reduction operation for each block of elements.

#### Key Elements:

```cpp
__global__ void ReduceSumKernel(const scalar_t* a, scalar_t* out, size_t reduce_size, size_t out_size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  
  if (gid < out_size) {
    scalar_t sum_val = 0;
    for (size_t j = 0; j < reduce_size; j++) {
      sum_val += a[gid * reduce_size + j];
    }
    out[gid] = sum_val;
  }
}
```
#### Explanation:

-   **Input Parameters**:
    
    -   `a`: A pointer to the input array (stored on the GPU).
    -   `out`: A pointer to the output array (also on the GPU) where the results of the sum reductions will be stored.
    -   `reduce_size`: The number of contiguous elements in each block that will be summed.
    -   `out_size`: The size of the output array, i.e., the number of blocks that will be reduced to single values.
-   **Thread Indexing**:
    
    -   `size_t gid = blockIdx.x * blockDim.x + threadIdx.x;`: This computes the global thread index (`gid`), which uniquely identifies each thread in the grid. Each thread is responsible for reducing one block.
-   **Sum Reduction Logic**:
    
    -   Each thread initializes `sum_val` to 0 and sums over `reduce_size` contiguous elements from the input array `a`.
    -   The loop `for (size_t j = 0; j < reduce_size; j++)` adds each element in the block to `sum_val`.
    -   After summing all the elements in the block, the result is written to the output array: `out[gid] = sum_val;`.
-   **Thread Bounds Check**:
    
    -   `if (gid < out_size)`: Ensures that only valid threads (i.e., those whose `gid` is within the output size) perform computations. This prevents out-of-bounds memory access if there are more threads than required.

2. **Host Function: `ReduceSum`**

This function is called from the host (CPU side) and is responsible for setting up the grid and block dimensions before launching the CUDA kernel.

#### Key Elements:
```cpp
void ReduceSum(const CudaArray& a, CudaArray* out, size_t reduce_size) {
  CudaDims dim = CudaOneDim(out->size);
  ReduceSumKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, reduce_size, out->size);
}
```
#### Explanation:

-   **Input Parameters**:
    
    -   `a`: The input array (`CudaArray`), which contains the data to be reduced. The input array is in compact form, meaning it is stored contiguously.
    -   `out`: The output array (`CudaArray`), which will store the results after reduction. Its size is smaller than the input array because it contains one reduced value per block of size `reduce_size`.
    -   `reduce_size`: The number of contiguous elements to sum over in each block (i.e., how many elements are being reduced into one).
-   **Grid and Block Dimensions**:
    
    -   `CudaDims dim = CudaOneDim(out->size);`: This line computes the grid and block dimensions to ensure that there are enough threads to cover the entire output array (`out->size`). This ensures that each thread is assigned one block of `reduce_size` elements to reduce.
-   **Kernel Launch**:
    
    -   `ReduceSumKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, reduce_size, out->size);`: This launches the `ReduceSumKernel` on the GPU with the calculated number of blocks and threads. The kernel reduces each block of `reduce_size` elements in `a` to a single value and stores the result in `out`.

#### How It Works:

1.  **Input Array Structure**:
    
    -   The input array `a` is divided into blocks of size `reduce_size`. Each thread is responsible for computing the sum of one block.
2.  **Output Array**:
    
    -   The output array `out` has a size of `a.size / reduce_size` because each block of `reduce_size` elements is reduced to a single value.
3.  **Example**:
    
    -   Suppose the input array `a` contains 1000 elements and `reduce_size = 10`. The kernel reduces each block of 10 elements into a single sum. The output array `out` will contain 100 elements (one for each block).

#### Summary:

-   **Reduction Operation**: The kernel computes the sum (or maximum) over `reduce_size` contiguous elements for each row of the input. This is done for each block of data elements independently.
    
-   **Thread Assignment**: In the current implementation, each thread is responsible for reducing a single row, processing all `reduce_size` elements in that row and storing the result in the output array.
    
-   **Grid and Block Dimensions**: The function `CudaOneDim(out->size)` determines the number of threads and blocks necessary to ensure each thread processes one row of data. Typically, the grid is designed to launch enough threads to cover all rows, while the block size (`BASE_THREAD_NUM`) defines the number of threads per block.
    
-   **Efficiency**: In this implementation, **each thread processes the `reduce_size` elements of one row sequentially**. While this approach is straightforward, it does not fully leverage the parallel processing capabilities of CUDA, as multiple threads in the same block are not collaborating. A more efficient implementation would involve using **multiple threads to reduce each row in parallel**, allowing for faster reductions by distributing the work across threads within the same block and using **shared memory** for intermediate results.
    

This simple solution works but can be improved for performance, especially for larger datasets or when `reduce_size` is large.

### Explanation of Key Issue: **Inefficient Use of Threads**

In current implementation, we launch one thread per row.

For example when `reduce_size` = 20:

-   **Thread 1**: Processes all 20 elements of Row 1 sequentially.
-   **Thread 2**: Processes all 20 elements of Row 2 sequentially.
-   **Thread 3**: Processes all 20 elements of Row 3 sequentially.
- ...

-   **Sequential Processing within Threads**: Each thread processes the `reduce_size` elements sequentially for its assigned row. This is inefficient because CUDA is optimized for parallelism, and sequential processing within a single thread underutilizes the GPU's potential.
    
-   **Lack of Cooperation Between Threads in a Block**: While multiple threads exist in each block, they are not working together. In this implementation, each thread reduces an entire row independently, without collaborating with other threads in the same block. CUDA performs best when threads in a block cooperate, such as through **shared memory**, to reduce the number of sequential operations per thread.
    
#### Idle Threads and Underutilization:

-   **Idle Threads in Each Block**: If your array has fewer rows than the number of threads in a block (for example, a 100-row array with a block size of 256 threads), **only the first 100 threads** will be active, while **Threads 100 to 255 will remain idle** because there aren't enough rows for them to process. These idle threads contribute nothing to the computation, leading to poor thread utilization.

#### Optimized Approach for Performance:

A more efficient solution would involve:

1.  **Parallelizing the Reduction Within a Block**: Instead of assigning one thread per row, assign multiple threads to work together on reducing each row in parallel. This allows threads to collaborate and share the workload, which would make better use of CUDA’s parallelism.
    
2.  **Using Shared Memory**: Utilize shared memory within a block to store intermediate results, allowing threads to combine their results efficiently. This reduces global memory access latency and speeds up the reduction process.
    
By utilizing all available threads within a block and introducing thread cooperation, the overall performance of the reduction operation can be significantly improved, particularly for larger datasets or cases where `reduce_size` is large.
