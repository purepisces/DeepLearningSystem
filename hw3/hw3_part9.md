## Part 9: CUDA Backend - Matrix multiplication
  
Implement the following functions in `ndarray_backend_cuda.cu`:


* `Matmul()`

  
Finally, as your final exercise, you'll implement matrix multiplication on the GPU. Your implementation here can roughly follow the presentation in class. While you can pass the tests using fairly naive code here (i.e., you could just have a separate thread for each (i,j) location in the matrix, doing the matrix multiplication efficiently (to make it actually faster than a CPU version) requires cooperative fetching and the block shared memory register tiling covered in class. Try to implement using these methods, and see how much faster you can get your code than the C++ (or numpy) backends.


The Pseudocode in Class

```cpp
__global__ void mm(float A[N][N], float B[N][N], float C[N][N]) {
    __shared__ float sA[S][L], sB[S][L];
    float c[V][V] = {0};
    float a[V], b[V];
    int yblock = blockIdx.y;
    int xblock = blockIdx.x;

    for (int ko = 0; ko < N; ko += S) {
        __syncthreads();
        // needs to be implemented by thread cooperative fetching
        sA[:, :] = A[ko + S, yblock * L : yblock * L + L];
        sB[:, :] = B[ko + S, xblock * L : xblock * L + L];
        __syncthreads();

        for (int ki = 0; ki < S; ++ki) {
            a[:] = sA[ki, threadIdx.x * V + V];
            b[:] = sB[ki, threadIdx.x * V + V];
            for (int y = 0; y < V; ++y) {
                for (int x = 0; x < V; ++x) {
                    c[y][x] += a[y] * b[x];
                }
            }
        }
    }

    int ybase = blockIdx.y * blockDim.y + threadIdx.y;
    int xbase = blockIdx.x * blockDim.x + threadIdx.x;
    C[ybase * V : ybase * V + V, xbase * V : xbase * V + V] = c[:, :];
}
```


**Code Implementation**
```cpp
__global__ void MatmulKernel(const scalar_t* a, const scalar_t* b, scalar_t* c, uint32_t M, uint32_t N, uint32_t P) {
#define V 2

  // Get block and thread indices
  int block_x = blockIdx.x;  // Now block_x corresponds to columns of C (P)
  int block_y = blockIdx.y;  // Now block_y corresponds to rows of C (M)
  int thread_x = threadIdx.x;
  int thread_y = threadIdx.y;
  int thread_id = thread_x + thread_y * blockDim.x;
  int nthreads = blockDim.x * blockDim.y;

  // Shared memory for sub-matrices (tiles) of A and B
  __shared__ scalar_t a_shared[TILE][TILE];
  __shared__ scalar_t b_shared[TILE][TILE];

  // Registers for sub-block calculations
  scalar_t c_reg[V][V] = {0};  // Initialize output sub-block
  scalar_t a_reg[V] = {0};     // Temporary storage for row data from A
  scalar_t b_reg[V] = {0};     // Temporary storage for column data from B

  // Iterate over tiles of A and B
  for (int start = 0; start < N; start += TILE) {
    __syncthreads(); // Ensure all threads in the block finish the previous loop

    // Load tiles of A and B into shared memory, each thread loads one or more elements
    for (int idx = thread_id; idx < TILE * TILE; idx += nthreads) {
      int x = idx / TILE;  // Row index in the shared memory tile
      int y = idx % TILE;  // Column index in the shared memory tile

      // Load A tile from global memory to shared memory
      if (x + block_y * TILE < M && y + start < N) { // block_y now corresponds to rows of C
        a_shared[x][y] = a[(x + block_y * TILE) * N + y + start];  // Access elements of A using block_y for rows
      } else {
        a_shared[x][y] = 0.0f; // Out of bounds, set to 0
      }

      // Load B tile from global memory to shared memory
      if (x + start < N && y + block_x * TILE < P) { // block_x now corresponds to columns of C
        b_shared[x][y] = b[(x + start) * P + y + block_x * TILE];  // Access elements of B using block_x for columns
      } else {
        b_shared[x][y] = 0.0f; // Out of bounds, set to 0
      }
    }

    __syncthreads(); // Ensure all threads finish loading data to shared memory

    // Perform matrix multiplication on the loaded tiles
    int stripe_cnt = min(TILE, N - start); // Ensure we don't exceed matrix boundaries
    for (int stripe_i = 0; stripe_i < stripe_cnt; ++stripe_i) {
      if (thread_x * V < TILE && thread_y * V < TILE) {
        // Load row of A and column of B into registers for the current stripe
        for (int reg_x = 0; reg_x < V; ++reg_x) {
          int shared_x = thread_x * V + reg_x;
          if (shared_x < TILE) {
            a_reg[reg_x] = a_shared[shared_x][stripe_i];
          }
        }

        for (int reg_y = 0; reg_y < V; ++reg_y) {
          int shared_y = thread_y * V + reg_y;
          if (shared_y < TILE) {
            b_reg[reg_y] = b_shared[stripe_i][shared_y];
          }
        }

        // Compute the outer product and accumulate results in c_reg
        for (int i = 0; i < V; ++i) {
          for (int j = 0; j < V; ++j) {
            c_reg[i][j] += a_reg[i] * b_reg[j];
          }
        }
      }
    }

    __syncthreads(); // Ensure all threads finish computations for this tile
  }

  // Store the computed 2x2 sub-block from c_reg into global memory
  if (thread_x * V < TILE && thread_y * V < TILE) {
    for (int i = 0; i < V; ++i) {
      for (int j = 0; j < V; ++j) {
        int x = block_y * TILE + thread_x * V + i; // block_y is now for rows of C
        int y = block_x * TILE + thread_y * V + j; // block_x is now for columns of C
        if (x < M && y < P) {
          c[x * P + y] = c_reg[i][j];  // Store result in the output matrix
        }
      }
    }
  }
}


void Matmul(const CudaArray& a, const CudaArray& b, CudaArray* out, uint32_t M, uint32_t N,
            uint32_t P) {
  /**
   * Multiply two (compact) matrices into an output (also comapct) matrix.  You will want to look
   * at the lecture and notes on GPU-based linear algebra to see how to do this.  Since ultimately
   * mugrade is just evaluating correctness, you _can_ implement a version that simply parallelizes
   * over (i,j) entries in the output array.  However, to really get the full benefit of this
   * problem, we would encourage you to use cooperative fetching, shared memory register tiling, 
   * and other ideas covered in the class notes.  Note that unlike the tiled matmul function in
   * the CPU backend, here you should implement a single function that works across all size
   * matrices, whether or not they are a multiple of a tile size.  As with previous CUDA
   * implementations, this function here will largely just set up the kernel call, and you should
   * implement the logic in a separate MatmulKernel() call.
   * 
   *
   * Args:
   *   a: compact 2D array of size m x n
   *   b: comapct 2D array of size n x p
   *   out: compact 2D array of size m x p to write the output to
   *   M: rows of a / out
   *   N: columns of a / rows of b
   *   P: columns of b / out
   */

  /// BEGIN SOLUTION
  dim3 grid_dim = dim3((P + TILE - 1) / TILE, (M + TILE - 1) / TILE, 1);
  dim3 block_dim = dim3(2, 2, 1);
  MatmulKernel<<<grid_dim, block_dim>>>(a.ptr, b.ptr, out->ptr, M, N, P);
	
  /// END SOLUTION
}
```


```python
TILE = 4
N = 5
start = 0
stripe_cnt = min(TILE, N - start)
thread_x = 0
thread_y = 0
V = 2

# Initialize the shared memory for matrices A and B (for demonstration, these are 4x4 matrices)
a_shared = [[0 for _ in range(TILE)] for _ in range(TILE)]
b_shared = [[0 for _ in range(TILE)] for _ in range(TILE)]

# Initialize registers
a_reg = [0 for _ in range(V)]
b_reg = [0 for _ in range(V)]

# Initialize the output register (c_reg), which is a 2x2 block
c_reg = [[0 for _ in range(V)] for _ in range(V)]

# Loop over stripes
for stripe_i in range(stripe_cnt):
    if thread_x * V < TILE and thread_y * V < TILE:
        # Load row of A into registers for the current stripe
        for reg_x in range(V):
            shared_x = thread_x * V + reg_x
            if shared_x < TILE:
                a_reg[reg_x] = a_shared[shared_x][stripe_i]
                # For demonstration, let's print the values being loaded
                print(f"Loaded a_shared[{shared_x}][{stripe_i}] into a_reg[{reg_x}]")

        # Load column of B into registers for the current stripe
        for reg_y in range(V):
            shared_y = thread_y * V + reg_y
            if shared_y < TILE:
                b_reg[reg_y] = b_shared[stripe_i][shared_y]
                # For demonstration, let's print the values being loaded
                print(f"Loaded b_shared[{stripe_i}][{shared_y}] into b_reg[{reg_y}]")

        # Compute the outer product and accumulate results in c_reg
        for i in range(V):
            for j in range(V):
                c_reg[i][j] += a_reg[i] * b_reg[j]
                # For demonstration, print the computation details
                print(f"c_reg[{i}][{j}] += a_reg[{i}] * b_reg[{j}] => {c_reg[i][j]}")

# For demonstration, output the final c_reg values
print("Final c_reg values:")
for row in c_reg:
    print(row)
```
```css
Loaded a_shared[0][0] into a_reg[0]
Loaded a_shared[1][0] into a_reg[1]
Loaded b_shared[0][0] into b_reg[0]
Loaded b_shared[0][1] into b_reg[1]
c_reg[0][0] += a_reg[0] * b_reg[0] => 0
c_reg[0][1] += a_reg[0] * b_reg[1] => 0
c_reg[1][0] += a_reg[1] * b_reg[0] => 0
c_reg[1][1] += a_reg[1] * b_reg[1] => 0
Loaded a_shared[0][1] into a_reg[0]
Loaded a_shared[1][1] into a_reg[1]
Loaded b_shared[1][0] into b_reg[0]
Loaded b_shared[1][1] into b_reg[1]
c_reg[0][0] += a_reg[0] * b_reg[0] => 0
c_reg[0][1] += a_reg[0] * b_reg[1] => 0
c_reg[1][0] += a_reg[1] * b_reg[0] => 0
c_reg[1][1] += a_reg[1] * b_reg[1] => 0
Loaded a_shared[0][2] into a_reg[0]
Loaded a_shared[1][2] into a_reg[1]
Loaded b_shared[2][0] into b_reg[0]
Loaded b_shared[2][1] into b_reg[1]
c_reg[0][0] += a_reg[0] * b_reg[0] => 0
c_reg[0][1] += a_reg[0] * b_reg[1] => 0
c_reg[1][0] += a_reg[1] * b_reg[0] => 0
c_reg[1][1] += a_reg[1] * b_reg[1] => 0
Loaded a_shared[0][3] into a_reg[0]
Loaded a_shared[1][3] into a_reg[1]
Loaded b_shared[3][0] into b_reg[0]
Loaded b_shared[3][1] into b_reg[1]
c_reg[0][0] += a_reg[0] * b_reg[0] => 0
c_reg[0][1] += a_reg[0] * b_reg[1] => 0
c_reg[1][0] += a_reg[1] * b_reg[0] => 0
c_reg[1][1] += a_reg[1] * b_reg[1] => 0
Final c_reg values:
[0, 0]
[0, 0]
```

```python
# Define necessary variables
TILE = 4
V = 2
M = 9  # Number of rows in matrix C
P = 8  # Number of columns in matrix C

block_x = 1  # Example block column index
block_y = 0  # Example block row index

thread_x = 0  # Example thread row index in block
thread_y = 0  # Example thread column index in block

# Initialize c_reg (2x2 matrix for each thread)
c_reg = [[0 for _ in range(V)] for _ in range(V)]

# Initialize the global memory matrix 'c' with zeros (M x P)
c = [[0 for _ in range(P)] for _ in range(M)]

# Loop over the elements of the 2x2 block
for i in range(V):
    for j in range(V):
        # Calculate global indices (x and y) in matrix C
        x = block_y * TILE + thread_x * V + i  # Row index in C
        y = block_x * TILE + thread_y * V + j  # Column index in C

        # Check for out-of-bounds and write to matrix C if valid
        if x < M and y < P:
            c[x][y] = c_reg[i][j]  # Store the result in the global matrix C
            print(f"Loaded c_reg[{i}][{j}] into c[{x}][{y}]")      
```
```css
Loaded c_reg[0][0] into c[0][4]
Loaded c_reg[0][1] into c[0][5]
Loaded c_reg[1][0] into c[1][4]
Loaded c_reg[1][1] into c[1][5]
```

### Explain Kernel Launch:
When you call `Matmul`, it launches the CUDA kernel `MatmulKernel` with a **grid of blocks**, where each block contains a **2D array of threads**. This doesn't just call the kernel for one thread; instead, it invokes the kernel across all the **threads in all the blocks** simultaneously (or in parallel, depending on hardware).

The `MatmulKernel` function is **not** called for just a single thread (e.g., with `threadIdx.x = 0`, `threadIdx.y = 0`). It’s called for **all threads** in all blocks simultaneously, and each thread operates on its own portion of the matrices `A`, `B`, and `C`. This parallel execution is what makes CUDA so powerful for matrix multiplication and other large-scale tasks.

### Explain tiling strategy in CUDA

Lets' visualize the process of multiplying matrices using the tiling strategy in CUDA

**Example:**

Let's assume:

-   Matrix `A` is of size `8 x 6` (i.e., `M = 8`, `N = 6`).
-   Matrix `B` is of size `6 x 8` (i.e., `N = 6`, `P = 8`).
-   Matrix `C` will be of size `8 x 8` (i.e., `M = 8`, `P = 8`).

The tile size (`TILE`) is `4`. So, each block processes a `4 x 4` submatrix (tile) of the final matrix `C`.

### Step 1: Breaking `A`, `B`, and `C` into Tiles

#### Matrix `A` (8 x 6):

We will break matrix `A` into two horizontal tiles because its dimensions are `M x N = 8 x 6`. Since we are using a tile size of `TILE = 4`, we will have:

-   2 tiles along the row (because `M = 8`, and `TILE = 4` means 2 tiles).
-   2 tiles along the shared dimension `N` (because `N = 6`, but `TILE = 4` covers this in 2 steps).

**Visualizing Tiles in Matrix `A`:**
```css
A (8x6):
------------------------------------
|   Tile 1 (4x4)   |  Tile 3 (4x2) |
|------------------|---------------|
|   Tile 2 (4x4)   |  Tile 4 (4x2) |
------------------------------------
```

-   `Tile 1`: Top-left 4x4 block.
-   `Tile 2`: Bottom-left 4x4 block.
-   `Tile 3`: Top-right 4x2 block (due to N=6).
-   `Tile 4`: Bottom-right 4x2 block.

#### Matrix `B` (6 x 8):

Matrix `B` will also be broken into tiles, but along different dimensions:

-   2 tiles along the column (`P = 8`, and `TILE = 4` means 2 tiles).
-   2 tiles along the shared dimension `N` (since `N = 6`, covered in 2 steps).

**Visualizing Tiles in Matrix `B`:**
```css
B (6x8):
------------------------------------
|   Tile 1 (4x4)   |  Tile 2 (4x4) |
|------------------|---------------|
|   Tile 3 (2x4)   |  Tile 4 (2x4) |
------------------------------------
```
-   `Tile 1`: Top-left 4x4 block.
-   `Tile 2`: Top-right 4x4 block.
-   `Tile 3`: Bottom-left 2x4 block (due to N=6).
-   `Tile 4`: Bottom-right 2x4 block.

#### Matrix `C` (8 x 8):

Matrix `C` is the result of multiplying `A` and `B`, and it will be broken into 4 blocks:

**Visualizing Tiles in Matrix `C`:**

```css
C (8x8):
------------------------------------
|   Tile 1 (4x4)   |  Tile 2 (4x4) |
|------------------|---------------|
|   Tile 3 (4x4)   |  Tile 4 (4x4) |
------------------------------------
```
### Step 2: Processing Tiles in Matrix Multiplication

#### Tile 1 of `C`:

To compute **Tile 1 (4x4)** of matrix `C`:

-   Multiply **Tile 1 (4x4)** of `A` with **Tile 1 (4x4)** of `B`.
-   Multiply **Tile 3 (4x2)** of `A` with **Tile 3 (2x4)** of `B`.
-   Sum the results to form **Tile 1** of `C`.

This happens in the following steps:

1.  The block of threads loads **Tile 1 of `A`** and **Tile 1 of `B`** into shared memory.
2.  The threads compute the partial product of these tiles and store the intermediate result.
3.  The block of threads then loads **Tile 3 of `A`** and **Tile 3 of `B`** into shared memory.
4.  The threads compute the partial product of these tiles and add it to the intermediate result to form the final result for **Tile 1 of `C`**.

#### Tile 2 of `C`:

To compute **Tile 2 (4x4)** of matrix `C`:

-   Multiply **Tile 1 (4x4)** of `A` with **Tile 2 (4x4)** of `B`.
-   Multiply **Tile 3 (4x2)** of `A` with **Tile 4 (2x4)** of `B`.
-   Sum the results to form **Tile 2** of `C`.

#### Tile 3 of `C`:

To compute **Tile 3 (4x4)** of matrix `C`:

-   Multiply **Tile 2 (4x4)** of `A` with **Tile 1 (4x4)** of `B`.
-   Multiply **Tile 4 (4x2)** of `A` with **Tile 3 (2x4)** of `B`.
-   Sum the results to form **Tile 3** of `C`.

#### Tile 4 of `C`:

To compute **Tile 4 (4x4)** of matrix `C`:

-   Multiply **Tile 2 (4x4)** of `A` with **Tile 2 (4x4)** of `B`.
-   Multiply **Tile 4 (4x2)** of `A` with **Tile 4 (2x4)** of `B`.
-   Sum the results to form **Tile 4** of `C`.

### Step 3: Accumulating the Partial Products

For each tile of `C`, we accumulate the results of several partial products (multiplying smaller tiles from `A` and `B`) over the shared dimension `N`. This is why the code iterates over the dimension `N` and processes multiple tiles of `A` and `B` to compute a single tile of `C`.

#### Why It Works

Even though we're using tiles, we’re not trying to directly multiply two `4x4` matrices as you initially thought. Instead, each thread block works on a `4x4` submatrix of `C` and accumulates partial results by iterating over the dimension `N` and processing multiple tiles from `A` and `B`.

**Why We Need Multiple Tiles**

Example with Numbers:

Let's break down a specific case where you want to compute `C[0,0]`:

-   `C[0,0] = A[0,0] * B[0,0] + A[0,1] * B[1,0] + A[0,2] * B[2,0] + A[0,3] * B[3,0] + A[0,4] * B[4,0] + A[0,5] * B[5,0]`

You can see that the dot product involves all 6 elements from row `0` of `A` and column `0` of `B`.

##### How Tiling Handles This:

-   The first four terms (`A[0,0] * B[0,0]` to `A[0,3] * B[3,0]`) are computed by multiplying **Tile 1 of `A`** with **Tile 1 of `B`**.
-   The last two terms (`A[0,4] * B[4,0]` and `A[0,5] * B[5,0]`) are computed by multiplying **Tile 3 of `A`** with **Tile 3 of `B`**.

#### Summary

-   A block computes a `4x4` tile of the result matrix `C`.
-   To fully compute that tile, the block needs to iterate over the shared dimension `N`, loading and multiplying tiles from `A` and `B`, and accumulating the results.
-   The tile size (`TILE = 4`) just determines how much data each block of threads processes at a time, but the entire matrix multiplication is achieved by accumulating results from multiple partial products.


### Explain `thread_id = thread_x + thread_y * blockDim.x`

### Visualization Example:

Let’s consider a block of size `blockDim.x = 4` and `blockDim.y = 3`. This gives us a 2D grid of threads in a block like this:

```css
(0,0)  (1,0)  (2,0)  (3,0)
(0,1)  (1,1)  (2,1)  (3,1)
(0,2)  (1,2)  (2,2)  (3,2)
```


In this example:

- `blockDim.x = 4` (4 threads along the x-axis, i.e., columns)
- `blockDim.y = 3` (3 threads along the y-axis, i.e., rows)

Now, let’s compute the linear thread IDs for all the threads using the formula `thread_id = thread_x + thread_y * blockDim.x`:

| `thread_x` | `thread_y` | Computation (x + y * blockDim.x) | `thread_id` |
|------------|------------|-----------------------------------|-------------|
| 0          | 0          | 0 + 0 * 4 = 0                    | 0           |
| 1          | 0          | 1 + 0 * 4 = 1                    | 1           |
| 2          | 0          | 2 + 0 * 4 = 2                    | 2           |
| 3          | 0          | 3 + 0 * 4 = 3                    | 3           |
| 0          | 1          | 0 + 1 * 4 = 4                    | 4           |
| 1          | 1          | 1 + 1 * 4 = 5                    | 5           |
| 2          | 1          | 2 + 1 * 4 = 6                    | 6           |
| 3          | 1          | 3 + 1 * 4 = 7                    | 7           |
| 0          | 2          | 0 + 2 * 4 = 8                    | 8           |
| 1          | 2          | 1 + 2 * 4 = 9                    | 9           |
| 2          | 2          | 2 + 2 * 4 = 10                   | 10          |
| 3          | 2          | 3 + 2 * 4 = 11                   | 11          |

As you can see, the threads are indexed in a row-major order. The linear thread IDs start from `0` at `(0, 0)` and increment across the **x-axis** first (the row), before moving to the next row along the **y-axis**.

**Another Example**
Let’s visualize the full grid of blocks and the `thread_id`s within each block:
```css
Grid (2x2 blocks):
+-----------------------+-----------------------+
| Block 0 (thread_id)    | Block 1 (thread_id)   |
| +---+---+---+          | +---+---+---+         |
| | 0 | 1 | 2 |          | | 0 | 1 | 2 |         |
| +---+---+---+          | +---+---+---+         |
| | 3 | 4 | 5 |          | | 3 | 4 | 5 |         |
| +---+---+---+          | +---+---+---+         |
+-----------------------+-----------------------+
| Block 2 (thread_id)    | Block 3 (thread_id)   |
| +---+---+---+          | +---+---+---+         |
| | 0 | 1 | 2 |          | | 0 | 1 | 2 |         |
| +---+---+---+          | +---+---+---+         |
| | 3 | 4 | 5 |          | | 3 | 4 | 5 |         |
| +---+---+---+          | +---+---+---+         |
+-----------------------+-----------------------+
```
### Explain Shared memory

**Shared memory** in CUDA is a type of on-chip memory that is shared among all the threads within a block. It is a key component for optimizing performance on the GPU because it is much faster than **global memory** (which is off-chip and has higher latency). Shared memory can be used by threads within the same block to communicate and share data efficiently during computations.

### Explain `__shared__ scalar_t a_shared[TILE][TILE];`

In CUDA, `__shared__` is used to declare **shared memory**, a special type of memory that is shared by all threads within a **block**. Shared memory is much faster to access than global memory because it resides on the chip (GPU), and it can be accessed by all threads in the block. This makes it an excellent choice for storing intermediate data that is frequently accessed and updated by multiple threads within the same block.
```cpp
__shared__ scalar_t a_shared[TILE][TILE];
```

This line declares a 2D array named `a_shared` in **shared memory**:

-   `scalar_t`: This is the data type of the elements in the array. 
-   `TILE`: This is a predefined size (for example, `TILE = 4`). The array size is `TILE x TILE`, meaning it is a 2D square matrix.

In matrix multiplication, `a_shared` is used to temporarily store a tile (a small sub-matrix) of the matrix `A` in **shared memory** for use by all threads in the block.

### Expalin `c_reg[V][V]`, `a_reg[V]`, and `b_reg[V]`





### Explain 


```css
Tile (4x4):

|  0   |  1   |  2   |  3   |
|  4   |  5   |  6   |  7   |
|  8   |  9   | 10   | 11   |
| 12   | 13   | 14   | 15   |

Thread 0 (thread_id = 0) loads elements: 0, 4, 8, 12
Thread 1 (thread_id = 1) loads elements: 1, 5, 9, 13
Thread 2 (thread_id = 2) loads elements: 2, 6, 10, 14
Thread 3 (thread_id = 3) loads elements: 3, 7, 11, 15
```


