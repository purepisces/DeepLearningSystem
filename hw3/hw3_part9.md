```cpp
__global__ void MatmulKernel(const scalar_t* a, const scalar_t* b, scalar_t* c, uint32_t M, uint32_t N, uint32_t P) {
#define V 2
#define TILE 4

  // Get block and thread indices
  int block_x = blockIdx.x;
  int block_y = blockIdx.y;
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
      if (x + block_x * TILE < M && y + start < N) {
        a_shared[x][y] = a[(x + block_x * TILE) * N + y + start];
      } else {
        a_shared[x][y] = 0.0f; // Out of bounds, set to 0
      }

      // Load B tile from global memory to shared memory
      if (x + start < N && y + block_y * TILE < P) {
        b_shared[x][y] = b[(x + start) * P + y + block_y * TILE];
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
        int x = block_x * TILE + thread_x * V + i;
        int y = block_y * TILE + thread_y * V + j;
        if (x < M && y < P) {
          c[x * P + y] = c_reg[i][j];  // Store result in the output matrix
        }
      }
    }
  }
}
```
<img src="CUDA_Grid.png" alt="CUDA_Grid" width="600" height="500"/>

## Reference

- CUDA_Grid.png from https://www.microway.com/hpc-tech-tips/cuda-parallel-thread-management/





