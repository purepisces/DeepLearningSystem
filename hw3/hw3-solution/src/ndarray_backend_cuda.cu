#include <cuda_runtime.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <iostream>
#include <sstream>

namespace needle {
namespace cuda {

#define BASE_THREAD_NUM 256

#define TILE 4
typedef float scalar_t;
const size_t ELEM_SIZE = sizeof(scalar_t);

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

////////////////////////////////////////////////////////////////////////////////
// Fill call
////////////////////////////////////////////////////////////////////////////////

__global__ void FillKernel(scalar_t* out, scalar_t val, size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = val;
}

void Fill(CudaArray* out, scalar_t val) {
  CudaDims dim = CudaOneDim(out->size);
  FillKernel<<<dim.grid, dim.block>>>(out->ptr, val, out->size);
}

////////////////////////////////////////////////////////////////////////////////
// Compact and setitem cals
////////////////////////////////////////////////////////////////////////////////

// Untility function to convert contiguous index i to memory location from strides

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

////////////////////////////////////////////////////////////////////////////////
// Elementwise and scalar operations
////////////////////////////////////////////////////////////////////////////////

__global__ void EwiseAddKernel(const scalar_t* a, const scalar_t* b, scalar_t* out, size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = a[gid] + b[gid];
}

void EwiseAdd(const CudaArray& a, const CudaArray& b, CudaArray* out) {
  /**
   * Add together two CUDA array
   */
  CudaDims dim = CudaOneDim(out->size);
  EwiseAddKernel<<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, out->size);
}

__global__ void ScalarAddKernel(const scalar_t* a, scalar_t val, scalar_t* out, size_t size) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < size) out[gid] = a[gid] + val;
}

void ScalarAdd(const CudaArray& a, scalar_t val, CudaArray* out) {
  /**
   * Add together a CUDA array and a scalar value.
   */
  CudaDims dim = CudaOneDim(out->size);
  ScalarAddKernel<<<dim.grid, dim.block>>>(a.ptr, val, out->ptr, out->size);
}

/**
 * In the code the follows, use the above template to create analogous elementise
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

////////////////////////////////////////////////////////////////////////////////
// Elementwise and scalar operations
////////////////////////////////////////////////////////////////////////////////

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
        // Load row of A into registers for the current stripe (now thread_y controls row)
        for (int reg_x = 0; reg_x < V; ++reg_x) {
          int shared_x = thread_y * V + reg_x;  // Use thread_y for row control
          if (shared_x < TILE) {
            a_reg[reg_x] = a_shared[shared_x][stripe_i];
          }
        }

        // Load column of B into registers for the current stripe (now thread_x controls column)
        for (int reg_y = 0; reg_y < V; ++reg_y) {
          int shared_y = thread_x * V + reg_y;  // Use thread_x for column control
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
        int x = block_y * TILE + thread_y * V + i; // Now thread_y controls row index
        int y = block_x * TILE + thread_x * V + j; // Now thread_x controls column index
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

////////////////////////////////////////////////////////////////////////////////
// Max and sum reductions
////////////////////////////////////////////////////////////////////////////////

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

}  // namespace cuda
}  // namespace needle

PYBIND11_MODULE(ndarray_backend_cuda, m) {
  namespace py = pybind11;
  using namespace needle;
  using namespace cuda;

  m.attr("__device_name__") = "cuda";
  m.attr("__tile_size__") = TILE;

  py::class_<CudaArray>(m, "Array")
      .def(py::init<size_t>(), py::return_value_policy::take_ownership)
      .def_readonly("size", &CudaArray::size)
      .def("ptr", &CudaArray::ptr_as_int);

  // return numpy array, copying from CPU
  m.def("to_numpy", [](const CudaArray& a, std::vector<size_t> shape, std::vector<size_t> strides,
                       size_t offset) {
    std::vector<size_t> numpy_strides = strides;
    std::transform(numpy_strides.begin(), numpy_strides.end(), numpy_strides.begin(),
                   [](size_t& c) { return c * ELEM_SIZE; });

    // copy memory to host
    scalar_t* host_ptr = (scalar_t*)std::malloc(a.size * ELEM_SIZE);
    if (host_ptr == 0) throw std::bad_alloc();
    cudaError_t err = cudaMemcpy(host_ptr, a.ptr, a.size * ELEM_SIZE, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));

    // return numpy array
    py::capsule deallocate_buffer(host_ptr, [](void* p) { free(p); });
    return py::array_t<scalar_t>(shape, numpy_strides, host_ptr + offset, deallocate_buffer);
  });

  // copy numpy array to GPU
  m.def("from_numpy", [](py::array_t<scalar_t> a, CudaArray* out) {
    cudaError_t err =
        cudaMemcpy(out->ptr, a.request().ptr, out->size * ELEM_SIZE, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
  });

  m.def("fill", Fill);
  m.def("compact", Compact);
  m.def("ewise_setitem", EwiseSetitem);
  m.def("scalar_setitem", ScalarSetitem);
  m.def("ewise_add", EwiseAdd);
  m.def("scalar_add", ScalarAdd);

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

  m.def("matmul", Matmul);

  m.def("reduce_max", ReduceMax);
  m.def("reduce_sum", ReduceSum);
}