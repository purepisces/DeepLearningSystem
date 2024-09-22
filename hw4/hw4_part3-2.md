This is the notebook "Implementing convolutions" mentioned here:

**Convolution forward**
Implement the forward pass of 2D multi-channel convolution in `ops.py`. You should probably refer to [this notebook](https://github.com/dlsyscourse/public_notebooks/blob/main/convolution_implementation.ipynb) from lecture, which implements 2D multi-channel convolution using im2col in numpy.

# Implementing convolutions

This notebook will walk you through the process of implementing a reasonably efficient convolutional. We'll do this using native numpy for the purposes of illustration, but in the homework, you'll need to implement these all using your own straight-C (and GPU) implementations.

## Convolutions

Here we will build up some of the basic approaches for convolution, from a simple all-for-loop algorithm to an algorithm that uses a single matrix multiplication plus resize operations.

### Storage order

In the simple fully-connected networks we have been developing so far, hidden units are typically simply represented as vectors, i..e., a quantity $z \in \mathbb{R}^n$, or when representing an entire minibatch, a matrix $Z \in \mathbb{R}^{B \times n}$. But when we move to convolutional networks, we need to include additional structure in the hidden unit. This is typically done by representing each hidden vector as a 3D array, with dimensions `height x width x channels`, or in the minibatch case, with an additional batch dimension. That is, we could represent a hidden unit as an array

```c++
float Z[BATCHES][HEIGHT][WIDTH][CHANNELS];
```
The format above is referred to as NHWC format (number(batch)-height-width-channel). However, there are other ways we can represent the hidden unit as well. For example, PyTorch defaults to the NCHW format (indexing over channels in the second dimension, then height and width), though it can also support NHWC in later versions. There are subtle but substantial differences in the performance for each different setting: convolutions are typically faster in NHWC format, owing to their ability to better exploit tensor cores; but NCHW format is typically faster for BatchNorm operation (because batch norm for convolutional networks operates over all pixels in an individual channel).

Although less commonly discussed, there is a simliar trade-off to be had when it comes to storing the convolutional weights (filter) as well. Convolutional filters are specified by their kernel size (which can technically be different over different height and width dimensions, but this is quite uncommon), their input channels, and their output channels. We'll store these weights in the form:

```c++
float weights[KERNEL_SIZE][KERNEL_SIZE][IN_CHANNELS][OUT_CHANNELS];
```

Again, PyTorch does things a bit differently here (for no good reason, as far as I can tell, it was just done that way historically), storing weight in the order `OUT_CHANNELS x IN_CHANNELS x KERNELS_SIZE x KERNEL_SIZE`.



## Convolutions with simple loops

Let's begin by implementing a simple convolutional operator. We're going to implement a simple version, which allows for different kernel sizes but which _doesn't_ have any built-in padding: to implement padding, you'd just explicitly form a new ndarray with the padding built in. This means that if we have an $H \times W$ input image and convolution with kernel size $K$, we'll end up with a $(H - K + 1) \times (W - K + 1)$ image.

Although it's "cheating" in some sense, we're going to use PyTorch as a reference implementation of convolution that we will check against. However, since PyTorch, as mentioned above, uses the NCHW format (and stores the convolutional weights in a different ordering as well), and we'll use the NHWC format and the weights ordering stated above, we will need to swap things around for our reference implementation.

```python
import torch
import torch.nn as nn

def conv_reference(Z, weight):
    # NHWC -> NCHW
    Z_torch = torch.tensor(Z).permute(0,3,1,2)
    
    # KKIO -> OIKK
    W_torch = torch.tensor(weight).permute(3,2,0,1)
    
    # run convolution
    out = nn.functional.conv2d(Z_torch, W_torch)
    
    # NCHW -> NHWC
    return out.permute(0,2,3,1).contiguous().numpy()
```
```python
Z = np.random.randn(10,32,32,8)
W = np.random.randn(3,3,8,16)
out = conv_reference(Z,W)
print(out.shape)
# print result:(10, 30, 30, 16)
```
> - The number of output channels equals the number of filters, which is 16 in this case.
```python
%%time
out = conv_reference(Z,W)
```
```css
CPU times: user 3.72 ms, sys: 1.28 ms, total: 5 ms 
Wall time: 1.6 ms
```

### Explain of Performance Implications of NHWC vs NCHW

#### 1. **NHWC (Batch-Height-Width-Channel) Format:**

-   **Tensor Cores**: NHWC format can be more efficient for **convolutions** on certain hardware like NVIDIA GPUs, which use **tensor cores**. Tensor cores are specialized hardware units optimized for matrix operations, which are common in convolution operations. NHWC better exploits the memory alignment and processing capabilities of tensor cores, resulting in faster convolution operations.
-   **Use case**: NHWC is often favored when convolutional operations dominate the network, as it improves memory access patterns and speeds up convolutions.

#### 2. **NCHW (Batch-Channel-Height-Width) Format:**

-   **Batch Normalization**: In CNNs, **Batch Normalization** (BatchNorm) normalizes the activations over all the pixels in an individual channel. Since the channel dimension comes first in the NCHW format, accessing and computing over all the pixels in a channel is more efficient in this format.
-   **Use case**: NCHW is typically faster when using BatchNorm operations, which are common in CNNs. It allows for efficient computation over the channel dimension, especially in deep networks where batch normalization is applied frequently.

### Explain of CNN progress

If the task is to detect a **car** in an image:

-   The **early layers** might detect edges corresponding to the outlines of the car's wheels, windows, or body.
-   The **middle layers** might combine these edge features to detect parts of the car, such as the **shape of a wheel** or the **front of the car**.
-   The **deeper layers** would then combine these mid-level features to identify the **entire car** as a high-level object.
-   The **final output feature map** wouldn't show individual edges; instead, it would contain information about the car's **overall shape** or whether a car is present in that part of the image.

### Explanation of Output Shape 

**Convolution Without Padding**

-   **No Padding**: Since padding is not applied, the output dimensions decrease.
    
-   **Formula for Output Dimensions**:
$$\text{Output Dimension} = \left\lfloor \frac{\text{Input Dimension} - \text{Kernel Size}}{\text{Stride}} \right\rfloor + 1$$
    -   **Applying the Formula**:
    
        -   **Height**: $${\text{out}} = \left\lfloor \frac{32 - 3}{1} \right\rfloor + 1 = 30$$
        -   **Width**: $${\text{out}} = \left\lfloor \frac{32 - 3}{1} \right\rfloor + 1 = 30$$

**Output Shape Calculation**:

-   Without padding and with a stride of 1, the output dimensions decrease by `Kernel Size - 1` in each spatial dimension.
-   The number of output channels equals the number of filters. **Output Channels (`C_out`)**: 16

___
Now let's consider the simplest possible implementation of a convolution, that just does the entire operation using for loops.
```python
def conv_naive(Z, weight):
    N,H,W,C_in = Z.shape
    K,_,_,C_out = weight.shape
    
    out = np.zeros((N,H-K+1,W-K+1,C_out));
    for n in range(N):
        for c_in in range(C_in):
            for c_out in range(C_out):
                for y in range(H-K+1):
                    for x in range(W-K+1):
                        for i in range(K):
                            for j in range(K):
                                out[n,y,x,c_out] += Z[n,y+i,x+j,c_in] * weight[i,j,c_in,c_out]
    return out
```
We can check to make sure this implementation works by comparing to the PyTorch reference implementation.

```python
out2 = conv_naive(Z,W)
print(np.linalg.norm(out - out2))
#print result is 6.616561372320758e-13
```
The implementation works, but (not surprisingly, since you would never want to actually have a 7-fold loop in interpreted code), the PyTorch version is _much much_ faster; YMMV, but on my laptop, the naive implementation is more than 2000 times slower.

```python
%%time
out2 = conv_naive(Z,W)
```
```css
CPU times: user 7.64 s, sys: 68.9 ms, total: 7.71 s 
Wall time: 7.8 s
```

```python
%%time
out = conv_reference(Z,W)
```
```css
CPU times: user 4.14 ms, sys: 2.21 ms, total: 6.36 ms 
Wall time: 2.13 ms
```

## Convolutions as matrix mulitplications

  
Ok, but, no one is going to actually implement convolutions elementwise in Python. Let's see how we can start to do much better. The simplest way to make this much faster (and frankly, a very reasonable implementation of convolution) is to perform it as a sequence of matrix multiplications. Remember that a kernel size $K = 1$ convolution is equivalent to performing matrix multiplication over the channel dimensions. That is, suppose we have the following convolution.

```python
#Z = np.random.randn(10,32,32,8)
W1 = np.random.randn(1,1,8,16)
out = conv_reference(Z,W1)
print(out.shape)
# print result is (10, 32, 32, 16)
```
Then we could implement the convolution using a _single_ matrix multiplication.

```python
out2 = Z @ W1[0,0]
print(np.linalg.norm(out - out2))
#print result is 3.628750477299305e-14
```

```python
W1[0,0].shape
#print result is (8, 16)
```

We're here exploiting the nicety that in numpy, when you compute a matrix multiplication by a multi-dimensional array, it will treat the leading dimensions all as rows of a matrix. That is, the above operation would be equivalent to:

```python
out2 = (Z.reshape(-1,8) @ W1[0,0]).reshape(Z.shape[0], Z.shape[1], Z.shape[2], W1.shape[3])
```
This strategy immediately motivates a very natural approach to convolution: we can iterate over just the kernel dimensions $i$ and $j$, and use matrix multiplication to perform the convolution.

```python
def  conv_matrix_mult(Z, weight):
	N,H,W,C_in = Z.shape
	K,_,_,C_out = weight.shape
	out = np.zeros((N,H-K+1,W-K+1,C_out))
	for i in  range(K):
		for j in  range(K):
			out += Z[:,i:i+H-K+1,j:j+W-K+1,:] @ weight[i,j]
	return out
```
```python
Z = np.random.randn(100,32,32,8)
W = np.random.randn(3,3,8,16)

out = conv_reference(Z,W)
out2 = conv_matrix_mult(Z,W)
print(np.linalg.norm(out - out2))
# print result is 3.265213882042324e-12
```

This works as well, as (as expected) is _much_ faster, starting to be competetive even with the PyTorch version (about 2-3x slower on my machine). Let's in fact increase the batch size a bit to make this a more lengthy operation.

```python
%%time
out = conv_reference(Z,W)
```
```css
CPU times: user 35.7 ms, sys: 9.13 ms, total: 44.8 ms 
Wall time: 19.5 ms
```
```python
%%time
out = conv_matrix_mult(Z,W)
```
```css
CPU times: user 32.8 ms, sys: 7.35 ms, total: 40.1 ms 
Wall time: 39.8 ms
```

___

### Explain Kernel

```python
import numpy as np

weight = np.array([
    # Filter for output channel 1 (C_out = 1)
    [
        # Top-left and top-right positions of the 2x2 filter (applied across 3 input channels)
        [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],  # Values for the RGB channels at (0, 0) and (0, 1)
        # Bottom-left and bottom-right positions of the 2x2 filter (applied across 3 input channels)
        [[0.7, 0.8, 0.9], [1.0, 1.1, 1.2]]   # Values for the RGB channels at (1, 0) and (1, 1)
    ],

    # Filter for output channel 2 (C_out = 2)
    [
        # Top-left and top-right positions of the 2x2 filter (applied across 3 input channels)
        [[1.3, 1.4, 1.5], [1.6, 1.7, 1.8]],  # Values for the RGB channels at (0, 0) and (0, 1)
        # Bottom-left and bottom-right positions of the 2x2 filter (applied across 3 input channels)
        [[1.9, 2.0, 2.1], [2.2, 2.3, 2.4]]   # Values for the RGB channels at (1, 0) and (1, 1)
    ]
])

print("weight[0, 0]")
print(weight[0, 0])
print("weight[0, 1]")
print(weight[0, 1])
print("weight[1, 0]")
print(weight[1, 0])
print("weight[1, 1]")
print(weight[1, 1])
```
```css
weight[0, 0]
[[0.1 0.2 0.3]
 [0.4 0.5 0.6]]
weight[0, 1]
[[0.7 0.8 0.9]
 [1.  1.1 1.2]]
weight[1, 0]
[[1.3 1.4 1.5]
 [1.6 1.7 1.8]]
weight[1, 1]
[[1.9 2.  2.1]
 [2.2 2.3 2.4]]
```

-   **`weight[0, 0]`**: This corresponds to the first filter (for output channel 1) and includes the values for the top-left and top-right positions `(0, 0)` and `(0, 1)` of the `2x2` kernel. The values are:
    
    -   `[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]` for the RGB channels at those positions.
-   **`weight[0, 1]`**: This corresponds to the first filter (for output channel 1) and includes the values for the bottom-left and bottom-right positions `(1, 0)` and `(1, 1)` of the `2x2` kernel. The values are:
    
    -   `[[0.7, 0.8, 0.9], [1.0, 1.1, 1.2]]` for the RGB channels at those positions.
-   **`weight[1, 0]`**: This corresponds to the second filter (for output channel 2) and includes the values for the top-left and top-right positions `(0, 0)` and `(0, 1)` of the `2x2` kernel. The values are:
    
    -   `[[1.3, 1.4, 1.5], [1.6, 1.7, 1.8]]` for the RGB channels at those positions.
-   **`weight[1, 1]`**: This corresponds to the second filter (for output channel 2) and includes the values for the bottom-left and bottom-right positions `(1, 0)` and `(1, 1)` of the `2x2` kernel. The values are:
    
    -   `[[1.9, 2.0, 2.1], [2.2, 2.3, 2.4]]` for the RGB channels at those positions.

___
## Manipulating matrices via strides

Before implementing convolutions via im2col, let's consider an example that actually has nothing to do with convolution. Instead, let's consider the efficient matrix multiplication operations that we discussed in an earlier lecture. Normally we think of storing a matrix as a 2D array:

```c++
float A[M][N];
```

In the typical row-major format, this will store each N dimensional row of the matrix one after the other in memory. However, recall that in order to make better use of the caches and vector operations in modern CPUs, it was beneficial to lay our our matrix memory groups by individual small "tiles", so that the CPU vector operations could efficiently access operators

	float A[M/TILE][N/TILE][TILE][TILE];

where `TILE` is some small constant (like 4), which allows the CPU to use its vector processor to perform very efficient operations on `TILE x TILE` blocks. Importantly, what enables this to be so efficient is that in the standard memory ordering for an ND array, this grouping would locate all `TILE x TILE` block consecutively in memroy, so they could quickly be loaded in and out of cache / registers / etc.

How exactly would we convert a matrix to this form? You could imagine how to manually copy from one matrix type to another, but it would be rather cumbersome to write this code each time you wanted to experiment with different (and in order for the code to be efficient, you'd need to write it in C/C++ as well, which could get to be a pain). Instead, we're going to show you how to do this using the handy function `np.lib.stride_tricks.as_strided()`, which lets you create new matrices by manually manipulating the strides of a matrix but _not_ changing the data; we can then use `np.ascontiguousarray()` to lay out the memory sequentially. This sets of tricks let us rearrange matrices fairly efficiently in just one or two lines of numpy code.

### An example: a 6x6 2D array

To see how this works, let's consider an example 6x6 numpy array.

```python
import numpy as np
n = 6
A = np.arange(n**2, dtype=np.float32).reshape(n,n)
print(A)
```
```css
[[ 0. 1. 2. 3. 4. 5.] 
[ 6. 7. 8. 9. 10. 11.] 
[12. 13. 14. 15. 16. 17.] 
[18. 19. 20. 21. 22. 23.] 
[24. 25. 26. 27. 28. 29.] 
[30. 31. 32. 33. 34. 35.]]
```
This array is layed out in memory by row. It's actually a bit of a pain to access the underlying raw memory of a numpy array in Python (numpy goes to great things to try to _prevent_ you from doing this, but we can see how the array is layed out using the following code):

```python
import ctypes
print(np.frombuffer(ctypes.string_at(A.ctypes.data, A.nbytes), dtype=A.dtype, count=A.size))
```
```css
[ 0. 1. 2. 3. 4. 5. 6. 7. 8. 9. 10. 11. 12. 13. 14. 15. 16. 17. 
18. 19. 20. 21. 22. 23. 24. 25. 26. 27. 28. 29. 30. 31. 32. 33. 34. 35.]
```

A few lectures ago, we discussed the use of the `strides` structure as a way to lay out n-dimensional arrays in memory. In order to access the `A[i][j]` element of a 2D array, for instance, we would access the memory location at:

  

```c++
A.bytes[i * strides[0] + j * strides[1]];
```

The same can be done e.g., with a 3D tensor, accessing `A[i][j][k]` at memory location:

  

```c++
A.bytes[i * strides[0] + j * strides[1] + k * strides[2]];
```

For an array in row-major format, we would thus have

```c++
strides[0] = num_cols;
strides[1] = 1;
```
We can look at the strides of the array we have created using the `.strides` property.

```python
print(A.strides)
```

```css
(24, 4)
```

Note that numpy, somewhat unconventionally, actually uses strides equal to the total number of _bytes_, so these numbers are all multiplied by 4 from the above, because a `float32` type takes up 4 bytes.

### Tiling a matrix using strides

Now let's consider how to create a tiled form of the `A` array by _just_ changing the strides. For simplicity, let's assume we want to tile into 2x2 blocks, and thus we want to convert `A` into a `3 x 3 x 2 x 2` array. What would the strides be in this case? In other words, if we accessed the element `A[i][j][k][l]`, how would this index into a memory location in the array as layed out above? Incrementing the first index, `i`, would move down two rows in the matrix, so `strides[0] = 12`; similarly, incrementing the second index `j` would move over two columns, so `strides[1]=2`. Things get a bit tricker next, but are still fairly straightforward: incrementing the next index `k` moves down one row in the matrix, so `strides[2]=6`, and finally incrementing the last index `l` just moves us over one column, so `strides[3]=1`.


Let's create a matrix with this form using the `np.lib.stride_tricks.as_strided()`. This function lets you specify the shape and stride of a new matrix, created from the same memory as an old matrix. That is, it doesn't do any memory copies, so it's very efficient. But you also have to be careful when you use it, because it's directly creating a new view of an existing array, and without proper care you could e.g., go outside the bounds of the array.

  

Here's how we can use it to create the tiled view of the matrix `A`.

```python
B = np.lib.stride_tricks.as_strided(A, shape=(3,3,2,2), strides=np.array((12,2,6,1))*4)

print(B)
```
```css
[[[[ 0.  1.]
   [ 6.  7.]]

  [[ 2.  3.]
   [ 8.  9.]]

  [[ 4.  5.]
   [10. 11.]]]


 [[[12. 13.]
   [18. 19.]]

  [[14. 15.]
   [20. 21.]]

  [[16. 17.]
   [22. 23.]]]


 [[[24. 25.]
   [30. 31.]]

  [[26. 27.]
   [32. 33.]]

  [[28. 29.]
   [34. 35.]]]]
```
Parsing numpy output for ND array isn't the most intuitive thing, but if you look you can see that these basically lay out each 2x2 block of the matrix, as desired. However, can also see the fact that this call didn't change the actual memory layout by again inspecting the raw memory.

```python
print(np.frombuffer(ctypes.string_at(B.ctypes.data, size=B.nbytes), B.dtype, B.size))
```
```css
[ 0. 1. 2. 3. 4. 5. 6. 7. 8. 9. 10. 11. 12. 13. 14. 15. 16. 17. 
18. 19. 20. 21. 22. 23. 24. 25. 26. 27. 28. 29. 30. 31. 32. 33. 34. 35.]
```
```python
B.strides
```
```css
(48, 8, 24, 4)
```

In order to change reorder the memory so that the underlying matrix is continguous/compact (which is what we need for making the matrix multiplication efficient), we can use the `np.ascontinugousarray()` function.

```python
C = np.ascontiguousarray(B)

print(np.frombuffer(ctypes.string_at(C.ctypes.data, size=C.nbytes), C.dtype, C.size))
```
```css
[ 0. 1. 6. 7. 2. 3. 8. 9. 4. 5. 10. 11. 12. 13. 18. 19. 14. 15. 
20. 21. 16. 17. 22. 23. 24. 25. 30. 31. 26. 27. 32. 33. 28. 29. 34. 35.]
```

As you can see, the `C` array is layed out in compact order. This can also be verified by looking as it's `.strides` property.

```python
print(C.strides)
```
```css
(48, 16, 8, 4)
```
## Convolutions via im2col

Let's consider finally the "real" way to implement convolutions, which will end up being about as fast as PyTorch's implementation. Essentially, we want to bundle all the computation needed for convolution into a _single_ matrix multiplication, which will then leverage all the optimizations that we can implement for normal matrix multiplication.

They key approach to doing this is called the `im2col` operator, which "unfolds" a 4D array into exactly the form needed to perform multiplication via convolution. Let's see an example of how this works using a simple 2D array, before we move to the 4D case. Let's consider the following array we used above in the first section.

```python
A = np.arange(36, dtype=np.float32).reshape(6,6)
print(A)
```
```css
[[ 0. 1. 2. 3. 4. 5.] 
[ 6. 7. 8. 9. 10. 11.] 
[12. 13. 14. 15. 16. 17.] 
[18. 19. 20. 21. 22. 23.] 
[24. 25. 26. 27. 28. 29.] 
[30. 31. 32. 33. 34. 35.]]
```
And let's consider convolting with a 3x3 filter.

```python
W = np.arange(9, dtype=np.float32).reshape(3,3)
print(W)
```
```css
[[0. 1. 2.] 
[3. 4. 5.] 
[6. 7. 8.]]
```

Recall that a convolution will multiply this filter with every 3x3 block in the image. So how can we extract every such 3x3 block. The key will be to form a $(H - K + 1) \times (W - K + 1) \times K \times K$ array, that contains all of these blocks, then flatten it to a matrix we can multiply by the filter (this is the same process we did mathematically in the previous lecture on convolutions for 1D convolutions, but we're now doing to do it for real for the 2D case). But how can we go about creating this array of all blocks, short of manual copying. Fortunately, it turns out that the `as_strided()` call we talked about above is actually exactly what we needed for this.

Specifically, if we created a new view in the matrix, of size `(4,4,3,3)`, how can we use `as_strided()` to return the matrix we want? Well, note that the first two dimenions will have strides of 6 and 1, just like in the regular array: incrementing the first index by 1 will move to the next row, and incrementing the next will move to the next column. But interestingly (and this is the "trick"), the third and fourth dimensions _also_ have strides of 6 and 1 respectively, because incrementing the third index by one _also_ moves to the next row, and similarly for the fourth index. Let's see what this looks like in practice.

```python
B = np.lib.stride_tricks.as_strided(A, shape=(4,4,3,3), strides=4*(np.array((6,1,6,1))))

print(B)
```
```css
[[[[ 0.  1.  2.]
   [ 6.  7.  8.]
   [12. 13. 14.]]

  [[ 1.  2.  3.]
   [ 7.  8.  9.]
   [13. 14. 15.]]

  [[ 2.  3.  4.]
   [ 8.  9. 10.]
   [14. 15. 16.]]

  [[ 3.  4.  5.]
   [ 9. 10. 11.]
   [15. 16. 17.]]]


 [[[ 6.  7.  8.]
   [12. 13. 14.]
   [18. 19. 20.]]

  [[ 7.  8.  9.]
   [13. 14. 15.]
   [19. 20. 21.]]

  [[ 8.  9. 10.]
   [14. 15. 16.]
   [20. 21. 22.]]

  [[ 9. 10. 11.]
   [15. 16. 17.]
   [21. 22. 23.]]]


 [[[12. 13. 14.]
   [18. 19. 20.]
   [24. 25. 26.]]

  [[13. 14. 15.]
   [19. 20. 21.]
   [25. 26. 27.]]

  [[14. 15. 16.]
   [20. 21. 22.]
   [26. 27. 28.]]

  [[15. 16. 17.]
   [21. 22. 23.]
   [27. 28. 29.]]]


 [[[18. 19. 20.]
   [24. 25. 26.]
   [30. 31. 32.]]

  [[19. 20. 21.]
   [25. 26. 27.]
   [31. 32. 33.]]

  [[20. 21. 22.]
   [26. 27. 28.]
   [32. 33. 34.]]

  [[21. 22. 23.]
   [27. 28. 29.]
   [33. 34. 35.]]]]
```

This is exactly the 4D array we want. Now, if we want to compute the convolution as a "single" matrix multiply, we just flatten reshape this array to a $(4 \cdot 4) \times (3 \cdot 3)$ matrix, reshape the weights to a $9$ dimensional vector (the weights will become a matrix again for the case of multi-channel convolutions), and perform the matrix multiplication. We then reshape the resulting vector back into a $4 \times 4$ array to perform the convolution.

```python
print(np.frombuffer(ctypes.string_at(B.ctypes.data, size=A.nbytes), B.dtype, A.size))
```
```css
[ 0.  1.  2.  3.  4.  5.  6.  7.  8.  9. 10. 11. 12. 13. 14. 15. 16. 17.
 18. 19. 20. 21. 22. 23. 24. 25. 26. 27. 28. 29. 30. 31. 32. 33. 34. 35.]
```
```python
B.strides
```
```css
(24, 4, 24, 4)
```

```python
C = B.reshape(16,9)
```
```python
C.strides
```
```css
(36, 4)
```
```python
(B.reshape(16,9) @ W.reshape(9)).reshape(4,4)
```
```css
array([[ 366.,  402.,  438.,  474.],
       [ 582.,  618.,  654.,  690.],
       [ 798.,  834.,  870.,  906.],
       [1014., 1050., 1086., 1122.]], dtype=float32)
```
### A critical note on memory efficiency

There is a _very_ crucial point to make regarding memory efficiency of this operation. While reshaping `W` into an array (or what will be a matrix for multi-channel convolutions) is "free", in that it doesn't allocate any new memory, reshaping the `B` matrix above is very much _not_ a free operation. Specifically, while the strided form of `B` uses the same memory as `A`, once we actually convert `B` into a 2D matrix, there is no way to represent this data using any kind of strides, and we have to just allocate the entire matrix. This means we actually need to _form_ the full im2col matrix, which requires $O(K^2)$ more memory than the original image, which can be quite costly for large kernel sizes.

  

For this reason, in practice it's often the case that the best modern implementations _won't_ actually instatiate the full im2col matrix, and will instead perform a kind of "lazy" formation, or specialize the matrix operation natively to im2col matrices in their native strided form. These are all fairly advanced topics that we won't deal with any further in the course, because for our purposes, it will be sufficient to just allocate this matrix and then quickly deallocate it after we perform the convolution (remember that we aren't e.g., doing backprop through the im2col operation).

### im2col for multi-channel convolutions

So how do we actually implement an im2col operation for real multi-channel, minibatched convolutions? It turns out the process is not much more complicated. Instead of forming a 4D $(H - K + 1) \times (W - K + 1) \times K \times K$ array, we form a 6D $N \times (H - K + 1) \times (W - K + 1) \times K \times K \times C$ array (leaving the minibatch and channel dimensions untouched). And, after thinking about it for a bit, it should be pretty clear that we can apply the same trick by just repeating the strides for dimensions 1 and 2 (the height and width) for dimensions 3 and 4 (the $K \times K$ blocks), and leave the stirdes for the minibatch and channels unchanged. Furthermore, you don't even need to worry about manually computing the strides manually: you can just use the strides of the $Z$ input and repeat whatever they are.

To compute the convolution, tou then flatten the im2col matrix to a $(N \cdot (H - K + 1) \cdot (W - K + 1)) \times (K \cdot K \cdot C)$ matrix (remember, this operation is highly memory inefficient), flatten the weights array to a $(K \cdot K \cdot C) \times C_{out}$ matrix, perform the multiplication, and resize back to the desired size of the final 4D array output. Here's the complete operation.

```python
def  conv_im2col(Z, weight):

	N,H,W,C_in = Z.shape
	K,_,_,C_out = weight.shape
	Ns, Hs, Ws, Cs = Z.strides
	inner_dim = K * K * C_in
	A = np.lib.stride_tricks.as_strided(Z, shape = (N, H-K+1, W-K+1, K, K, C_in),
	strides = (Ns, Hs, Ws, Hs, Ws, Cs)).reshape(-1,inner_dim)
	out = A @ weight.reshape(-1, C_out)
	return out.reshape(N,H-K+1,W-K+1,C_out)
```
Again, we can check that this version produces the same output as the PyTorch reference (or our other implementations, at this point):

```python
Z = np.random.randn(100,32,32,8)
W = np.random.randn(3,3,8,16)
out = conv_reference(Z,W)
out2 = conv_im2col(Z,W)
print(np.linalg.norm(out - out2))
# print result 3.644603550232283e-12
```
However, at this point we're finally starting to get competetive with PyTorch, taking only 25% more time than the PyTorch implementation on my machine.

```python
%%time
out3 = conv_im2col(Z,W)
```
```css
CPU times: user 16.8 ms, sys: 6.56 ms, total: 23.4 ms 
Wall time: 22.3 ms
```
```python
%%time
out3 = conv_reference(Z,W)
```
```css
CPU times: user 38.6 ms, sys: 8.01 ms, total: 46.6 ms 
Wall time: 16.9 ms
```
## Final notes

Hopefully this quick intro gave you a bit of appreciation and understanding of what is going on "under the hood" of modern convolution implementations. It hopefully also gave you some understanding how just how powerful stride manipulation can be, able to accomplish some very complex operations without actually needing to explicitly loop over matrices (though, as we'll see on the homework, a bit of the complexity is still being outsourced to the `.reshape` and it's implicit `np.ascontinugousarray()` call, which is not completely trivial; but we'll deal with this on the homework.

___
### Explain of kernel/filter 🌟🌟🌟


for **multi-channel convolutions**, each **kernel** (also called a filter) has a size of **`K x K x C_in`**, where:

-   **`K x K`**: Represents the spatial size of the filter (e.g., 3x3, 5x5). This refers to how the filter moves over the height and width of the input.
-   **`C_in`**: Refers to the **number of input channels** (or depth). For example, if the input is an RGB image, `C_in` would be 3 because there are 3 color channels (Red, Green, Blue).

### Example of a Kernel with Multi-Channel Input (RGB Image)

Let’s say we have an RGB image (3 channels: Red, Green, Blue) with a **height** and **width** of 5x5, and we are using a **3x3** kernel.

For an RGB image:

-   The input has 3 channels, so the kernel must also operate across all 3 channels.
-   The kernel would be a **3D tensor** with size **`3x3x3`**, where the third dimension corresponds to the 3 input channels.

#### Visualization of the Kernel (3x3x3):

-   Each kernel will have 3 slices, one for each channel.

```python

Kernel (3x3x3):

Slice 1 (for Red channel):
[[k11, k12, k13],
 [k14, k15, k16],
 [k17, k18, k19]]

Slice 2 (for Green channel):
[[k21, k22, k23],
 [k24, k25, k26],
 [k27, k28, k29]]

Slice 3 (for Blue channel):
[[k31, k32, k33],
 [k34, k35, k36],
 [k37, k38, k39]]` 
```
Here:

-   **Each slice** operates on one channel of the input.
-   All three slices together form the full kernel that operates on a **3-channel input**.

### How the Kernel is Applied to the Input:

When this **`3x3x3` kernel** is applied to an RGB image, it performs **element-wise multiplication** between the kernel and the corresponding **3x3 patch of the input** across all 3 channels, and then sums the results. This gives a single scalar value, which becomes one element in the output feature map.

#### Example:

Let’s say the input is a small patch from an RGB image:

-   **Input Patch (3x3x3)**:
    
```python
    
    `Red channel:    Green channel:    Blue channel:
    [[1, 2, 3],     [[4, 5, 6],       [[7, 8, 9],
     [0, 1, 2],      [3, 2, 1],        [6, 5, 4],
     [1, 0, 1]]      [4, 3, 2]]        [7, 6, 5]]` 
```

-   **Kernel (3x3x3)**:
  
```python
    Red channel:    Green channel:    Blue channel:
    [[ 1, 0, -1],   [[ 0, 1, 0],      [[-1, 0, 1],
     [ 1, 0, -1],    [ 1, 0, -1],       [ 0, 1, 0],
     [ 1, 0, -1]],   [ 1, 0, -1]],      [-1, 0, 1]]
```    

The kernel is applied to the input patch by performing element-wise multiplication and summing the results across all channels:


```python
Red channel:
(1*1) + (2*0) + (3*(-1)) + (0*1) + (1*0) + (2*(-1)) + (1*1) + (0*0) + (1*(-1)) = -2

Green channel:
(4*0) + (5*1) + (6*0) + (3*1) + (2*0) + (1*(-1)) + (4*1) + (3*0) + (2*(-1)) = 8

Blue channel:
(7*(-1)) + (8*0) + (9*1) + (6*0) + (5*1) + (4*0) + (7*(-1)) + (6*0) + (5*1) = 7

Total sum = -2 (Red) + 8 (Green) + 7 (Blue) = 13
```
The result of this operation for this 3x3 patch is **13**, which becomes one value in the output feature map.

### **Multiple Filters**:

In most convolutional layers, you don’t just apply one filter; you apply multiple filters to extract different features from the input. Each filter might detect different patterns, such as edges, textures, or specific shapes.

For example:

-   You could have 16 filters of size **`3x3x3`**, where each filter detects different features.
-   The result of applying each filter would be **16 output channels**, resulting in a **feature map** of size **`HxWx16`**.

### Recap:

-   **Kernel Size** for a multi-channel convolution is **`KxKxC_in`**, where:
    -   `KxK` is the spatial size (height and width) of the filter.
    -   `C_in` is the number of input channels (e.g., 3 for RGB images, or more in deeper layers of a neural network).
-   The filter is applied across all channels of the input, and the resulting dot products are summed to give a single output value at each position in the output feature map.
-   If there are multiple filters, each filter produces its own output channel, resulting in a multi-channel output feature map.
