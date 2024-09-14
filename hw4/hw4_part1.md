# 10-714 Homework 4

In this homework, you will leverage all of the components built in the last three homeworks to solve some modern problems with high performing network structures. We will start by adding a few new ops leveraging our new CPU/CUDA backends. Then, you will implement convolution, and a convolutional neural network to train a classifier on the CIFAR-10 image classification dataset. Then, you will implement recurrent and long-short term memory (LSTM) neural networks, and do word-level prediction language modeling on the Penn Treebank dataset.

As always, we will start by copying this notebook and getting the starting code.

Reminder: __you must save a copy in drive__.


## Part 1: ND Backend [10 pts]

Recall that in homework 2, the `array_api` was imported as `numpy`. In this part, the goal is to write the necessary operations with `array_api` imported from the needle backend `NDArray` in `python/needle/backend_ndarray/ndarray.py`. Make sure to copy the solutions for `reshape`, `permute`, `broadcast_to` and `__getitem__` from homework 3.

Fill in the following classes in `python/needle/ops_logarithmic.py` and `python/needle/ops_mathematic.py`:

- `PowerScalar`

- `EWiseDiv`

- `DivScalar`

- `Transpose`

- `Reshape`

- `BroadcastTo`

- `Summation`

- `MatMul`

- `Negate`

- `Log`

- `Exp`

- `ReLU`

- `LogSumExp`

- `Tanh` (new)

- `Stack` (new)

- `Split` (new)

  
Note that for most of these, you already wrote the solutions in the previous homework and you should not change most part of your previous solution, if issues arise, please check if the `array_api` function used is supported in the needle backend.

`TanhOp`, `Stack`, and `Split` are newly added. `Stack` concatenates same-sized tensors along a new axis, and `Split` undoes this operation. The gradients of the two operations can be written in terms of each other. We do not directly test `Split`, and only test the backward pass of `Stack` (for which we assume you used `Split`).


**Note:** You may want to make your Summation op support sums over multiple axes; you will likely need it for the backward pass of the BroadcastTo op if yours supports broadcasting over multiple axes at a time. However, this is more about ease of use than necessity, and we leave this decision up to you (there are no corresponding tests).

**Note:** Depending on your implementations, you may want to ensure that you call `.compact()` before reshaping arrays. (If this is necessary, you will run into corresponding error messages later in the assignment.)

```python
matmul
summation
transpose
LogSumExp
```

___
## Explain `stack`

Let’s go through a more detailed example where the tensor `A` has the shape $4 \times 3$ (4 rows and 3 columns) and we'll stack it with another tensor `B` of the same shape. We’ll explore different values of `axis` to show how the `compute` method works.

### Example:
```python
A = [[ 1,  2,  3],   # shape (4, 3)
     [ 4,  5,  6],
     [ 7,  8,  9],
     [10, 11, 12]]

B = [[13, 14, 15],   # shape (4, 3)
     [16, 17, 18],
     [19, 20, 21],
     [22, 23, 24]]
```
We are now going to stack `A` and `B` along different axes and explain how the shape evolves.

1. **Stack Along `axis=0`**:

When `axis=0`, we are adding a new first dimension, and `A` and `B` will be stacked along that dimension. This means that `A` will be placed at index `0` in the first axis, and `B` at index `1`.

#### Step-by-Step:

-   The shape of `A` and `B` is initially `(4, 3)`.
-   We insert `len(args) = 2` at position `0` (axis=0).
    -   New shape: `(2, 4, 3)`.

#### Result:
```python
stack([A, B], axis=0)

# The result is:
[[[ 1,  2,  3],
  [ 4,  5,  6],
  [ 7,  8,  9],
  [10, 11, 12]],   # This is tensor A

 [[13, 14, 15],
  [16, 17, 18],
  [19, 20, 21],
  [22, 23, 24]]]   # This is tensor B
```
-   The first axis (axis 0) has size 2 because we stacked 2 tensors.
-   Each "slice" along axis 0 is a $4 \times 3$ matrix corresponding to one of the input tensors (`A` or `B`).

2. **Stack Along `axis=1`**:

When `axis=1`, we are adding a new second dimension (between rows and columns). This means we are stacking corresponding rows from `A` and `B` together.

#### Step-by-Step:

-   The shape of `A` and `B` is initially `(4, 3)`.
-   We insert `len(args) = 2` at position `1` (axis=1).
    -   New shape: `(4, 2, 3)`.

#### Result:
```python
stack([A, B], axis=1)

# The result is:
[[[ 1,  2,  3],  [13, 14, 15]],   # Stacking the first row of A and B
 [[ 4,  5,  6],  [16, 17, 18]],   # Stacking the second row of A and B
 [[ 7,  8,  9],  [19, 20, 21]],   # Stacking the third row of A and B
 [[10, 11, 12],  [22, 23, 24]]]   # Stacking the fourth row of A and B
```
-   The first axis (axis 0) still represents the rows (4 rows).
-   The second axis (axis 1) has size 2, representing the new dimension created by stacking `A` and `B` row-wise.
-   The third axis (axis 2) still represents the columns (3 columns).

3. **Stack Along `axis=2`**:

When `axis=2`, we are adding a new third dimension, meaning that the elements of `A` and `B` will be stacked within each row and column.

#### Step-by-Step:

-   The shape of `A` and `B` is initially `(4, 3)`.
-   We insert `len(args) = 2` at position `2` (axis=2).
    -   New shape: `(4, 3, 2)`.

#### Result:
```python
stack([A, B], axis=2)

# The result is:
[[[ 1, 13],  [ 2, 14],  [ 3, 15]],   # Stacking corresponding elements from A and B in each column
 [[ 4, 16],  [ 5, 17],  [ 6, 18]],   # Stacking corresponding elements from A and B in each column
 [[ 7, 19],  [ 8, 20],  [ 9, 21]],   # Stacking corresponding elements from A and B in each column
 [[10, 22],  [11, 23],  [12, 24]]]   # Stacking corresponding elements from A and B in each column
```
-   The first axis (axis 0) still represents the rows (4 rows).
-   The second axis (axis 1) still represents the columns (3 columns).
-   The third axis (axis 2) has size 2 because you’re stacking the corresponding elements from `A` and `B` within each row and column.

### Explain `ret[:, 3, :]`

-   **`:` along axis 0**: Select all rows (so both the first and second row will be included).
-   **`3` along axis 1**: Select the 3rd slice along axis 1 (the fourth sub-array, as Python uses 0-based indexing).
-   **`:` along axis 2**: Select all columns for each selected slice.

So, `ret[:, 3, :]` selects the **3rd slice** (sub-array) from each row, including all columns in that slice.

Example Setup:
```python
ret = [[[ 1,  2,  3],  [ 4,  5,  6],  [ 7,  8,  9],  [10, 11, 12]],  # First row (axis 0, index 0)
       [[13, 14, 15],  [16, 17, 18],  [19, 20, 21],  [22, 23, 24]]]  # Second row (axis 0, index 1)
```
This `ret` tensor has:

-   2 rows (axis 0),
-   4 slices per row (axis 1),
-   3 columns per slice (axis 2).

Final Output of `ret[:, 3, :]`:
```python
[[10, 11, 12],   # 3rd slice from the first row
 [22, 23, 24]]   # 3rd slice from the second row
```
