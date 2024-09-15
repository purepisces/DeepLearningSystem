modified these function's code different from previous homework, also take a look at tanh's gradient, split's gradient, stack's gradient
```css
matmul
summation
transpose
LogSumExp
broadcast to in ndarray.py
reduce_view_out in ndarray.py:  if axis is None:
            view = self.compact().reshape((1,) * (self.ndim - 1) + (prod(self.shape),))
            out = NDArray.make((1,) * self.ndim if keepdims else (1,), device=self.device)
```
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



___
## Explain `stack`

Let’s go through a more detailed example where the tensor `A` has the shape $4 \times 3$ (4 rows and 3 columns) and we'll stack it with another tensor `B` of the same shape. We’ll explore different values of `axis` to show how the `compute` method works.

### Explain With Example:
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

### Explain why the result is NDArray not Tensor
The `result = array_api.empty()` line creates an `NDArray` (not a `Tensor`) because:

-   **NDArray** is responsible for numerical storage and computation.
-   **Tensor** is a higher-level structure that wraps around `NDArray` to add additional functionality like gradients and computational graph management.

___
## Explain Split

### Understanding Axes in Tensor with Examples

```python
A = Tensor([[1, 2, 3],
            [4, 5, 6]])
```
-   The **first axis (axis 0)** refers to the **rows** of the tensor.
-   The **second axis (axis 1)** refers to the **columns** of the tensor.

Let's break it down:

#### **Axis 0 (First Axis)**: Rows

-   The elements along axis 0 are the **rows** of the tensor. Each row is treated as a distinct element along axis 0.
    -   The first element (along axis 0) is the row `[1, 2, 3]`.
    -   The second element (along axis 0) is the row `[4, 5, 6]`.

So, the elements in axis 0 are:
```css
[1, 2, 3]  # First row
[4, 5, 6]  # Second row
```
#### **Axis 1 (Second Axis)**: Columns

-   The elements along axis 1 are the **columns** of the tensor. Each column is treated as a distinct element along axis 1.
    -   The first element (along axis 1) is the column `[1, 4]`.
    -   The second element (along axis 1) is the column `[2, 5]`.
    -   The third element (along axis 1) is the column `[3, 6]`.

So, the elements in axis 1 are:
```css
[1, 4]  # First column
[2, 5]  # Second column
[3, 6]  # Third column
```
#### Conclusion:

-   **Axis 0 (rows)**: `[1, 2, 3]`, `[4, 5, 6]`
-   **Axis 1 (columns)**: `[1, 4]`, `[2, 5]`, `[3, 6]`

Each axis refers to a different way of slicing through the tensor: axis 0 slices through rows, and axis 1 slices through columns.

### Explain split `compute` code with example

**Example:** Let `A = Tensor([[1, 2, 3], [4, 5, 6]])`.
#### Case 1: `axis = 0`

In this case, we are splitting along the rows (axis 0), so we expect the output to be two separate rows.

-   **Initial state:**
```python
A.shape = (2, 3)  # 2 rows, 3 columns
axis = 0
```
- **Step-by-step Explanation:**
```python
axis_size = A.shape[self.axis]
# axis_size = 2, since A.shape[0] = 2
```
-   We are splitting along axis 0, which has 2 elements (rows).
```python
split_tensors = []
```
-   We initialize an empty list `split_tensors` to store the split tensors.
```python
output_shape = list(A.shape)
output_shape.pop(self.axis)
# output_shape = [3], since we removed the axis 0 dimension
```
- `output_shape` becomes `[3]` because we are removing axis 0 (which had size 2), leaving us with 3 columns.
```python
for i in range(axis_size):  # Loop over the two rows
```
-   We loop through the two elements along axis 0.

**First Iteration (`i=0`):**
```python
slices = [slice(None)] * len(A.shape)
slices[self.axis] = i
# slices = [0, slice(None)], since axis = 0 and i = 0
```
- `slices` becomes `[0, slice(None)]`, meaning we take the 0th row and all columns.
```python
split_tensors.append(A[tuple(slices)].compact().reshape(output_shape))
# A[tuple(slices)] gives Tensor([1, 2, 3])
# reshape(output_shape) keeps it as [1, 2, 3]
```
-   This extracts the first row `[1, 2, 3]` from the tensor `A` and appends it to `split_tensors`.

**Second Iteration (`i=1`):**
```python
slices = [slice(None)] * len(A.shape)
slices[self.axis] = i
# slices = [1, slice(None)], since axis = 0 and i = 1
```
- `slices` becomes `[1, slice(None)]`, meaning we take the 1st row and all columns.
```python
split_tensors.append(A[tuple(slices)].compact().reshape(output_shape))
# A[tuple(slices)] gives Tensor([4, 5, 6])
# reshape(output_shape) keeps it as [4, 5, 6]
```
- This extracts the second row `[4, 5, 6]` and appends it to `split_tensors`.
```python
return tuple(split_tensors)
```
- The final result is
```python
(Tensor([1, 2, 3]), Tensor([4, 5, 6]))
```
#### Case 2: `axis = 1`

In this case, we are splitting along the columns (axis 1), so we expect the output to be three separate columns.

-   **Initial state:**
```python
A.shape = (2, 3)  # 2 rows, 3 columns
axis = 1
```
- **Step-by-step Explanation:**
```python
axis_size = A.shape[self.axis]
# axis_size = 3, since A.shape[1] = 3
```
-   We are splitting along axis 1, which has 3 elements (columns).
```python
split_tensors = []
```
-   We initialize an empty list `split_tensors` to store the split tensors.
```python
output_shape = list(A.shape)
output_shape.pop(self.axis)
# output_shape = [2], since we removed the axis 1 dimension
```
- `output_shape` becomes `[2]` because we are removing axis 1 (which had size 3), leaving us with 2 rows.
```python
for i in range(axis_size):  # Loop over the three columns
```
-   We loop through the three elements along axis 1.

**First Iteration (`i=0`):**
```python
slices = [slice(None)] * len(A.shape)
slices[self.axis] = i
# slices = [slice(None), 0], since axis = 1 and i = 0
```
- `slices` becomes `[slice(None), 0]`, meaning we take all rows and the 0th column.

```python
split_tensors.append(A[tuple(slices)].compact().reshape(output_shape))
# A[tuple(slices)] gives Tensor([1, 4])
# reshape(output_shape) keeps it as [1, 4]
```
-   This extracts the first column `[1, 4]` and appends it to `split_tensors`.

**Second Iteration (`i=1`):**
```python
slices = [slice(None)] * len(A.shape)
slices[self.axis] = i
# slices = [slice(None), 1], since axis = 1 and i = 1
```
- `slices` becomes `[slice(None), 1]`, meaning we take all rows and the 1st column.
```python
split_tensors.append(A[tuple(slices)].compact().reshape(output_shape))
# A[tuple(slices)] gives Tensor([2, 5])
# reshape(output_shape) keeps it as [2, 5]
```
-   This extracts the second column `[2, 5]` and appends it to `split_tensors`.

**Third Iteration (`i=2`):**
```python
slices = [slice(None)] * len(A.shape)
slices[self.axis] = i
# slices = [slice(None), 2], since axis = 1 and i = 2
```
- `slices` becomes `[slice(None), 2]`, meaning we take all rows and the 2nd column.
```python
split_tensors.append(A[tuple(slices)].compact().reshape(output_shape))
# A[tuple(slices)] gives Tensor([3, 6])
# reshape(output_shape) keeps it as [3, 6]
```
- This extracts the third column `[3, 6]` and appends it to `split_tensors`.
```python
return tuple(split_tensors)
```
- The final result is:
```python
(Tensor([1, 4]), Tensor([2, 5]), Tensor([3, 6]))
```
#### Summary:

-   **Splitting along axis 0 (rows):**
```css
(Tensor([1, 2, 3]), Tensor([4, 5, 6]))
```
- **Splitting along axis 1 (columns):**
```css
(Tensor([1, 4]), Tensor([2, 5]), Tensor([3, 6]))
```
