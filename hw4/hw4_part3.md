
## Part 3: Convolutional neural network [40 points]

Here's an outline of what you will do in this task.


In `python/needle/backend_ndarray/ndarray.py`, implement:

- `flip`

- `pad`

  
In `python/needle/ops_mathematic.py`, implement (forward and backward):

- `Flip`

- `Dilate`

- `UnDilate`

- `Conv`


In `python/needle/nn/nn_conv.py`, implement:

- `Conv`

  

In `apps/models.py`, fill in the `ResNet9` class.

  
In `apps/simple_ml.py`, fill in:

- `epoch_general_cifar10`,

- `train_cifar10`

- `evaluate_cifar10`

  

We have provided a `BatchNorm2d` implementation in `python/needle/nn/nn_basic.py` for you as a wrapper around your previous `BatchNorm1d` implementation.

  

**Note**: Remember to copy the solution of `nn_basic.py` from previous homework, make sure to not overwrite the `BatchNorm2d` module.

___

### Padding ndarrays

  

Convolution as typically implemented in deep learning libraries cuts down the size of inputs;

e.g., a (1, 32, 32, 3) image convolved with a 3x3 filter would give a (1, 30, 30, c) output.

A way around this is to pad the input ndarray before performing convolution, e.g., pad with zeros to get a (1, 34, 34, 3) ndarray so that the result is (1, 32, 32, 3).

  

Padding is also required for the backward pass of convolution.

  

You should implement `pad` in `ndarray.py` to closely reflect the behavior of `np.pad`.

That is, `pad` should take a tuple of 2-tuples with length equal to the number of dimensions of the array,

where each element in the 2-tuple corresponds to "left padding" and "right padding", respectively.

For example, if `A` is a (10, 32, 32, 8) ndarray (think NHWC), then `A.pad( (0, 0), (2, 2), (2, 2), (0, 0) )` would be a (10, 36, 36, 8) ndarray where the "spatial" dimension has been padded by two zeros on all sides.

**Code Implementation**

```python
def pad(self, axes):
        """
        Pad this ndarray by zeros by the specified amount in `axes`,
        which lists for _all_ axes the left and right padding amount, e.g.,
        axes = ( (0, 0), (1, 1), (0, 0)) pads the middle axis with a 0 on the left and right side.
        """
        ### BEGIN YOUR SOLUTION
        new_shape = tuple(self._shape[i] + axes[i][0] + axes[i][1] for i in range(len(self._shape)))
        new_array = self.make(new_shape, device=self.device)
        pad_slices = tuple(slice(axes[i][0], axes[i][0] + self._shape[i]) for i in range(len(self._shape)))
        new_array[pad_slices] = self
        return new_array
        ### END YOUR SOLUTION
```

### Example Setup

Suppose we have an `NDArray` with the following properties:

-   **Shape:** `(2, 3)` → This means the array has 2 rows and 3 columns, i.e., a 2x3 matrix.
    
-   **Array Data:** Let's assume the contents of the array are:
```css
[[1, 2, 3],
 [4, 5, 6]]
```
Now, let's say we want to pad this array. We'll pad with zeros on both sides of the second dimension (columns) and the first dimension (rows). We'll use the following padding amounts:

-   **Padding amounts (axes):** `((1, 1), (2, 2))`
    -   This means:
        -   Pad 1 row before and 1 row after the first dimension (rows).
        -   Pad 2 columns before and 2 columns after the second dimension (columns).

### Step-by-Step Breakdown of `pad` Function

1.  **Calculate New Shape:**
```python
new_shape = tuple(self._shape[i] + axes[i][0] + axes[i][1] for i in range(len(self._shape)))
```
-   -   We are iterating over each dimension of the original array's shape to calculate the new shape after padding.
    -   The new shape is calculated by adding the original size of each dimension (`self._shape[i]`) and the padding on both sides of that dimension (`axes[i][0] + axes[i][1]`).
    
    In this example:
    
    -   For the first dimension (rows): Original size = 2, padding = 1 (before) + 1 (after) → New size = 2 + 1 + 1 = 4.
    -   For the second dimension (columns): Original size = 3, padding = 2 (before) + 2 (after) → New size = 3 + 2 + 2 = 7.
    
    So, the new shape of the array will be `(4, 7)`.
2.  **Create New Array with Padded Shape:**
```python
new_array = self.make(new_shape, device=self.device)
```
Here, we create a new `NDArray` of shape `(4, 7)` using the same device as the original array (so if it's on CPU, it remains on CPU; if it's on GPU, it remains on GPU). Initially, this array is filled with zeros.

So, `new_array` will look like this:
```python
[[0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0]]
```
3.  **Determine the Slices Where the Original Array Will Fit:**
```python
pad_slices = tuple(slice(axes[i][0], axes[i][0] + self._shape[i]) for i in range(len(self._shape)))
```
-   This line creates a tuple of slices that represent where the original array will be placed inside the new padded array.
    
    For each dimension, the slice starts at the left padding (`axes[i][0]`) and ends at the left padding plus the size of the original dimension (`axes[i][0] + self._shape[i]`).
    
    In this example:
    
    -   For the first dimension (rows): The slice is from 1 to 3 (`slice(1, 1 + 2)`, because we pad with 1 row before and the original array has 2 rows).
    -   For the second dimension (columns): The slice is from 2 to 5 (`slice(2, 2 + 3)`, because we pad with 2 columns before and the original array has 3 columns).
    
    So, `pad_slices` will be `(slice(1, 3), slice(2, 5))`.
    
4.   **Place the Original Array in the Padded Array:**
```python
new_array[pad_slices] = self
```
Now, the original array is placed into the new padded array at the positions determined by `pad_slices`.

After this operation, the `new_array` will look like this:
```python
[[0, 0, 0, 0, 0, 0, 0],
 [0, 0, 1, 2, 3, 0, 0],
 [0, 0, 4, 5, 6, 0, 0],
 [0, 0, 0, 0, 0, 0, 0]]
```
5.  **Return the Padded Array:**
    
    Finally, the function returns the padded `NDArray`.
    

### Final Result:

The original 2x3 array:

```python
[[1, 2, 3],
 [4, 5, 6]]
```
Becomes the padded 4x7 array:
```css
[[0, 0, 0, 0, 0, 0, 0],
 [0, 0, 1, 2, 3, 0, 0],
 [0, 0, 4, 5, 6, 0, 0],
 [0, 0, 0, 0, 0, 0, 0]]
```
### Key Points:

-   **Padding amounts** determine how many zeros are added before and after each dimension.
-   The **new shape** is calculated by adding the padding amounts to each dimension of the original array.
-   **Slicing** is used to place the original array into the correctly padded section of the new array.
___
