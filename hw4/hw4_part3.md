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
### Flipping ndarrays & FlipOp
```python
import numpy as np
import ctypes
```
*Some* utility code for a demonstration below which you can probably ignore. It might be instructive to check out the `offset` function.
```python
# reads off the underlying data array in order (i.e., offset 0, offset 1, ..., offset n)

# i.e., ignoring strides

def  raw_data(X):

X = np.array(X) # copy, thus compact X

return np.frombuffer(ctypes.string_at(X.ctypes.data, X.nbytes), dtype=X.dtype, count=X.size)

  
# Xold and Xnew should reference the same underlying data

def  offset(Xold, Xnew):

assert Xold.itemsize == Xnew.itemsize

# compare addresses to the beginning of the arrays

return (Xnew.ctypes.data - Xold.ctypes.data)//Xnew.itemsize

  

def  strides(X):

return  ', '.join([str(x//X.itemsize) for x in X.strides])

  

def  format_array(X, shape):

assert  len(shape) == 3, "I only made this formatting work for ndims = 3"

def  chunks(l, n):

n = max(1, n)

return (l[i:i+n] for i in  range(0, len(l), n))

a = [str(x) if x >= 10  else  ' ' + str(x) for x in X]

a = ['(' + ' '.join(y) + ')'  for y in [x for x in chunks(a, shape[-1])]]

a = ['|' + ' '.join(y) + '|'  for y in [x for x in chunks(a, shape[-2])]]

return  ' '.join(a)

  

def  inspect_array(X, *, is_a_copy_of):

# compacts X, then reads it off in order

print('Data: %s' % format_array(raw_data(X), X.shape))

# compares address of X to copy_of, thus finding X's offset

print('Offset: %s' % offset(is_a_copy_of, X))

print('Strides: %s' % strides(X))
```
In order to implement the backwards pass of 2D convolution, we will (probably) need a function which _flips_

axes of ndarrays. We say "probably" because you could probably cleverly implement your convolution forward

function to avoid this. However, we think it is easiest to think about this if you have the ability to "flip" the kernel along its vertical and horizontal dimensions.

  

We will try to build up your intuition for the "flip" operation below in order to help you figure out how to implement it in `ndarray.py`. To do that, we explore numpy's `np.flip` function below. One thing to note is that

`flip` is typically implemented by using negative strides and changing the _offset_ of the underlying array.

  

For example, flipping an array on _all_ of its axes is equivalent to reversing the array. In this case, you can imagine that we would want all the strides to be negative, and the offset to be the length of the array (to start at the end of the array and "stride" backwards).

  

Since we did not explicitly support negative strides in our implementation for the last homework, we will merely call `NDArray.make` with them to make our "flipped" array and then immediately call `.compact()`. Other than changing unsigned ints to signed ints in a few places, we suspect your existing `compact` function should not have to change at all to accomodate negative strides. In the .cc and .cu files we distributed, we have already changed the function signatures to reflect this.

  

Alternatively, you could simply implement `flip` in the CPU backend by copying memory, which you _may_ find more intuitive. We suggest following our mini tutorial below to keep your implementation Python-focused, since we believe it is involves approximately the same amount of effort to implement it slightly more naively in C.

Use this array as reference for the other examples:

```python
A = np.arange(1, 25).reshape(3, 2, 4)
inspect_array(A, is_a_copy_of=A)
```
running result
```css
Data: |( 1 2 3 4) ( 5 6 7 8)| |( 9 10 11 12) (13 14 15 16)| |(17 18 19 20) (21 22 23 24)| 
Offset: 0 
Strides: 8, 4, 1
```
We have put brackets around each axis of the array. Notice that for this array, the offset is 0 and the strides are all positive.

See what happens when you flip the array along the last axis below. Note that the `inspect_array` function compacts the array after flipping it so you can see the "logical" order of the data, and the offset is calculated by comparing the address of the **non**-compacted flipped array with that of `is_copy_of`, i.e., the array `A` we looked at above.

That is, we are looking at how numpy calculates the strides and offset for flipped arrays in order to copy this behavior in our own implementation.

```python
inspect_array(np.flip(A, (2,)), is_a_copy_of=A)
```
running result:
```css
Data: |( 4 3 2 1) ( 8 7 6 5)| |(12 11 10 9) (16 15 14 13)| |(20 19 18 17) (24 23 22 21)| 
Offset: 3 
Strides: 8, 4, -1
```
So flipping the last axis reverses the order of the elements within each 4-dimensional "cell", as you can see above. The stride corresponding to the axis we flipped has been negated. And the offset is 3 -- this makes sense, e.g., because we want the new "first" element of the array to be 4, which was at index 3 in `A`.

```python
inspect_array(np.flip(A, (1,)), is_a_copy_of=A)
```
running result:
```css
Data: |( 5 6 7 8) ( 1 2 3 4)| |(13 14 15 16) ( 9 10 11 12)| |(21 22 23 24) (17 18 19 20)| 
Offset: 4 
Strides: 8, -4, 1
```
Again for the middle axis: we negate the middle stride, and the offset is 4, which seems reasonable since we now want the first element to be 5, which was at index 4 in the original array `A`.

```python
inspect_array(np.flip(A, (0,)), is_a_copy_of=A)
```
```css
Data: |(17 18 19 20) (21 22 23 24)| |( 9 10 11 12) (13 14 15 16)| |( 1 2 3 4) ( 5 6 7 8)| 
Offset: 16 
Strides: -8, 4, 1
```
Try to infer the more general algorithm for computing the offset given the axis to flip.

Observe what happens when we flip _all_ axes.
```python
inspect_array(np.flip(A, (0,1,2)), is_a_copy_of=A)
```
running result:
```css
Data: |(24 23 22 21) (20 19 18 17)| |(16 15 14 13) (12 11 10 9)| |( 8 7 6 5) ( 4 3 2 1)| 
Offset: 23 
Strides: -8, -4, -1
```
As mentioned earlier, the offset is then sufficient to point to the last element of the array, and this is just the "reverse order" version of `A`.

When we flip just axes 1 and 0...

```python
inspect_array(np.flip(A, (0,1)), is_a_copy_of=A)
```
running result:
```css
Data: |(21 22 23 24) (17 18 19 20)| |(13 14 15 16) ( 9 10 11 12)| |( 5 6 7 8) ( 1 2 3 4)| 
Offset: 20 
Strides: -8, -4, 1
```
The offset is 20. Looking back on our previous offset computations, do you notice something?

-------------------

  

With this exploration of numpy's ndarray flipping functionality, which uses negative strides and a custom offset,

try to implement `flip` in `ndarray.py`. You also must implement "flip" forward and backward functions in `ops.py`; note that these should be extremely short.

  

**Important:** You should call NDArray.make with the new strides and offset, and then immediately `.compact()` this array. The resulting array is then copied and has positive strides. We want this (less-than-optimal) behavior because we did not account for negative strides in our previous implementation. _Aside:_ If you want, consider where/if negative strides break your implementation. `__getitem__` definitely doesn't work due to how we processed slices; is there anything else? (_Note_: this isn't graded.)

  
Also, if you want to instead add a `flip` operator on the CPU/CUDA backends, that's also okay.


**Code Implementation**
```python
class Flip(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes
        
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        if self.axes is None:
            self.axes = tuple(range(len(a.shape)))  # Flip along all axes
        return array_api.flip(a, self.axes)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return flip(out_grad, self.axes)
        ### END YOUR SOLUTION
```
```python
    def flip(self, axes):
        """
        Flip this ndarray along the specified axes.
        Note: compact() before returning.
        """
        ### BEGIN YOUR SOLUTION
        assert isinstance(axes, tuple), "axes must be a tuple"
        
        # Create new strides and offset for the flipped array
        new_strides = list(self._strides)
        new_offset = self._offset
        
        for axis in axes:
            new_strides[axis] = -new_strides[axis]
            new_offset += (self._shape[axis] - 1) * self._strides[axis]
        # Create the flipped NDArray and immediately compact it
        flipped_array = self.make(self._shape, strides=tuple(new_strides), device=self.device, handle=self._handle, offset=new_offset)
        return flipped_array.compact()
        ### END YOUR SOLUTION
```


### Explanation of `new_strides` and `new_offset`

When flipping an array, both the strides and the offset need to be adjusted to correctly reflect the reversed traversal of the array. Let’s break down how this works with the example code:
```python
# Create new strides and offset for the flipped array
new_strides = list(self._strides)
new_offset = self._offset
```

-   **`new_strides`**: Strides determine how many elements you need to skip in memory to move along each dimension of the array. For instance, a stride of `3` along axis 0 means moving 3 elements in memory to get to the next row. This line makes a copy of the current strides (`self._strides`), which will be modified based on the axes being flipped.
    
-   **`new_offset`**: The offset represents the starting position in memory where the array begins. Initially, this is set to the current offset (`self._offset`), but it will be adjusted as we reverse the array along specific axes, effectively starting at the last element in that dimension.
    

#### Example 1: Strides and Offset for a Simple Array

Consider a 2D array with shape `(3, 3)`:
```python
[[1, 2, 3],
 [4, 5, 6],
 [7, 8, 9]]
```
For this array:

-   `self._strides = (3, 1)`:
    -   A stride of `3` for axis 0 means you move 3 elements to get to the next row.
    -   A stride of `1` for axis 1 means you move 1 element to get to the next column.
-   `self._offset = 0`: The offset starts at the very first element (`1`).

#### Adjusting Strides and Offset in the Loop

When you flip along a specific axis, the stride for that axis is reversed, and the offset is adjusted to account for starting at the last element of that axis.

```python
for axis in axes:
    new_strides[axis] = -new_strides[axis]
    new_offset += (self._shape[axis] - 1) * self._strides[axis]
```

-   **`new_strides[axis] = -new_strides[axis]`**: This line flips the stride for the specified axis, ensuring that traversal happens in reverse order.
-   **`new_offset += (self._shape[axis] - 1) * self._strides[axis]`**: This line updates the offset to the position of the last element in the specified axis. The term `(self._shape[axis] - 1)` calculates how far from the original offset the last element is, and multiplying by `self._strides[axis]` gives the correct number of elements to shift in memory.

#### Example 2: Flipping Along Axis 0

Suppose you flip along axis 0 (`axes = (0,)`) for the 3x3 array:

Original array:
```python
[[1, 2, 3],
 [4, 5, 6],
 [7, 8, 9]]
```
-   **`new_strides[0] = -3`**: This means that instead of moving forward by 3 elements to get to the next row, you now move backwards, effectively flipping the rows.
-   **`new_offset = 0 + (3 - 1) * 3 = 6`**: The offset is updated to point to the last row, starting at element `7`.

The resulting flipped array:
```css
[[7, 8, 9],
 [4, 5, 6],
 [1, 2, 3]]
```
#### Example 3: Flipping Along Axis 1

Now suppose you flip along axis 1 (`axes = (1,)`), flipping the columns of the original array:

Original array:
```python
[[1, 2, 3],
 [4, 5, 6],
 [7, 8, 9]]
```
-   **`new_strides[1] = -1`**: This means that instead of moving forward by 1 element to get to the next column, you now move backwards, effectively flipping the columns.
-   **`new_offset = 0 + (3 - 1) * 1 = 2`**: The offset is updated to point to the last column of the first row, starting at element `3`.

The resulting flipped array:
```css
[[3, 2, 1],
 [6, 5, 4],
 [9, 8, 7]]
```

#### Example 4: Flipping Along Both Axes

If you flip along both axes (`axes = (0, 1)`), the strides for both axes are reversed, and the offset is updated accordingly:

Original array:
```css
[[1, 2, 3],
 [4, 5, 6],
 [7, 8, 9]]
```
-   **`new_strides[0] = -3`**: Flip the rows.
-   **`new_strides[1] = -1`**: Flip the columns.
-   **`new_offset = 0 + (3 - 1) * 3 + (3 - 1) * 1 = 6 + 2 = 8`**: The offset is updated to point to the last element, `9`.

The resulting flipped array:
```css
[[9, 8, 7],
 [6, 5, 4],
 [3, 2, 1]]
 ```
___
