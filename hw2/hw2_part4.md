## Question 4

In this question, you will implement two data primitives: `needle.data.DataLoader` and `needle.data.Dataset`. `Dataset` stores the samples and their corresponding labels, and `DataLoader` wraps an iterable around the `Dataset` to enable easy access to the samples.

For this question, you will be working in the `python/needle/data` directory.


### Transformations

First we will implement a few transformations that are helpful when working with images. We will stick with a horizontal flip and a random crop for now. Fill out the following functions in `needle/data/data_transforms.py`.

___

#### RandomFlipHorizontal

`needle.data.RandomFlipHorizontal(p = 0.5)`

Flips the image horizontally, with probability `p`.

##### Parameters

- `p` (*float*) - The probability of flipping the input image.


Code Implementation:
```python
class RandomFlipHorizontal(Transform):
    def __init__(self, p = 0.5):
        self.p = p

    def __call__(self, img):
        """
        Horizonally flip an image, specified as an H x W x C NDArray.
        Args:
            img: H x W x C NDArray of an image
        Returns:
            H x W x C ndarray corresponding to image flipped with probability self.p
        Note: use the provided code to provide randomness, for easier testing
        """
        flip_img = np.random.rand() < self.p
        ### BEGIN YOUR SOLUTION
        if flip_img:
            return np.flip(img, axis=1)  # Flip the image horizontally
        return img  # Return the original image if not flipped
        ### END YOUR SOLUTION
```
___


### Explanation of RandomFlipHorizontal

The `RandomFlipHorizontal` class is designed to randomly flip an image horizontally with a certain probability. This is a common data augmentation technique used in training machine learning models, particularly in computer vision tasks, to help the model generalize better by introducing variability in the training data.

- **Input**: `img` is a NumPy array representing an image with dimensions Height x Width x Channels (`H x W x C`).

### Explanation of whole code

**Random Decision (`flip_img`)**:

-   `flip_img = np.random.rand() < self.p`:
    -   This line generates a random number between 0 and 1 using `np.random.rand()`.
    -   It then checks if this random number is less than `self.p` (the probability of flipping the image).
    -   If `flip_img` is `True`, the image will be flipped; if `False`, the image will remain unchanged.

**Flipping the Image**:

-   The `if flip_img:` block checks if the image should be flipped.
-   `np.flip(img, axis=1)`: If `flip_img` is `True`, this line flips the image horizontally by reversing the order of columns (flipping along the width axis, `axis=1`).
-   `return img`: If `flip_img` is `False`, the original image is returned without any modifications.

> **Randomness**: The flipping decision is based on randomness. For example, with `self.p = 0.5`, there is a 50% chance the image will be flipped and a 50% chance it will remain the same.
### Understanding `axis=1` in `np.max(axis=1)` vs. `np.flip(img, axis=1)`


1. **`np.max(axis=1)`**:
   - **Purpose**: Finds the maximum value along a specified axis.
   - **Behavior on a 2D Array**: Computes the maximum value for each row across all columns.
   - **Result**: Produces a 1D array where each element is the maximum value of the corresponding row.

```python
   import numpy as np

   array = np.array([[1, 2, 3, 4],
                     [5, 6, 7, 8],
                     [9, 10, 11, 12]])

   max_values = np.max(array, axis=1)
   print(max_values)  # Output: [4, 8, 12]
```

2. **`np.flip(img, axis=1)`**:

-   **Purpose**: Reverses the order of elements along a specified axis.
-   **Behavior on a 2D Array**: Reverses the order of columns within each row.
-   **Result**: Produces a 2D array where each row's elements are flipped horizontally.

```python
array = np.array([[1, 2, 3, 4],
                  [5, 6, 7, 8],
                  [9, 10, 11, 12]])

flipped_array = np.flip(array, axis=1)
print(flipped_array)
# Output:
# [[ 4  3  2  1]
#  [ 8  7  6  5]
#  [12 11 10  9]]
```
#### Summary

-   **`np.max(axis=1)`**: Finds the maximum value across columns for each row.
-   **`np.flip(axis=1)`**: Reverses the order of elements within each row, flipping the array horizontally.


