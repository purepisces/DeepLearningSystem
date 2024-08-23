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

___
#### RandomCrop

`needle.data.RandomCrop(padding=3)`

Padding is added to all sides of the image, and then the image is cropped back to it's original size at a random location. Returns an image the same size as the original image.

##### Parameters

- `padding` (*int*) - The padding on each border of the image.

Code Implementation:
```python
class RandomCrop(Transform):
    def __init__(self, padding=3):
        self.padding = padding

    def __call__(self, img):
        """ Zero pad and then randomly crop an image.
        Args:
             img: H x W x C NDArray of an image
        Return 
            H x W x C NAArray of cliped image
        Note: generate the image shifted by shift_x, shift_y specified below
        """
        shift_x, shift_y = np.random.randint(low=-self.padding, high=self.padding+1, size=2)
        ### BEGIN YOUR SOLUTION
        # Pad the image with zeros on all sides
        padded_img = np.pad(img, ((self.padding, self.padding), (self.padding, self.padding), (0, 0)), mode='constant')
        # Determine the crop coordinates
        # The shift_x determines how much the crop will move vertically (up or down)
        # The shift_y determines how much the crop will move horizontally (left or right)
        start_x = shift_x + self.padding
        start_y = shift_y + self.padding
        # img.shape[0] is the height of the original image (number of rows)
        # img.shape[1] is the width of the original image (number of columns)
        end_x = start_x + img.shape[0]
        end_y = start_y + img.shape[1]

        # Crop the image back to its original size
        cropped_img = padded_img[start_x:end_x, start_y:end_y, :]

        return cropped_img
        ### END YOUR SOLUTION
```
### Explanation of `RandomCrop`

The `RandomCrop` transformation is designed to add padding to an image and then crop it back to its original size, but from a randomly shifted position. This technique is commonly used in data augmentation for training machine learning models, as it introduces variability in the training data by presenting slightly altered versions of the same image.

### Understanding `np.random.randint(low=-self.padding, high=self.padding+1, size=2)`

The function `np.random.randint(low=-self.padding, high=self.padding+1, size=2)` generates two random integers within a specified range. Here's a detailed explanation:

#### Breakdown of the Arguments

1.  **`low=-self.padding`**:
    
    -   This is the minimum value that the generated integers can be.
    -   By setting `low=-self.padding`, you're allowing the random integers to be as low as `-self.padding`. For example, if `self.padding = 3`, then the lowest possible value for the generated integers will be `-3`.
2.  **`high=self.padding+1`**:
    
    -   This is the upper bound for the generated integers, but it's **exclusive**. This means the actual highest value will be `self.padding`.
    -   For example, if `self.padding = 3`, then the highest possible value for the generated integers will be `3`, because `np.random.randint` does not include the `high` value itself (it only goes up to `high-1`).
3.  **`size=2`**:
    
    -   This specifies how many random integers you want to generate.
    -   In this case, `size=2` means that the function will return an array with two random integers.

#### Purpose

-   **`shift_x, shift_y = np.random.randint(low=-self.padding, high=self.padding+1, size=2)`**:
    -   This line generates two random integers within the range from `-self.padding` to `self.padding`.
    -   The first integer, `shift_x`, will determine how much the image will be shifted horizontally.
    -   The second integer, `shift_y`, will determine how much the image will be shifted vertically.

#### Example

If `self.padding = 3`, the function call will generate two random integers within the range `-3` to `3`. These integers might look something like this:

-   `shift_x = 2`
-   `shift_y = -1`

This means the image will be shifted 2 pixels to the right (`shift_x = 2`) and 1 pixel upwards (`shift_y = -1`).

#### Summary

-   `np.random.randint(low=-self.padding, high=self.padding+1, size=2)` is used to generate two random integers that specify the horizontal and vertical shifts for cropping an image after padding. These shifts help to add randomness and variability to the data augmentation process.

### Understanding `np.pad(img, ((self.padding, self.padding), (self.padding, self.padding), (0, 0)), mode='constant')`

The `np.pad()` function in NumPy is used to add padding around an array. In the context of images, padding typically means adding extra rows or columns of pixels around the edges of the image. The `mode` parameter specifies what values should be used for the padding, and `mode='constant'` means that the padding will be filled with a constant value (usually 0).

Let's break down the specific call to `np.pad()`:
```python
padded_img = np.pad(img, ((self.padding, self.padding), (self.padding, self.padding), (0, 0)), mode='constant')
```
#### Parameters Explained

1.  **`img`**:
    
    -   This is the input image that you want to pad. It is expected to be a NumPy array with dimensions `H x W x C`, where:
        -   `H` is the height of the image.
        -   `W` is the width of the image.
        -   `C` is the number of channels (e.g., 3 for an RGB image, where the channels represent Red, Green, and Blue).
2.  **`((self.padding, self.padding), (self.padding, self.padding), (0, 0))`**:
    
    -   This tuple specifies the amount of padding to add to each dimension of the array.
    -   The tuple has the following structure: `((before_1, after_1), (before_2, after_2), (before_3, after_3))`, where:
        -   `before_1` and `after_1` refer to the amount of padding to add before and after the first dimension (height).
        -   `before_2` and `after_2` refer to the amount of padding to add before and after the second dimension (width).
        -   `before_3` and `after_3` refer to the amount of padding to add before and after the third dimension (channels).
    
    In the specific call:
    
    -   `self.padding` is the amount of padding you want to add.
    -   `(self.padding, self.padding)` means that you're adding `self.padding` pixels of padding before and after both the height and width dimensions of the image.
    -   `(0, 0)` means no padding is added to the third dimension (the channels).
3.  **`mode='constant'`**:
    
    -   This specifies the padding mode. `'constant'` means that the padding will be filled with a constant value, which defaults to 0. This is equivalent to adding black pixels around the image when working with images.

#### Example

Suppose `self.padding = 2` and `img` is a 3x3 RGB image (with 3 channels):

```python
import numpy as np

img = np.array([[[1, 1, 1], [2, 2, 2], [3, 3, 3]],
                [[4, 4, 4], [5, 5, 5], [6, 6, 6]],
                [[7, 7, 7], [8, 8, 8], [9, 9, 9]]])

padded_img = np.pad(img, ((2, 2), (2, 2), (0, 0)), mode='constant')
print(padded_img)
```
#### Output:

The output `padded_img` will be a 7x7 RGB image:
```python
[[[0 0 0] [0 0 0] [0 0 0] [0 0 0] [0 0 0] [0 0 0] [0 0 0]]
 [[0 0 0] [0 0 0] [0 0 0] [0 0 0] [0 0 0] [0 0 0] [0 0 0]]
 [[0 0 0] [0 0 0] [1 1 1] [2 2 2] [3 3 3] [0 0 0] [0 0 0]]
 [[0 0 0] [0 0 0] [4 4 4] [5 5 5] [6 6 6] [0 0 0] [0 0 0]]
 [[0 0 0] [0 0 0] [7 7 7] [8 8 8] [9 9 9] [0 0 0] [0 0 0]]
 [[0 0 0] [0 0 0] [0 0 0] [0 0 0] [0 0 0] [0 0 0] [0 0 0]]
 [[0 0 0] [0 0 0] [0 0 0] [0 0 0] [0 0 0] [0 0 0] [0 0 0]]]
```
#### Summary

-   **Padding**: Adds `self.padding` rows of zeros at the top and bottom, and `self.padding` columns of zeros at the left and right of the image.
-   **Mode**: The `'constant'` mode fills these added rows and columns with zeros, effectively creating a black border around the original image.

### Understanding the Crop Coordinates

```python
  # Determine the crop coordinates
  # The shift_x determines how much the crop will move vertically (up or down)
  # The shift_y determines how much the crop will move horizontally (left or right)
  start_x = shift_x + self.padding
  start_y = shift_y + self.padding
  # img.shape[0] is the height of the original image (number of rows)
  # img.shape[1] is the width of the original image (number of columns)
  end_x = start_x + img.shape[0]
  end_y = start_y + img.shape[1]

  # Crop the image back to its original size
  cropped_img = padded_img[start_x:end_x, start_y:end_y, :]

  return cropped_img
```

After padding an image, the original content starts at an offset position within the padded image. The crop coordinates (`start_x`, `start_y`) are adjusted based on both the padding and a random shift (`shift_x`, `shift_y`) to select different regions of the original image for cropping.

#### Example
**Original Image (3x3):**
```python
1 2 3
4 5 6
7 8 9
```
**Padded Image (Padding = 2):**
```python
0 0 0 0 0 0 0
0 0 0 0 0 0 0
0 0 1 2 3 0 0
0 0 4 5 6 0 0
0 0 7 8 9 0 0
0 0 0 0 0 0 0
0 0 0 0 0 0 0
```
Here, the original image content is surrounded by padding. The original image content starts at position `(2, 2)` in the padded image due to the 2-pixel padding on each side.

**Example with `shift_x = 1` and `shift_y = 0`:**

-   Without any shift, the crop would start at `(2, 2)` in the padded image and include the entire original image content.
-   With `shift_x = 1`, the crop is shifted downward by one pixel, so it now starts at `(3, 2)`.

**Cropped Area with Shift_x = 1:**
```python
4 5 6
7 8 9
0 0 0
```
In this example, the crop has shifted downward by one row. As a result, the top row of the original image (i.e., `1 2 3`) is excluded from the crop, while the rest of the original image is captured along with some padding. This downward shift introduces variability in the training data by presenting slightly altered views of the same image to the model.


#### Explanation of shift_x and shift_y

-   **`shift_x`** causes a  vertical (up-down) shift.
-   **`shift_y`** causes a horizontal (left-right) shift.

Example to illustrate:
```python
import numpy as np

# Create a 3D NumPy array representing a 3x3 image with 3 color channels
img = np.array([[[1, 1, 1], [2, 2, 2], [3, 3, 3]],
                [[4, 4, 4], [5, 5, 5], [6, 6, 6]],
                [[7, 7, 7], [8, 8, 8], [9, 9, 9]]])

print(img[0])
# Output:
# [[1 1 1]
#  [2 2 2]
#  [3 3 3]]

print(img[2][0][0])
# Output: 7
```
___
### Dataset

 
Each `Dataset` subclass must implement three functions: `__init__`, `__len__`, and `__getitem__`. The `__init__` function initializes the images, labels, and transforms. The `__len__` function returns the number of samples in the dataset. The `__getitem__` function retrieves a sample from the dataset at a given index `idx`, calls the transform functions on the image (if applicable), converts the image and label to a numpy array (the data will be converted to Tensors elsewhere). The output of `__getitem__` and `__next__` should be NDArrays, and you should follow the shapes such that you're accessing an array of size (Datapoint Number, Feature Dim 1, Feature Dim 2, ...).

Fill out these functions in the `MNISTDataset` class in `needle/data/datasets/mnist_dataset.py`. You can use your solution to `parse_mnist` from the previous homework for the `__init__` function.

  
### MNISTDataset

`needle.data.MNISTDataset(image_filesname, label_filesname, transforms)`

  

##### Parameters

- `image_filesname` - path of file containing images

- `label_filesname` - path of file containing labels

- `transforms` - an optional list of transforms to apply to data

Code Implementation:
```python
from typing import List, Optional
from ..data_basic import Dataset
import numpy as np

import gzip

def parse_mnist(image_filesname, label_filename):
    """Read an images and labels file in MNIST format.  See this page:
    http://yann.lecun.com/exdb/mnist/ for a description of the file format.

    Args:
        image_filename (str): name of gzipped images file in MNIST format
        label_filename (str): name of gzipped labels file in MNIST format

    Returns:
        Tuple (X,y):
            X (numpy.ndarray[np.float32]): 2D numpy array containing the loaded
                data.  The dimensionality of the data should be
                (num_examples x input_dim) where 'input_dim' is the full
                dimension of the data, e.g., since MNIST images are 28x28, it
                will be 784.  Values should be of type np.float32, and the data
                should be normalized to have a minimum value of 0.0 and a
                maximum value of 1.0.

            y (numpy.ndarray[dypte=np.int8]): 1D numpy array containing the
                labels of the examples.  Values should be of type np.int8 and
                for MNIST will contain the values 0-9.
    """
    ### BEGIN YOUR SOLUTION
    # Read the labels file
    with gzip.open(label_filename, 'rb') as lbl_f:
        magic, num_items = struct.unpack(">II", lbl_f.read(8))
        if magic != 2049:
            raise ValueError(f"Invalid magic number in label file: {magic}")
        labels = np.frombuffer(lbl_f.read(num_items), dtype=np.uint8)
    
    # Read the images file
    with gzip.open(image_filesname, 'rb') as img_f:
        magic, num_images, num_rows, num_cols = struct.unpack(">IIII", img_f.read(16))
        if magic != 2051:
            raise ValueError(f"Invalid magic number in image file: {magic}")
        images = np.frombuffer(img_f.read(num_images * num_rows * num_cols), dtype=np.uint8)
        images = images.reshape(num_images, num_rows * num_cols).astype(np.float32)
        images /= 255.0  # Normalize to range [0, 1]
    
    return images, labels
    ### END YOUR SOLUTION

class MNISTDataset(Dataset):
    def __init__(
        self,
        image_filename: str,
        label_filename: str,
        transforms: Optional[List] = None,
    ):
        ### BEGIN YOUR SOLUTION
        self.images, self.labels = parse_mnist(image_filename, label_filename)
        self.transforms = transforms
        ### END YOUR SOLUTION

    def __getitem__(self, index) -> object:
        ### BEGIN YOUR SOLUTION
        # Shape of image: (784,), label is a single scalar value
        image = self.images[index]
        label = self.labels[index]
        if self.transforms:
            # Reshape the flat image (784,) to (28, 28, -1) before applying transforms
            image = image.reshape((28, 28, -1))
            image = self.apply_transforms(image)
            image = image.reshape(28 * 28)
        return image, label
        ### END YOUR SOLUTION

    def __len__(self) -> int:
        ### BEGIN YOUR SOLUTION
        return self.images.shape[0]
        ### END YOUR SOLUTION
```

### Explanation of the whole code

The provided code defines a custom dataset class `MNISTDataset` that is designed to handle the MNIST dataset, which consists of handwritten digit images and their corresponding labels. This class is a subclass of a base class `Dataset` and implements the required methods for working with the data: `__init__`, `__len__`, and `__getitem__`.

#### Breakdown of the Code

1.  **`parse_mnist` Function**:
    
    -   **Purpose**: This function reads and parses the MNIST image and label files. The MNIST data is stored in a specific binary format, so this function extracts the images and labels, normalizes the images, and returns them as NumPy arrays.
    -   **Details**:
        -   The labels are read and stored as a 1D array of integers (`labels`), where each value corresponds to the digit in the image (0-9).
        -   The images are read and stored as a 2D array (`images`) with shape `(num_images, 784)`, where each row represents a flattened 28x28 pixel image.
        -   The images are normalized to have values between 0.0 and 1.0 by dividing by 255.0.
2.  **`MNISTDataset` Class**:
    
    -   **`__init__` Method**:
        
        -   **Purpose**: Initializes the dataset by loading the images and labels using the `parse_mnist` function and optionally applies a list of transformations to the data.
        -   **Details**: The `image_filename` and `label_filename` are passed to `parse_mnist` to load the data, and any transformations provided are stored for later use.
    -   **`__getitem__` Method**:
        
        -   **Purpose**: Retrieves a single sample (image and label) from the dataset based on the given index.
        -   **Details**:
            -   The image corresponding to the specified `index` is retrieved from `self.images`. Since `self.images` is a 2D array with each row representing a flattened image, the shape of `image` is `(784,)`.
            -   The corresponding label is also retrieved from `self.labels` as a scalar value.
            -   If transformations are provided, the image is reshaped into a 3D array with shape `(28, 28, 1)` to allow spatial transformations (such as flipping or cropping) to be applied correctly.
            -   After applying the transformations, the image is reshaped back to its flattened form `(784,)`.
            -   The method returns the transformed image and its label.
    -   **`__len__` Method**:
        
        -   **Purpose**: Returns the number of samples in the dataset.
        -   **Details**: The number of images (and hence the number of labels) is determined by the first dimension of `self.images`, which is accessed using `self.images.shape[0]`. This provides the total number of data points in the dataset.

#### Summary:

-   The `MNISTDataset` class allows for loading, transforming, and accessing individual samples from the MNIST dataset.
-   The `__getitem__` method handles retrieving and optionally transforming each image, ensuring that spatial transformations are applied correctly by reshaping the image into its original dimensions before and after transformations.
-   The `__len__` method provides the total number of images in the dataset, making it easy to iterate over the dataset in a structured manner.

### Explanation of `__getitem__`

The transformations, such as `RandomFlipHorizontal` and `RandomCrop`, expect the input image to be in a 3D format where `H` represents height, `W` represents width, and `C` represents the number of channels. This is crucial because these transformations operate on the spatial dimensions of the image. Without reshaping the flat MNIST image array (originally 784 elements) into a 3D format (28x28 with 1 channel), the transformations wouldn't be able to correctly interpret and manipulate the image. After applying the transformations, the image is often reshaped back to a flat array if required by the downstream model, ensuring both compatibility and correct functionality of the transformations.

```python
import numpy as np

class Transform:
    def __call__(self, x):
        raise NotImplementedError

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

class RandomCrop(Transform):
    def __init__(self, padding=3):
        self.padding = padding

    def __call__(self, img):
        """ Zero pad and then randomly crop an image.
        Args:
             img: H x W x C NDArray of an image
        Return 
            H x W x C NAArray of cliped image
        Note: generate the image shifted by shift_x, shift_y specified below
        """
 ```
### Choosing Between `reshape` Method and `np.reshape` Function in NumPy
When working with NumPy arrays, you have two options for reshaping: using the instance method `reshape` directly on the array, or using the standalone function `np.reshape`.
```python
import numpy as np

# Example array
image = np.arange(784)

# Using the instance method
image_reshaped = image.reshape(28, 28)

# Using the np.reshape function
image_reshaped_via_np = np.reshape(image, (28, 28))
```
Both methods are correct and will give the same result, but `image.reshape(...)` is generally preferred for its readability and directness when you're already working with the array.
