Modify
```css
DataLoader's: Tensor.make_const(x) to Tensor(x) 
```
## Part 2: CIFAR-10 dataset [10 points]

 
Next, you will write support for the [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) image classification dataset, which consists of 60,000 32x32 color images in 10 classes, with 6,000 images per class. There are 50k training images and 10k test images.

  

Start by implementing the `__init__` function in the `CIFAR10Dataset` class in `python/needle/data/datasets/cifar10_dataset.py`. You can read in the link above how to properly read the CIFAR-10 dataset files you downloaded at the beginning of the homework. Also fill in `__getitem__` and `__len__`. Note that the return shape of the data from `__getitem__` should be in order (3, 32, 32).

  

Copy `python/needle/data/data_transforms.py` and `python/needle/data/data_basic.py` from previous homeworks.

**Code Implementation**
```python
class CIFAR10Dataset(Dataset):
    def __init__(
        self,
        base_folder: str,
        train: bool,
        p: Optional[int] = 0.5,
        transforms: Optional[List] = None
    ):
        """
        Parameters:
        base_folder - cifar-10-batches-py folder filepath
        train - bool, if True load training dataset, else load test dataset
        Divide pixel values by 255. so that images are in 0-1 range.
        Attributes:
        X - numpy array of images
        y - numpy array of labels
        """
        ### BEGIN YOUR SOLUTION
        self.transforms = transforms
        
        # List of training and test file names
        batch_names = [f"data_batch_{i}" for i in range(1, 6)] if train else ["test_batch"]

        # Load and concatenate data from the selected batches
        self.X, self.y = [], []
        for batch_name in batch_names:
            file_path = os.path.join(base_folder, batch_name)
            with open(file_path, 'rb') as f:
                batch = pickle.load(f, encoding='bytes')
            self.X.append(batch[b'data'])
            self.y.append(batch[b'labels'])

        # Concatenate all data and labels from the list of batches, normalize the data to [0, 1]
        self.X = np.concatenate(self.X).reshape(-1, 3, 32, 32).astype(np.float32) / 255.0
        self.y = np.concatenate(self.y)
        ### END YOUR SOLUTION

    def __getitem__(self, index) -> object:
        """
        Returns the image, label at given index
        Image should be of shape (3, 32, 32)
        """
        ### BEGIN YOUR SOLUTION
        if isinstance(index, int):
            img = self.X[index]
            img = self.apply_transforms(img)
        elif isinstance(index, np.ndarray):
            img = np.array([self.apply_transforms(self.X[i]) for i in index], dtype=np.float32)
        label = self.y[index]
        return img, label
        ### END YOUR SOLUTION

    def __len__(self) -> int:
        """
        Returns the total number of examples in the dataset
        """
        ### BEGIN YOUR SOLUTION
        return len(self.X)
        ### END YOUR SOLUTION
```


```python
import numpy as np
from ..autograd import Tensor

from typing import Iterator, Optional, List, Sized, Union, Iterable, Any

class Dataset:
    r"""An abstract class representing a `Dataset`.

    All subclasses should overwrite :meth:`__getitem__`, supporting fetching a
    data sample for a given key. Subclasses must also overwrite
    :meth:`__len__`, which is expected to return the size of the dataset.
    """

    def __init__(self, transforms: Optional[List] = None):
        self.transforms = transforms

    def __getitem__(self, index) -> object:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError
    
    def apply_transforms(self, x):
        if self.transforms is not None:
            # apply the transforms
            for tform in self.transforms:
                x = tform(x)
        return x


class DataLoader:
    r"""
    Data loader. Combines a dataset and a sampler, and provides an iterable over
    the given dataset.
    Args:
        dataset (Dataset): dataset from which to load the data.
        batch_size (int, optional): how many samples per batch to load
            (default: ``1``).
        shuffle (bool, optional): set to ``True`` to have the data reshuffled
            at every epoch (default: ``False``).
     """
    dataset: Dataset
    batch_size: Optional[int]

    def __init__(
        self,
        dataset: Dataset,
        batch_size: Optional[int] = 1,
        shuffle: bool = False,
    ):

        self.dataset = dataset
        self.shuffle = shuffle
        self.batch_size = batch_size
        if not self.shuffle:
            self.ordering = np.array_split(np.arange(len(dataset)), 
                                          range(batch_size, len(dataset), batch_size))

    def __iter__(self):
        ### BEGIN YOUR SOLUTION
        if self.shuffle:
            # Generate a new shuffled array of indices
            # e.g. permutation = np.random.permutation(10) = array([2 8 4 9 1 0 7 3 6 5])
            permutation = np.random.permutation(len(self.dataset))
            # Split the shuffled indices into batches
            # e.g. self.ordering = np.array_split(array([2, 8, 4, 9, 1, 0, 7, 3, 6, 5]), [3, 6, 9]) 
            # self.ordering = [
            #   array([2, 8, 4]),  # First batch
            #   array([9, 1, 0]),  # Second batch
            #   array([7, 3, 6]),  # Third batch
            #   array([5])         # Remaining elements in the final batch
            # ]
            self.ordering = np.array_split(permutation, range(self.batch_size, len(self.dataset), self.batch_size))
        self.index = 0
        ### END YOUR SOLUTION
        return self

    def __next__(self):
        ### BEGIN YOUR SOLUTION
        if self.index >= len(self.ordering):
            raise StopIteration  # No more batches to return
            
        # Retrieve the batch of indices for the current batch
        # Example: batch_indices could be something like array([9, 1, 0])
        batch_indices = self.ordering[self.index]
        
        # Fetch the actual data for these indices from the dataset
        # For example, if batch_indices = array([9, 1, 0]), then:
        # self.dataset[batch_indices] will call `__getitem__` method in Dataset Class, and might return a tuple like:
        # 
        # self.dataset[batch_indices]:
        # (
        #     array([
        #         [0., 0., 0., ..., 0., 0., 0.],
        #         [0., 0., 0., ..., 0., 0., 0.],
        #         [0., 0., 0., ..., 0., 0., 0.],
        #         [0., 0., 0., ..., 0., 0., 0.],
        #         [0., 0., 0., ..., 0., 0., 0.]
        #     ], dtype=float32),
        #     
        #     array([7, 2, 1, 0, 4], dtype=uint8)
        # )
        # 
        # This could be represented a batch of image data and corresponding labels:
        # (ndarray(image7, image2, image1, image0, image4), ndarray(label7, label2, label1, label0, label4))
        #
        # In the for loop, 'x' would be an ndarray for either image data or labels.
        # Convert the dataset batch into the desired format, e.g., Tensor objects
        # Example: Tensor(x) will wrap each element in the batch not using`Tensor.make_const(x)`
        #
        # Incorrect behavior when using Tensor.make_const(x):
        # - The cached data for the tensors was of type `numpy.ndarray`, causing the test to fail.
        # - Example of incorrect output:
        #   Type of X.cached_data: <class 'numpy.ndarray'>
        #   Result: FAILED
        #
        # Correct behavior when using Tensor(x):
        # - Using Tensor(x) ensures that the cached data is correctly converted to `needle.backend_ndarray.ndarray.NDArray`, passing the test.
        # - Example of correct output:
        #   Type of X.cached_data: <class 'needle.backend_ndarray.ndarray.NDArray'>
        #   Result: PASSED
        #
        # Example of what batch_data might look like:
        # [needle.Tensor([[0. 0. 0. ... 0. 0. 0.],
        #                 [0. 0. 0. ... 0. 0. 0.],
        #                 [0. 0. 0. ... 0. 0. 0.],
        #                 ...
        #                 [0. 0. 0. ... 0. 0. 0.],
        #                 [0. 0. 0. ... 0. 0. 0.],
        #                 [0. 0. 0. ... 0. 0. 0.]]), 
        #  needle.Tensor([7, 2, 1, 0, 4, 1, 4, 9, 5, 9])]
        batch_data = [Tensor(x) for x in self.dataset[batch_indices]]
        # Move to the next batch for the next iteration
        self.index += 1
        
        # Return the batch data
        return batch_data
        ### END YOUR SOLUTION
```
