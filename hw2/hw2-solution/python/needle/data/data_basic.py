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
        # Example: Tensor.make_const(x) will wrap each element in the batch

        # Example of what batch_data might look like:
        # [needle.Tensor([[0. 0. 0. ... 0. 0. 0.],
        #                 [0. 0. 0. ... 0. 0. 0.],
        #                 [0. 0. 0. ... 0. 0. 0.],
        #                 ...
        #                 [0. 0. 0. ... 0. 0. 0.],
        #                 [0. 0. 0. ... 0. 0. 0.],
        #                 [0. 0. 0. ... 0. 0. 0.]]), 
        #  needle.Tensor([7, 2, 1, 0, 4, 1, 4, 9, 5, 9])]
        batch_data = [Tensor.make_const(x) for x in self.dataset[batch_indices]]
        
        # Move to the next batch for the next iteration
        self.index += 1
    
        # Return the batch data
        return batch_data
        ### END YOUR SOLUTION
