from typing import List, Optional
from ..data_basic import Dataset
import numpy as np

import gzip
import struct

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
        # The `index` parameter can either be an integer or a numpy array (as used in the Dataloader's implementation for fetching batch data).
	    # If `index` is an integer, it will fetch a single image and its corresponding label.
	    # If `index` is a numpy array, it will fetch a batch of images and their corresponding labels.
	    #
	    # Shape of `image`: 
	    # - For a single index, `image` will have the shape (784,), representing a flattened 28x28 image.
	    # - For a batch of indices, `image` will have the shape (batch_size, 784), where `batch_size` equals len(index).
	    #
	    # Shape of `label`:
	    # - For a single index, `label` will be a scalar value representing the class label.
	    # - For a batch of indices, `label` will have the shape (batch_size,), where `batch_size` equals len(index).

	    image = self.images[index]
	    label = self.labels[index]

	    if self.transforms:
	        # When applying transformations, `index` is assumed to be a single integer.
	        # Reshape the flattened image from (784,) to (28, 28, -1) before applying transformations.
	        # The "-1" indicates that the channel dimension will be inferred automatically.
	        image = image.reshape((28, 28, -1))
	        
	        # Apply the sequence of transformations to the image.
	        image = self.apply_transforms(image)
	        
	        # After transformations, reshape the image back to its original flattened shape (784,).
	        image = image.reshape(28 * 28)
        # Return a tuple where: 
        # - `image` is a numpy ndarray with shape (batch_size, 784) or (784,) depending on whether `index` is a batch or single index. 
        # - `label` is a numpy ndarray with shape (batch_size,) or a scalar value depending on whether `index` is a batch or single index. 
	    return image, label
        ### END YOUR SOLUTION

    def __len__(self) -> int:
        ### BEGIN YOUR SOLUTION
        return self.images.shape[0]
        ### END YOUR SOLUTION