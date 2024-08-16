from typing import List, Optional
from ..data_basic import Dataset
import numpy as np
import struct
import gzip

class MNISTDataset(Dataset):
    def __init__(
        self,
        image_filename: str,
        label_filename: str,
        transforms: Optional[List] = None,
    ):
        ### BEGIN YOUR SOLUTION
        self.imgs, self.labels = self.parse_mnist(image_filename, label_filename)
        self.transforms = transforms
        ### END YOUR SOLUTION

    def __getitem__(self, index) -> object:
        if isinstance(index, slice):
          # Handle slice index
          start, stop, step = index.indices(len(self.imgs))
          # Assume step is always 1 for simplicity
          imgs = []
          labels = []
          for i in range(start, stop):
            img, label = self.__getitem__(i)
            imgs.append(img)
            labels.append(label)
          imgs = np.stack(imgs)
          labels = np.array(labels)
          return imgs, labels

        else:
          img, label = self.imgs[index], self.labels[index]

          img = img.reshape(self.rows, self.cols, 1)
          if self.transforms:
            for transform in self.transforms:
              img = transform(img)
          return img, label


    def __len__(self) -> int:
        ### BEGIN YOUR SOLUTION
        return len(self.imgs)
        ### END YOUR SOLUTION


    def parse_mnist(self, image_filename, label_filename):
      with gzip.open(image_filename, 'rb') as img_file:
        magic, num_img, rows, cols = struct.unpack(">IIII", img_file.read(16))
        self.rows = rows
        self.cols = cols
        assert magic == 2051 # magic number for image file

        imgs = np.frombuffer(img_file.read(), dtype=np.uint8).reshape(num_img, rows*cols)
        # normalized & change type np.float32
        imgs = imgs.astype(np.float32) / 255.0

      with gzip.open(label_filename, 'rb') as lab_file:
        magic, num_labels = struct.unpack(">II", lab_file.read(8))
        assert magic == 2049 # magic number for label file

        labels = np.frombuffer(lab_file.read(), dtype=np.uint8)

      return imgs, labels