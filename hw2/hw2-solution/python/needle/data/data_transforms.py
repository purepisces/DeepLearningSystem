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
        flip_img = np.random.rand() < self.p
        ### BEGIN YOUR SOLUTION
        if flip_img:
            return np.flip(img, axis=1)  # Flip the image horizontally
        return img  # Return the original image if not flipped
        ### END YOUR SOLUTION


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
