```python
  with gzip.open(label_filename, 'rb') as lbl_f:
        magic, num_items = struct.unpack(">II", lbl_f.read(8))
        if magic != 2049:
            raise ValueError(f"Invalid magic number in label file: {magic}")
        labels = np.frombuffer(lbl_f.read(num_items), dtype=np.uint8)
```

```python
with gzip.open(label_filename, 'rb') as lbl_f:
  magic, num_items = struct.unpack(">II", lbl_f.read(8))
```
Purpose: Read and unpack the first 8 bytes of the file to get the magic number and the number of items (labels).
lbl_f.read(8): Reads the first 8 bytes from the file. These 8 bytes contain two 4-byte integers: the magic number and the number of items.
struct.unpack(">II", ...): Unpacks the 8 bytes into two 32-bit unsigned integers (I), in big-endian order (>). The result is a tuple containing the magic number and the number of items.
`>`: Indicates big-endian byte order.
I: Indicates a 32-bit unsigned integer.

Big Endian:

The most significant byte (the "big end") is stored or transmitted first.
For example, the 32-bit integer 0x12345678 would be stored as:
```
Byte 0: 0x12
Byte 1: 0x34
Byte 2: 0x56
Byte 3: 0x78
```
Little Endian:

The least significant byte (the "little end") is stored or transmitted first.
For example, the 32-bit integer 0x12345678 would be stored as:
```
Byte 0: 0x78
Byte 1: 0x56
Byte 2: 0x34
Byte 3: 0x12
```


The struct.unpack function will always return a tuple, even if it contains only a single element.
Code
```python
import struct
big_endian_bytes = b'\x12\x34\x56\x78'
value1 = struct.unpack('>I', big_endian_bytes)
print(f"value1: {value1}")
```
Output
```
value1: (305419896,)
```
Code
```python
import struct
big_endian_bytes = b'\x12\x34\x56\x78'
value1 = struct.unpack('>I', big_endian_bytes)[0]
print(f"Unpacked value1 (Big Endian): {value1}")
```
Output
```
Unpacked value1 (Big Endian): 305419896
```
Code
```python
import struct
big_endian_bytes = b'\x12\x34\x56\x78\x22\x24\x36\x78\x22\x24\x36\x18'
value1, value2, value3 = struct.unpack('>III', big_endian_bytes)
print(f"Unpacked value1 (Big Endian): {value1}")  
print(f"Unpacked value2 (Big Endian): {value2}")  
print(f"Unpacked value3 (Big Endian): {value3}") 
```
Output:
```
Unpacked value1 (Big Endian): 305419896
Unpacked value2 (Big Endian): 572798584
Unpacked value3 (Big Endian): 572798488
```
--------
```
TEST SET LABEL FILE (t10k-labels-idx1-ubyte):
[offset] [type]          [value]          [description]
0000     32 bit integer  0x00000801(2049) magic number (MSB first)
0004     32 bit integer  10000            number of items
0008     unsigned byte   ??               label
0009     unsigned byte   ??               label
........
xxxx     unsigned byte   ??               label
The labels values are 0 to 9.
```


lbl_f.read(num_items): Reads num_items bytes from the file, where num_items is the number of labels.

---------


```python
binary_data = b'\x00\xff\x7f\x80\x01
```
binary_data = b'\x00\xff\x7f\x80\x01' is an example of a buffer. In Python, a buffer is essentially a contiguous block of memory that can store binary data. The bytes object b'\x00\xff\x7f\x80\x01' is a sequence of bytes, where each byte is represented as a hexadecimal value.

The bytes object b'\x00\xff\x7f\x80\x01' is composed of 5 bytes, and each byte is indeed 8 bits.

b'': The prefix b indicates that the following string is a bytes object.
\x00\xff\x7f\x80\x01: Each of these sequences represents a byte in hexadecimal notation:
\x00: The hexadecimal value 00, which is 00000000 in binary (8 bits).
\xff: The hexadecimal value FF, which is 11111111 in binary (8 bits).
\x7f: The hexadecimal value 7F, which is 01111111 in binary (8 bits).
\x80: The hexadecimal value 80, which is 10000000 in binary (8 bits).
\x01: The hexadecimal value 01, which is 00000001 in binary (8 bits).

Byte: A byte is a unit of digital information that consists of 8 bits. It can represent 256 different values (from 0 to 255 for unsigned integers).
Bit: A bit is the smallest unit of data in computing and can have a value of either 0 or 1.

```python
import numpy as np

# Example binary data (5 pixels with values 0, 255, 127, 128, 1)
binary_data = b'\x00\xff\x7f\x80\x01'

# Convert the binary data to a numpy array of uint8
pixels = np.frombuffer(binary_data, dtype=np.uint8)

print(pixels)  # Output: [  0 255 127 128   1]
```
Unsigned Integers

An unsigned integer is a type of integer that can only represent non-negative values. For an 8-bit unsigned integer (uint8), the possible range of values is from 0 to 255.

Understanding uint8

u: Stands for "unsigned." In computing, unsigned numbers are those that can only represent non-negative values (i.e., 0 and positive integers). They do not have a sign bit to indicate positive or negative values.
int: Short for "integer," which refers to whole numbers that do not have fractional parts.
8: Indicates that the integer is represented using 8 bits (1 byte).


Using np.frombuffer
When we use np.frombuffer, we are taking an existing buffer (like a bytes object) and interpreting its binary data as a numpy array of a specified data type. This is useful because it allows us to work with the binary data directly in a structured format (like an array of integers or floats) without having to copy the data, which is efficient.

---------------------------------

struct.unpack vs np.frombuffer
struct.unpack:

Purpose: Unpack fixed-size, structured binary data into Python objects.
Usage: Suitable for reading structured headers or records with a predefined format.
Example: Reading the header of the MNIST file which has a specific structure with multiple fields (magic number, number of items).
np.frombuffer:

Purpose: Interpret a continuous block of memory as a numpy array of a specified type.
Usage: Suitable for reading homogeneous data where each element is the same size and type, such as arrays of pixel values or labels.
Example: Reading the sequence of label bytes in the MNIST file.

---------------------------------

```python
 # Read the images file
    with gzip.open(image_filename, 'rb') as img_f:
        magic, num_images, num_rows, num_cols = struct.unpack(">IIII", img_f.read(16))
        if magic != 2051:
            raise ValueError(f"Invalid magic number in image file: {magic}")
        images = np.frombuffer(img_f.read(num_images * num_rows * num_cols), dtype=np.uint8)
        images = images.reshape(num_images, num_rows * num_cols).astype(np.float32)
        images /= 255.0  # Normalize to range [0, 1]
```
For reshape(num_images, num_rows * num_cols):
Reshapes the 1D array into a 2D array with shape (num_images, num_rows * num_cols).
Each row represents one image, and each image is flattened into a 1D array of pixel values.


astype(np.float32): Converts the array's data type from uint8 to float32. This is necessary for normalization and further processing.

Why Convert to np.float32?

Precision:
np.uint8: This data type can store integer values from 0 to 255. It is not capable of representing fractional values, which are needed for normalization.
np.float32: This data type can store floating-point numbers, which can represent a much wider range of values, including fractions.

--------------------
**axis=0**: Sum along the rows (down the columns). This reduces the dimension by summing over the first axis.

**axis=1**: Sum along the columns (across the rows). This reduces the dimension by summing over the second axis.

```python
import numpy as np

# Example 2D array
arr = np.array([[1, 2, 3],
                [4, 5, 6],
                [7, 8, 9]])

# Sum along the rows (down the columns)
sum_axis_0 = np.sum(arr, axis=0)
print("Sum along axis=0 (down the columns):", sum_axis_0)
# Output: [12 15 18]

# Sum along the columns (across the rows)
sum_axis_1 = np.sum(arr, axis=1)
print("Sum along axis=1 (across the rows):", sum_axis_1)
# Output: [ 6 15 24]
```
-------------------
```python
import numpy as np

# Logits for a batch of 3 samples and 4 classes
Z = np.array([[2.0, 1.0, 0.1, 0.5],
              [1.5, 2.1, 0.2, 0.7],
              [1.1, 1.8, 0.3, 0.4]])

# True labels for the 3 samples
y = np.array([0, 1, 2])

# np.arange(Z.shape[0]) creates an array [0, 1, 2]
row_indices = np.arange(Z.shape[0])
print("Row indices:", row_indices)  # Output: [0 1 2]

# y is [0, 1, 2]
print("True class labels:", y)  # Output: [0 1 2]

# Advanced indexing: Z[np.arange(Z.shape[0]), y] selects Z[0, 0], Z[1, 1], Z[2, 2]
correct_class_logits = Z[row_indices, y]

print("Correct class logits:", correct_class_logits)
# Output: [2.0, 2.1, 0.3]
```

