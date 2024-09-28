In the context of convolutional neural networks (CNNs), when we say the input tensor $X$ is **convolved** with the weights $W$ (the convolution kernel or filter), it refers to the **convolution operation**, which is a mathematical operation that combines the input and the kernel to produce an output (often called the **feature map**).

### What Does "Convolve" Mean?
**Convolution** is an operation where a small filter (or kernel) is systematically moved across an input (like an image or feature map), and at each position, the filter and a corresponding patch of the input are multiplied element-wise and summed up to produce a single value. This value becomes part of the output feature map.


### Example of Convolution:

Consider an example where:

- The input $X$ is a $5 \times 5$ matrix.
- The kernel $W$ is a $3 \times 3$ matrix.

#### Input $X$:

$$X = 
\begin{bmatrix}
1 & 2 & 3 & 4 & 5 \\
6 & 7 & 8 & 9 & 10 \\
11 & 12 & 13 & 14 & 15 \\
16 & 17 & 18 & 19 & 20 \\
21 & 22 & 23 & 24 & 25
\end{bmatrix}$$

#### Kernel $W$:

$$W = 
\begin{bmatrix}
1 & 0 & -1 \\
1 & 0 & -1 \\
1 & 0 & -1
\end{bmatrix}$$


The convolution involves sliding the $3 \times 3$ kernel over the input. Let's compute the convolution for the top-left position in the output.

#### Step 1: Extract the top-left $3 \times 3$ patch from the input:

$$Patch = 
\begin{bmatrix}
1 & 2 & 3 \\
6 & 7 & 8 \\
11 & 12 & 13
\end{bmatrix}$$

#### Step 2: Perform element-wise multiplication with the kernel:

$$\text{Element-wise product} = 
\begin{bmatrix}
1 \times 1 & 2 \times 0 & 3 \times (-1) \\
6 \times 1 & 7 \times 0 & 8 \times (-1) \\
11 \times 1 & 12 \times 0 & 13 \times (-1)
\end{bmatrix}=
\begin{bmatrix}
1 & 0 & -3 \\
6 & 0 & -8 \\
11 & 0 & -13
\end{bmatrix}
$$

#### Step 3: Sum the element-wise products:

$$1 + 0 - 3 + 6 + 0 - 8 + 11 + 0 - 13 = -6$$

The value at the top-left corner of the output feature map is $-6$.

This process is repeated as the kernel slides across the entire input to compute the full output feature map.

### Feature Map (Output):

The result of the convolution operation is called the **feature map**. Each element of the feature map represents the result of the convolution at a particular position in the input, capturing local spatial patterns like edges, textures, or other features in the input.


#### **Why Dilation is Used in the Gradient (X_grad) Calculation:**

In backpropagation, the gradient of the loss with respect to the input (i.e., $\frac{\partial \text{loss}}{\partial X}$​) is essentially performing a convolution in reverse. The convolution operation in the forward pass is performed with a certain stride. To reverse this during backpropagation, you need to "dilate" the gradient that comes from the output (`out_grad`), filling the gaps that were skipped during the forward pass due to the stride.

-   **In the forward pass**, when using a stride greater than 1, the kernel skips certain elements of the input.
-   **In the backward pass**, to compute the gradient with respect to the input, we need to expand the positions where the kernel was applied in the forward pass. This is done using **dilation** to match the positions of the skipped elements from the forward pass.

Thus, dilation is used in backpropagation to "fill in" the gaps that were skipped by the stride in the forward pass.

### Explain why we use W.flip in calculating X_grad
I will use a small example which about why gradient with respect to Input $X$ (Why We Use $W^T$) to illustrate it
#### Setup for a Simple Dense Layer

Assume you have an input matrix $X$ with two features and one sample:

$$X = \begin{bmatrix} x_1 & x_2 \end{bmatrix}$$

You also have a weight matrix \( W \) for two neurons:

$$W = \begin{bmatrix} w_{11} & w_{12} \\ w_{21} & w_{22} \end{bmatrix}$$

The matrix multiplication in the forward pass is:

$$Z = X \times W = \begin{bmatrix} x_1 & x_2 \end{bmatrix} \times \begin{bmatrix} w_{11} & w_{12} \\ w_{21} & w_{22} \end{bmatrix}$$

This results in:

$$Z = \begin{bmatrix} x_1 \cdot w_{11} + x_2 \cdot w_{21} & x_1 \cdot w_{12} + x_2 \cdot w_{22} \end{bmatrix}$$

### Backward Pass (Using the Gradient of \( Z \), i.e., \( dZ \))

Now, during backpropagation, suppose we have the gradient of the loss with respect to \( Z \), denoted as \( dZ \):

$$dZ = \begin{bmatrix} dZ_1 & dZ_2 \end{bmatrix}$$

This means:

- $dZ_1$is the gradient of the loss with respect to the first output neuron (influenced by $w_{11}$ and $w_{21}$),
- $dZ_2$ is the gradient with respect to the second output neuron (influenced by $w_{12}$ and $w_{22}$).

#### Gradient with Respect to Input $X$ (Why We Use $W^T$)

To compute how the input $X = \begin{bmatrix} x_1 & x_2 \end{bmatrix}$ contributed to the output gradients $dZ$, we need to propagate the gradients back through the weight matrix $W$.

We do this by multiplying $dZ$ by the transpose of $W$, i.e., $W^T$, because the transpose reverses the flow of the forward pass.

The transpose of $W$ is:

$$W^T = \begin{bmatrix} w_{11} & w_{21} \\ w_{12} & w_{22} \end{bmatrix}$$

Now, we multiply $dZ$ by $W^T$ to compute the gradient with respect to the input, $dX$:

$$dX = dZ \times W^T = \begin{bmatrix} dZ_1 & dZ_2 \end{bmatrix} \times \begin{bmatrix} w_{11} & w_{21} \\ w_{12} & w_{22} \end{bmatrix}$$

The result of this multiplication is:

$$dX = \begin{bmatrix} dZ_1 \cdot w_{11} + dZ_2 \cdot w_{12} & dZ_1 \cdot w_{21} + dZ_2 \cdot w_{22} \end{bmatrix}$$

So, the gradients with respect to $x_1$ and $x_2$ are:

$$dX_1 = dZ_1 \cdot w_{11} + dZ_2 \cdot w_{12}$$
$$dX_2 = dZ_1 \cdot w_{21} + dZ_2 \cdot w_{22}$$
