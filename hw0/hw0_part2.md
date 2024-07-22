## Question 5: SGD for a two-layer neural network

Now that you've written SGD for a linear classifier, let's consider the case of a simple two-layer neural network. Specifically, for input $x \in \mathbb{R}^n$, we'll consider a two-layer neural network (without bias terms) of the form

$$\begin{equation}
z = W_2^T \mathrm{ReLU}(W_1^T x)
\end{equation}$$

where $W_1 \in \mathbb{R}^{n \times d}$ and $W_2 \in \mathbb{R}^{d \times k}$ represent the weights of the network (which has a $d$-dimensional hidden unit), and where $z \in \mathbb{R}^k$ represents the logits output by the network. We again use the softmax / cross-entropy loss, meaning that we want to solve the optimization problem

$$\begin{equation}
minimize_{W_1, W_2} \;\; \frac{1}{m} \sum_{i=1}^m \ell_{\mathrm{softmax}}(W_2^T \mathrm{ReLU}(W_1^T x^{(i)}), y^{(i)}).
\end{equation}$$

Or alternatively, overloading the notation to describe the batch form with matrix $X \in \mathbb{R}^{m \times n}$, this can also be written

$$\begin{equation}
minimize_{W_1, W_2} \;\; \ell_{\mathrm{softmax}}(\mathrm{ReLU}(X W_1) W_2, y).
\end{equation}$$

Using the chain rule, we can derive the backpropagation updates for this network (we'll briefly cover these in class, on 9/8, but also provide the final form here for ease of implementation). Specifically, let

$$\begin{equation}
\begin{split}
Z_1 \in \mathbb{R}^{m \times d} & = \mathrm{ReLU}(X W_1) \\
G_2 \in \mathbb{R}^{m \times k} & = normalize(\exp(Z_1 W_2)) - I_y \\
G_1 \in \mathbb{R}^{m \times d} & = \mathrm{1}\{Z_1 > 0\} \circ (G_2 W_2^T)
\end{split}
\end{equation}$$

where $\mathrm{1}\{Z_1 > 0\}$ is a binary matrix with entries equal to zero or one depending on whether each term in $Z_1$ is strictly positive and where $\circ$ denotes elementwise multiplication. Then the gradients of the objective are given by

$$\begin{equation}
\begin{split}
\nabla_{W_1} \ell_{\mathrm{softmax}}(\mathrm{ReLU}(X W_1) W_2, y) & = \frac{1}{m} X^T G_1 \\
\nabla_{W_2} \ell_{\mathrm{softmax}}(\mathrm{ReLU}(X W_1) W_2, y) & = \frac{1}{m} Z_1^T G_2. \\
\end{split}
\end{equation}$$

  

**Note:** If the details of these precise equations seem a bit cryptic to you (prior to the 9/8 lecture), don't worry too much. These _are_ just the standard backpropagation equations for a two-layer ReLU network: the $Z_1$ term just computes the "forward" pass while the $G_2$ and $G_1$ terms denote the backward pass. But the precise form of the updates can vary depending upon the notation you've used for neural networks, the precise ways you formulate the losses, if you've derived these previously in matrix form, etc. If the notation seems like it might be familiar from when you've seen deep networks in the past, and makes more sense after the 9/8 lecture, that is more than sufficient in terms of background (after all, the whole _point_ of deep learning systems, to some extent, is that we don't need to bother with these manual calculations). But if these entire concepts are _completely_ foreign to you, then it may be better to take a separate course on ML and neural networks prior to this course, or at least be aware that there will be substantial catch-up work to do for the course.

Using these gradients, now write the `nn_epoch()` function in the `src/simple_ml.py` file. As with the previous question, your solution should modify the `W1` and `W2` arrays in place. After implementing the function, run the following test. Be sure to use matrix operations as indicated by the expresssions above to implement the function: this will be _much_ faster, and more efficient, than attempting to use loops (and it requires far less code).

------------------------------------
### Key Terms

- **Logits**: These are the **raw, unnormalized scores output by a neural network's final layer before applying an activation function like softmax**. Logits are used to compute probabilities.
- **Probabilities**: These are the normalized scores obtained by applying the softmax function to the logits. They represent the predicted probabilities for each class.

### Layers in the Two-Layer (layers with trainable parameters (i.e., weights)) Neural Network

Network Structure: **Input Layer**, **First Layer: Hidden Layer (Input to Hidden)**, **Second Layer: Output Layer (Hidden to Output)**.

#### Input to Hidden Layer

- **Weight Matrix**: 
  $W_1 \in \mathbb{R}^{n \times d}$ where $n$ is the number of input features and $d$ is the number of hidden units.
- **Hidden Layer Activations**: 
  $Z_1 = \mathrm{ReLU}(X W_1)$
  - $X \in \mathbb{R}^{m \times n}$ is the input matrix.
  - $Z_1 \in \mathbb{R}^{m \times d}$ is the output of the hidden layer after applying the ReLU activation function. These are the activations, not logits or probabilities.

#### Hidden to Output Layer

- **Weight Matrix**: 
  $W_2 \in \mathbb{R}^{d \times k}$ where $d$ is the number of hidden units and $k$ is the number of output classes.
- **Output Logits**: 
  $Z_2 = Z_1 W_2$
  - $Z_2 \in \mathbb{R}^{m \times k}$ are the logits for the output layer.

#### Softmax Probabilities

- **Probabilities**: 
$P = \frac{\exp(Z_2)}{\sum \exp(Z_2)}$
  - $P \in \mathbb{R}^{m \times k}$ are the probabilities after applying the softmax function to the logits.

### Summary

- $Z_1 = \mathrm{ReLU}(X W_1)$ are the activations of the hidden layer after applying the ReLU function.
- $Z_2 = Z_1 W_2$ are the logits of the output layer.
- $P = \frac{\exp(Z_2)}{\sum \exp(Z_2)}$ are the probabilities after applying the softmax function to the logits $Z_2$.

## Backpropagation Overview
Backpropagation is the algorithm used to calculate the gradient of the loss function with respect to each parameter (weight) in a neural network. It allows the network to update these weights in a way that minimizes the loss function, enabling the network to learn from the data.

#### Forward Pass:
- Input data passes through the network, and the output is calculated.
- Activations and intermediate values (such as $Z_i$) are stored for use in the backward pass.

#### Backward Pass:
- The loss is computed using the output of the network and the true labels.
- Gradients of the loss with respect to each parameter are calculated using the chain rule of calculus.
- These gradients are used to update the parameters (weights) using an optimization algorithm like gradient descent.

### Backpropagation with Respect to $Z_i$ and $W_i$

In backpropagation, we need to calculate gradients with respect to both the activations $Z_i$ and the weights $W_i$. Here’s how this is done:

#### Gradients with Respect to Activations $Z_i$
The gradients with respect to the activations $Z_i$ are intermediate steps in the backpropagation process. They are used to calculate the gradients with respect to the weights. Specifically, for layer $i$, we compute $\frac{\partial \ell}{\partial Z_i}$, which represents how the loss changes with respect to the activations of that layer.

#### Gradients with Respect to Weights $W_i$
The ultimate goal is to compute the gradients of the loss with respect to the weights $W_i$, denoted as $\frac{\partial \ell}{\partial W_i}$. These gradients tell us how to adjust the weights to minimize the loss.



