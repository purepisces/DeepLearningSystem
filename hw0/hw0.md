The assignment will require you to build a basic softmax regression algorithm, plus a simple two-layer neural network.  You will create these implementations both in native Python (using the numpy library), and (for softmax regression) in native C/C++. 

Dataset:

The MNIST (Modified National Institute of Standards and Technology) dataset is a well-known dataset in the field of machine learning and computer vision. It consists of images of handwritten digits (0-9) and their corresponding labels. 

Structure of the MNIST Dataset
The MNIST dataset is split into two parts:

Training set: 60,000 examples
Test set: 10,000 examples
Each example in the dataset consists of:

Image: A 28x28 grayscale image of a handwritten digit.
Label: A label corresponding to the digit in the image (0-9).

Softmax(a.k.a. cross-entropy) loss:

Implement the softmax (a.k.a. cross-entropy) loss as defined in `softmax_loss()` function in `src/simple_ml.py`.  Recall (hopefully this is review, but we'll also cover it in lecture on 9/1), that for a multi-class output that can take on values $y \in \{1,\ldots,k\}$, the softmax loss takes as input a vector of logits $z \in \mathbb{R}^k$, the true class $y \in \{1,\ldots,k\}$ returns a loss defined by

$$\begin{equation}
\ell_{\mathrm{softmax}}(z, y) = \log\sum_{i=1}^k \exp z_i - z_y.
\end{equation}$$

```python
def softmax_loss(Z, y):
    """ Return softmax loss.  Note that for the purposes of this assignment,
    you don't need to worry about "nicely" scaling the numerical properties
    of the log-sum-exp computation, but can just compute this directly.

    Args:
        Z (np.ndarray[np.float32]): 2D numpy array of shape
            (batch_size, num_classes), containing the logit predictions for
            each class.
        y (np.ndarray[np.uint8]): 1D numpy array of shape (batch_size, )
            containing the true label of each example.

    Returns:
        Average softmax loss over the sample.
    """
    ### BEGIN YOUR CODE
    # Formula: \begin{equation} \ell_{\mathrm{softmax}}(z, y) = \log\sum_{i=1}^k \exp z_i - z_y. \end{equation}
    
    # Compute the log of the sum of exponentials of logits for each sample
    log_sum_exp = np.log(np.sum(np.exp(Z), axis = 1))
    # Extract the logits corresponding to the true class for each sample
    # np.arange(Z.shape[0]) generates array [0, 1, 2, ..., batch_size-1]
    # Z[np.arange(Z.shape[0]), y] = Z[[row_indices], [col_indices]]
    # This selects the logits Z[i, y[i]] for each i which is each row
    correct_class_logits = Z[np.arange(Z.shape[0]), y]
    losses = log_sum_exp - correct_class_logits
    return np.mean(losses)
    ### END YOUR CODE
```
