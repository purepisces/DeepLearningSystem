
## Question 5

  

Given you have now implemented all the necessary components for our neural network library, let's build and train an MLP ResNet. For this question, you will be working in `apps/mlp_resnet.py`. First, fill out the functions `ResidualBlock` and `MLPResNet` as described below:

  

### ResidualBlock

`ResidualBlock(dim, hidden_dim, norm=nn.BatchNorm1d, drop_prob=0.1)`

  

Implements a residual block as follows:

  
<img src="residualblock.png" alt="residualblock" width="400" height="500"/>


**NOTE**: if the figure does not render, please see the figure in the `figures` directory.

  

where the first linear layer has `in_features=dim` and `out_features=hidden_dim`, and the last linear layer has `out_features=dim`. Returns the block as type `nn.Module`.

  

##### Parameters

- `dim` (*int*) - input dim

- `hidden_dim` (*int*) - hidden dim

- `norm` (*nn.Module*) - normalization method

- `drop_prob` (*float*) - dropout probability

  
**Code Implementation:**
 
```python
  def ResidualBlock(dim, hidden_dim, norm=nn.BatchNorm1d, drop_prob=0.1):
    ### BEGIN YOUR SOLUTION
    return nn.Sequential(
        nn.Residual(
            nn.Sequential(
                nn.Linear(dim, hidden_dim),
                norm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(drop_prob),
                nn.Linear(hidden_dim, dim),
                norm(dim)
            )
        ),
        nn.ReLU()
    )
    ### END YOUR SOLUTION
```
___

### Explaination of ResidualBlock
 
 The `ResidualBlock` function  is designed to implement a residual block commonly used in deep learning models, specifically in ResNet architectures. A residual block helps in training deeper networks by allowing gradients to flow through the network more effectively, mitigating the vanishing gradient problem.

#### 1. **Function Definition:**

-   **`dim` (int):** The dimensionality of the input.
-   **`hidden_dim` (int):** The dimensionality of the hidden layer inside the residual block.
-   **`norm` (nn.Module):** A normalization method (like `nn.BatchNorm1d`), applied after each linear layer.
-   **`drop_prob` (float):** The dropout probability, used to prevent overfitting by randomly setting a fraction of the input units to 0 during training.

#### 2. **Return Value:**

-   The function returns an instance of `nn.Sequential`, which is a sequential container of layers/modules that will be executed in the order they are defined.

#### 3. **Inner Structure:**

-   **`nn.Sequential`:** The outer `nn.Sequential` container applies layers sequentially.
    
-   **`nn.Residual`:**
    
    -   This wrapper encapsulates the sequence of layers defined within it. The `nn.Residual` module adds the input (skip connection) to the output of the layers inside the sequence, forming a residual connection. This is key to allowing the gradients to bypass the non-linearities, aiding in the training of deeper networks.
-   **Inner `nn.Sequential`:**
    
    -   **`nn.Linear(dim, hidden_dim)`:** The first linear layer projects the input of dimension `dim` to a higher dimension `hidden_dim`.
    -   **`norm(hidden_dim)`:** Applies normalization (like BatchNorm) to the output of the first linear layer to stabilize and speed up training.
    -   **`nn.ReLU()`:** Applies the ReLU (Rectified Linear Unit) activation function, introducing non-linearity.
    -   **`nn.Dropout(drop_prob)`:** Applies dropout to the output, with the probability `drop_prob` of dropping units, to prevent overfitting.
    -   **`nn.Linear(hidden_dim, dim)`:** The second linear layer projects the `hidden_dim` back to the original input dimension `dim`.
    -   **`norm(dim)`:** Normalization is applied again to stabilize the final output before adding the residual connection.
-   **`nn.ReLU()`:** Finally, after the residual connection is applied, another ReLU activation is applied to the combined output, which is a common practice in residual networks.

#### **Explanation of  Layers and Their Input:**

1.  **First Linear Layer (`nn.Linear(dim, hidden_dim)`):**
    
    -   **Input:** The input to this layer has the shape `(batch_size, dim)`.
    -   **Function:** This layer transforms the input from the original dimensionality `dim` to a higher dimensionality `hidden_dim`.
    -   **Output:** The output shape is `(batch_size, hidden_dim)`.
2.  **Normalization Layer (`norm(hidden_dim)`):**
    
    -   **Input:** The input is the output from the first linear layer, which has a shape of `(batch_size, hidden_dim)`.
    -   **Function:** This normalization layer (e.g., BatchNorm1d) normalizes the output to stabilize the training process and make the model converge faster.
    -   **Output:** The output shape remains `(batch_size, hidden_dim)`.
3.  **ReLU Activation (`nn.ReLU()`):**
    
    -   **Input:** The input is the normalized output from the previous layer, with a shape of `(batch_size, hidden_dim)`.
    -   **Function:** ReLU introduces non-linearity into the model by applying the ReLU activation function.
    -   **Output:** The output shape remains `(batch_size, hidden_dim)`.
4.  **Dropout Layer (`nn.Dropout(drop_prob)`):**
    
    -   **Input:** The input is the activated output from the ReLU layer, with a shape of `(batch_size, hidden_dim)`.
    -   **Function:** The Dropout layer randomly sets a fraction of the inputs to zero with probability `drop_prob` to prevent overfitting.
    -   **Output:** The output shape remains `(batch_size, hidden_dim)`.
5.  **Second Linear Layer (`nn.Linear(hidden_dim, dim)`):**
    
    -   **Input:** The input is the output from the Dropout layer, which has a shape of `(batch_size, hidden_dim)`.
    -   **Function:** This layer projects the data back from the higher dimensionality `hidden_dim` to the original input dimensionality `dim`.
    -   **Output:** The output shape is `(batch_size, dim)`.
6.  **Second Normalization Layer (`norm(dim)`):**
    
    -   **Input:** The input is the output from the second linear layer, which has a shape of `(batch_size, dim)`.
    -   **Function:** This normalization layer normalizes the output again to stabilize the output of the residual block.
    -   **Output:** The output shape remains `(batch_size, dim)`.

#### **Residual Connection (Implemented in `nn.Residual`):**

-   **Input:** The original input to the block, with a shape of `(batch_size, dim)`, is passed unchanged to the `nn.Residual` wrapper.
-   **Function:** Inside the `nn.Residual`, the output from the inner `nn.Sequential` block (which has gone through all the transformations) is added element-wise to the original input.
-   **Output:** The final output after the residual addition has the same shape as the original input, `(batch_size, dim)`.

#### **Final ReLU Activation:**

-   **Input:** The input to this layer is the result of the residual connection, which has a shape of `(batch_size, dim)`.
-   **Function:** This ReLU activation is applied to the output of the residual connection.
-   **Output:** The final output shape remains `(batch_size, dim)`.

#### **Summary:**

-   The input starts with a shape of `(batch_size, dim)`.
-   It is transformed into a higher-dimensional space `(batch_size, hidden_dim)` through the first linear layer and goes through normalization, ReLU activation, and dropout.
-   It is then projected back to `(batch_size, dim)` through the second linear layer and normalization.
-   Finally, the original input (with shape `(batch_size, dim)`) is added to this transformed output, creating a residual connection, which is followed by a ReLU activation.

The entire block helps maintain the identity of the input through the network while still allowing transformations to occur, enabling the training of very deep networks more effectively.

___

  

### MLPResNet

`MLPResNet(dim, hidden_dim=100, num_blocks=3, num_classes=10, norm=nn.BatchNorm1d, drop_prob=0.1)`

  

Implements an MLP ResNet as follows:

  

<img src="mlp-resnet.png" alt="mlp-resnet" width="400" height="500"/>


  

where the first linear layer has `in_features=dim` and `out_features=hidden_dim`, and each ResidualBlock has `dim=hidden_dim` and `hidden_dim=hidden_dim//2`. Returns a network of type `nn.Module`.

  

##### Parameters

- `dim` (*int*) - input dim

- `hidden_dim` (*int*) - hidden dim

- `num_blocks` (*int*) - number of ResidualBlocks

- `num_classes` (*int*) - number of classes

- `norm` (*nn.Module*) - normalization method

- `drop_prob` (*float*) - dropout probability (0.1)

___

  

Once you have the deep learning model architecture correct, let's train the network using our new neural network library components. Specifically, implement the functions `epoch` and `train_mnist`.

  

### Epoch

  

`epoch(dataloader, model, opt=None)`

  

Executes one epoch of training or evaluation, iterating over the entire training dataset once (just like `nn_epoch` from previous homeworks). Returns the average error rate (as a *float*) and the average loss over all samples (as a *float*). Set the model to `training` mode at the beginning of the function if `opt` is given; set the model to `eval` if `opt` is not given (i.e. `None`). When setting the modes, use `.train()` and `.eval()` instead of modifying the training attribute.

  

##### Parameters

- `dataloader` (*`needle.data.DataLoader`*) - dataloader returning samples from the training dataset

- `model` (*`needle.nn.Module`*) - neural network

- `opt` (*`needle.optim.Optimizer`*) - optimizer instance, or `None`

  

___

  

### Train Mnist

  

`train_mnist(batch_size=100, epochs=10, optimizer=ndl.optim.Adam, lr=0.001, weight_decay=0.001, hidden_dim=100, data_dir="data")`

Initializes a training dataloader (with `shuffle` set to `True`) and a test dataloader for MNIST data, and trains an `MLPResNet` using the given optimizer (if `opt` is not None) and the softmax loss for a given number of epochs. Returns a tuple of the training accuracy, training loss, test accuracy, test loss computed in the last epoch of training. If any parameters are not specified, use the default parameters.

  

##### Parameters

- `batch_size` (*int*) - batch size to use for train and test dataloader

- `epochs` (*int*) - number of epochs to train for

- `optimizer` (*`needle.optim.Optimizer` type*) - optimizer type to use

- `lr` (*float*) - learning rate

- `weight_decay` (*float*) - weight decay

- `hidden_dim` (*int*) - hidden dim for `MLPResNet`

- `data_dir` (*int*) - directory containing MNIST image/label files
