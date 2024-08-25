import sys

sys.path.append("../python")
import needle as ndl
import needle.nn as nn
import numpy as np
import time
import os

np.random.seed(0)
# MY_DEVICE = ndl.backend_selection.cuda()


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


def MLPResNet(
    dim,
    hidden_dim=100,
    num_blocks=3,
    num_classes=10,
    norm=nn.BatchNorm1d,
    drop_prob=0.1,
):
    ### BEGIN YOUR SOLUTION
    layers = [
        nn.Linear(dim, hidden_dim),
        nn.ReLU()
    ]
    
    for _ in range(num_blocks):
        layers.append(ResidualBlock(hidden_dim, hidden_dim // 2, norm, drop_prob))
    
    layers.append(nn.Linear(hidden_dim, num_classes))
    
    return nn.Sequential(*layers)
    ### END YOUR SOLUTION


def epoch(dataloader, model, opt=None):
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    # Set the model to training or evaluation mode once, based on whether an optimizer is provided
    if opt is None:
        model.eval()
    else:
        model.train()
        
    # Initialize the loss function and metrics
    loss_func = nn.SoftmaxLoss()
    total_loss = 0
    total_error = 0
    num_samples = len(dataloader.dataset)
    
    # Iterate over batches
    for batch_x, batch_y in dataloader:
        # Forward pass
        y_pred = model(batch_x)
        # Compute batch loss (average loss across the batch), which is a 0 dimensional tensor(a scalar tensor) with shape().
        # Type of batch_loss: <class 'needle.autograd.Tensor'>
        # Shape of batch_loss : ()
        batch_loss = loss_func(y_pred, batch_y)
        # Accumulate total loss (convert batch_loss to a NumPy scalar and multiply by batch size)
        # batch_loss.numpy() is a NumPy scalar (e.g., numpy.float64) with no dimensions (shape ()).
        total_loss += batch_loss.numpy() * batch_x.shape[0]
        
        # Backward pass and optimization (if in training mode)
        if opt is not None:
            # Clear the old gradients before computing the new ones for the current batch
            opt.reset_grad()
            # Perform backpropagation to compute the gradients of the loss with respect to each model parameter
            batch_loss.backward()
            # Update the model's parameters using the computed gradients
            opt.step()
            
        # Convert predictions and labels to numpy arrays
        batch_y = batch_y.numpy()
        y_pred = y_pred.numpy()
        # Calculate the number of incorrect predictions
        y_pred = np.argmax(y_pred, axis=1)
        total_error += np.sum(y_pred != batch_y)
        
    # Calculate error rate and average loss, both returned as NumPy scalars of type <class 'numpy.float64'>
    error_rate = total_error / num_samples
    average_loss = total_loss / num_samples
    return error_rate, average_loss
    ### END YOUR SOLUTION


def train_mnist(
    batch_size=100,
    epochs=10,
    optimizer=ndl.optim.Adam,
    lr=0.001,
    weight_decay=0.001,
    hidden_dim=100,
    data_dir="data",
):
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    # Load the data
    train_dataset = ndl.data.MNISTDataset(data_dir + "/train-images-idx3-ubyte.gz", data_dir+ "/train-labels-idx1-ubyte.gz")
    test_dataset = ndl.data.MNISTDataset(data_dir + "/t10k-images-idx3-ubyte.gz", data_dir + "/t10k-labels-idx1-ubyte.gz")
    
    train_dataloader = ndl.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = ndl.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # train_dataset[0] invokes the dataset's __getitem__ method. 
    # Since the index (0 in this case) is an integer, it retrieves a single image  and its corresponding label,     returning them as a tuple (image, label). 
    # The image is a flattened 28x28 array, giving it a shape of (784,). 
    # Thus, train_dataset[0] accesses the tuple (image, label), and train_dataset[0][0] specifically accesses       the image. 
    # The shape of this image is (784,), and train_dataset[0][0].shape[0] returns 784, which is the                 dimensionality of the flattened image.
    input_dim = train_dataset[0][0].shape[0]
    # Initialize the model
    model = MLPResNet(dim=input_dim, hidden_dim=hidden_dim)
    
    # Initialize the optimizer
    opt = optimizer(params=model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Training loop
    for epoch_num in range(epochs):
        train_error, train_loss = epoch(train_dataloader, model, opt=opt)
        test_error, test_loss = epoch(test_dataloader, model, opt=None)
    return train_error, train_loss, test_error, test_loss
    ### END YOUR SOLUTION


if __name__ == "__main__":
    train_mnist(data_dir="../data")
