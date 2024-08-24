
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
