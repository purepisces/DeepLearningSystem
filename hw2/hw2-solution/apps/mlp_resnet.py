import sys

from numpy.random import shuffle

sys.path.append("../python")
import needle as ndl
import needle.nn as nn
import numpy as np
import time
import os

np.random.seed(0)
# MY_DEVICE = ndl.backend_selection.cuda()




class ResidualBlock(nn.Module):
    def __init__(self, dim, hidden_dim, norm=nn.BatchNorm1d, drop_prob=0.1):
        super(ResidualBlock, self).__init__()
        
        self.linear1 = nn.Linear(dim, hidden_dim)
        self.norm1 = norm(hidden_dim)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(drop_prob)
        self.linear2 = nn.Linear(hidden_dim, dim)
        self.norm2 = norm(dim)

    def forward(self, x):
        identity = x
        out = self.linear1(x)
        out = self.norm1(out)
        out = self.activation(out)
        out = self.dropout(out)
        out = self.linear2(out)
        out = self.norm2(out)
        out += identity
        out = self.activation(out)
        return out

class MLPResNet(nn.Module):
    def __init__(self, dim, hidden_dim=100, num_blocks=3, num_classes=10, norm=nn.BatchNorm1d, drop_prob=0.1):
      super(MLPResNet, self).__init__()
      self.linear1 = nn.Linear(dim, hidden_dim)
      self.activation = nn.ReLU()

        # Residual Blocks
      self.res_blks = nn.Sequential(
          *[ResidualBlock(hidden_dim, hidden_dim // 2, norm, drop_prob) for _ in range(num_blocks)]
      )

        # Final Linear Layer
      self.linear2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
      out = self.linear1(x)
      out = self.activation(out)
      out = self.res_blks(out)
      out = self.linear2(out)
      return out


def epoch(dataloader, model, opt=None):
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    # summary result
    total_loss = 0.0
    correct = 0
    total_samples = 0
    criterion = nn.SoftmaxLoss()
    if opt:
        model.train()
    else:
        model.eval()

    for inputs, labels in dataloader:

        inputs = ndl.reshape(inputs,(inputs.shape[0],-1))

        predicted = model(inputs)

        loss = criterion(predicted, labels)
        total_loss += loss.numpy() * inputs.shape[0]
        total_samples += labels.shape[0]
        # correct += (predicted == labels).sum()

        predicted_np = predicted.detach().numpy()
        labels_np = labels.detach().numpy()
        predicted_np = np.argmax(predicted_np, axis=1)

        correct += (predicted_np == labels_np).sum()


        # Backpropagation
        if opt:
          opt.reset_grad()
          loss.backward()
          opt.step()

    avg_loss = total_loss / total_samples
    error_rate = 1.0 - (correct / total_samples)
    return np.array([error_rate, avg_loss])
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
    train_img_path = data_dir + "/train-images-idx3-ubyte.gz"
    train_idx_path = data_dir + "/train-labels-idx1-ubyte.gz"
    train_dataset = ndl.data.MNISTDataset(
        train_img_path, train_idx_path
    )
    train_dataloader = ndl.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle = True)
    test_img_path = data_dir + "/t10k-images-idx3-ubyte.gz"
    test_idx_path = data_dir + "/t10k-labels-idx1-ubyte.gz"
    test_dataset = ndl.data.MNISTDataset(test_img_path, test_idx_path)
    test_dataloader = ndl.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle = False)
    model = MLPResNet(784, hidden_dim)
    opt = optimizer(model.parameters(),lr = lr, weight_decay = weight_decay)
    for e in range(epochs):
      train_metrics = epoch(train_dataloader, model, opt)
      test_metrics = epoch(test_dataloader, model)
      
      print(f"Epoch {e + 1}/{epochs}:")
      print(f"Train Error Rate: {train_metrics[0]}, Train Loss: {train_metrics[1]}")
      print(f"Test Error Rate: {test_metrics[0]}, Test Loss: {test_metrics[1]}")
    return train_metrics[0], train_metrics[1], test_metrics[0], test_metrics[1]
    ### END YOUR SOLUTION


if __name__ == "__main__":
    train_mnist(data_dir="../data")
