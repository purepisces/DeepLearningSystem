## Part 4: Recurrent neural network [10 points]

**Note:** In the following sections, you may find yourself wanting to index into tensors, i.e., to use getitem or setitem. However, we have not implemented these for tensors in our library; instead, you should use `stack` and `split` operations.

In `python/needle/nn_sequence.py`, implement `RNNCell`.

$h^\prime = \text{tanh}(xW_{ih} + b_{ih} + hW_{hh} + b_{hh})$. If nonlinearity is 'relu', then ReLU is used in place of tanh.

All weights and biases should be initialized from $\mathcal{U}(-\sqrt{k}, \sqrt{k})$ where $k=\frac{1}{\text{hiddensize}}$.

In `python/needle/nn_sequence.py`, implement `RNN`.

For each element in the input sequence, each layer computes the following function:

$h_t = \text{tanh}(x_tW_{ih} + b_{ih} + h_{(t-1)}W_{hh} + b_{hh})$

where $h_t$ is the hidden state at time $t$, $x_t$ is the input at time $t$, and $h_{(t-1)}$ is the hidden state of the previous layer at time $t-1$ or the initial hidden state at time $0$. If nonlinearity is 'relu', then ReLU is used in place of tanh.

In a multi-layer RNN, the input $x_t^{(l)}$ of the $l$-th layer ($l \ge 2$) is the hidden state $h_t^{(l-1)}$ of the previous layer.
