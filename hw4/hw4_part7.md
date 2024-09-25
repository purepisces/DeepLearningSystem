## Part 7: Training a word-level language model [10 points]


Finally, you will use the `RNN` and `LSTM` components you have written to construct a language model that we will train on the Penn Treebank dataset.

First, in `python/needle/nn/nn_sequence.py` implement `Embedding`. Consider we have a dictionary with 1000 words. Then for a word which indexes into this dictionary, we can represent this word as a one-hot vector of size 1000, and then use a linear layer to project this to a vector of some embedding size.

In `apps/models.py`, you can now implement `LanguageModel`. Your language model should consist of

- An embedding layer (which maps word IDs to embeddings)

- A sequence model (either RNN or LSTM)

- A linear layer (which outputs probabilities of the next word)


In `apps/simple_ml.py` implement `epoch_general_ptb`, `train_ptb`, and `evaluate_ptb`.

**Code Implementation**
```python
class Embedding(Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype="float32"):
        super().__init__()
        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.randn(num_embeddings, embedding_dim, device=device, dtype=dtype))
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        one_hot = init.one_hot(self.weight.shape[0], x, device=x.device, dtype=x.dtype)
        seq_len, bs, num_embeddings = one_hot.shape
        one_hot = one_hot.reshape((seq_len*bs, num_embeddings))
        
        return ops.matmul(one_hot, self.weight).reshape((seq_len, bs, self.weight.shape[1]))
        ### END YOUR SOLUTION
```
```python
class LanguageModel(nn.Module):
    def __init__(self, embedding_size, output_size, hidden_size, num_layers=1,
                 seq_model='rnn', device=None, dtype="float32"):
        super(LanguageModel, self).__init__()
        ### BEGIN YOUR SOLUTION
        self.embedding = nn.Embedding(output_size, embedding_size, device=device, dtype=dtype)
        if seq_model == 'rnn':
            self.model = nn.RNN(embedding_size, hidden_size, num_layers, device=device, dtype=dtype)
        elif seq_model == 'lstm':
            self.model = nn.LSTM(embedding_size, hidden_size, num_layers, device=device, dtype=dtype)
        self.linear = nn.Linear(hidden_size, output_size, device=device, dtype=dtype)
        ### END YOUR SOLUTION

    def forward(self, x, h=None):
        ### BEGIN YOUR SOLUTION
        x = self.embedding(x) # (seq_len, bs, embedding_size)
        out, h = self.model(x, h)
        seq_len, bs, hidden_size = out.shape
        out = out.reshape((seq_len * bs, hidden_size))
        out = self.linear(out)
        return out, h
        ### END YOUR SOLUTION
```
```python
```python3
def epoch_general_ptb(data, model, seq_len=40, loss_fn=nn.SoftmaxLoss(), opt=None,
        clip=None, device=None, dtype="float32"):
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    if opt:
        model.train()
    else:
        model.eval()
    total_loss = 0
    total_error = 0
    n_batch, batch_size = data.shape
    iter_num = n_batch - seq_len
    for iter_idx in range(iter_num):
        X, target = ndl.data.get_batch(data, iter_idx, seq_len, device=device, dtype=dtype)
        if opt:
            opt.reset_grad()
        pred, _ = model(X)
        loss = loss_fn(pred, target)
        if opt:
            opt.reset_grad()
            loss.backward()
            if clip:
                opt.clip_grad_norm(clip)
            opt.step()
        total_loss += loss.numpy()
        total_error += np.sum(pred.numpy().argmax(1)!=target.numpy())
    avg_loss = total_loss / iter_num
    avg_acc = 1 - total_error / (iter_num * seq_len)
    return avg_acc, avg_loss
    ### END YOUR SOLUTION
```
```python
def train_ptb(model, data, seq_len=40, n_epochs=1, optimizer=ndl.optim.SGD,
          lr=4.0, weight_decay=0.0, loss_fn=nn.SoftmaxLoss, clip=None,
          device=None, dtype="float32"):
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    for epoch in range(n_epochs):
        avg_acc, avg_loss = epoch_general_ptb(data, model, seq_len, loss_fn(), optimizer(model.parameters(), lr=lr, weight_decay=weight_decay), clip=clip, device=device, dtype=dtype)
    return avg_acc, avg_loss
    ### END YOUR SOLUTION

def evaluate_ptb(model, data, seq_len=40, loss_fn=nn.SoftmaxLoss,
        device=None, dtype="float32"):
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    avg_acc, avg_loss = epoch_general_ptb(data, model, seq_len, loss_fn(), device=device, dtype=dtype)
    return avg_acc, avg_loss
    ### END YOUR SOLUTION
```

### Explanation of `class Embedding`

#### What is an Embedding Layer?

In natural language processing, words are represented as numbers (word IDs). An **embedding layer** takes these word IDs and maps them into dense vectors (embeddings). These embeddings are more meaningful representations of the words, capturing similarities between words.

> A **dense vector** is a vector where most (if not all) of the elements have meaningful, non-zero values. In contrast, a **sparse vector** has many elements that are zero.

#### Components of the `Embedding` Class
```python
class Embedding(Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype="float32"):
        super().__init__()
        self.weight = Parameter(init.randn(num_embeddings, embedding_dim, device=device, dtype=dtype))
```

#### Parameters:

1.  **num_embeddings**: The size of the vocabulary, or the number of unique words. For example, if we have 1000 unique words in our dataset, `num_embeddings = 1000`.
    
2.  **embedding_dim**: The size of the embedding vector. For instance, if we want each word to be represented by a 50-dimensional vector, then `embedding_dim = 50`.
    
3.  **self.weight**: This is a matrix of size `(num_embeddings, embedding_dim)`, initialized randomly using `init.randn`. This matrix will store the embeddings of all the words in the vocabulary.
    
    -   **num_embeddings**: Number of words in the dictionary (rows in the weight matrix).
    -   **embedding_dim**: Length of the vector for each word (columns in the weight matrix).


#### Example:

Suppose we have a vocabulary with 3 words:

-   "dog" → ID 0
-   "cat" → ID 1
-   "fish" → ID 2

And we choose an embedding size of 4. The `self.weight` matrix might look like this (with random values initially):

| Word ID | Embedding (4D Vector)         |
|---------|-------------------------------|
| 0 (dog) | [0.1, -0.2, 0.5, 1.0]         |
| 1 (cat) | [-0.3, 0.6, -0.7, 0.2]        |
| 2 (fish)| [0.8, -1.2, 0.3, -0.9]        |

Each word is mapped to a unique row in this weight matrix.

#### Forward Pass: Mapping Word IDs to Embeddings

Now let’s look at the forward pass where the embeddings are actually used:

```python
def forward(self, x: Tensor) -> Tensor:
    one_hot = init.one_hot(self.weight.shape[0], x, device=x.device, dtype=x.dtype)
    seq_len, bs, num_embeddings = one_hot.shape
    one_hot = one_hot.reshape((seq_len*bs, num_embeddings))
        
    return ops.matmul(one_hot, self.weight).reshape((seq_len, bs, self.weight.shape[1]))
```
#### Steps in the Forward Pass:

1.  **Input `x`:**
    
    -   `x` is a tensor containing word IDs. Suppose `x = [[0, 1, 2], [2, 0, 1]]` represents a batch of word sequences. Each sequence contains word IDs that correspond to "dog", "cat", "fish", and so on.
2.  **One-Hot Encoding:**
    
    -   The word IDs are converted into one-hot vectors using the `one_hot` function. A one-hot vector for a vocabulary of size 3 looks like this:
        -   "dog" (ID 0) → `[1, 0, 0]`
        -   "cat" (ID 1) → `[0, 1, 0]`
        -   "fish" (ID 2) → `[0, 0, 1]`
    -   The result is a tensor with shape `(seq_len, batch_size, num_embeddings)`.
3.  **Matrix Multiplication:**
    
    -   A matrix multiplication (`ops.matmul`) is performed between the one-hot encoded input and the `weight` matrix to get the corresponding embeddings for each word.
4.  **Reshape:**
    
    -   Finally, the output is reshaped to match the original sequence length and batch size. The output has a shape of `(seq_len, batch_size, embedding_dim)`.

#### Example:

Let’s say the input `x` is a batch of 2 sequences of word IDs:
```python
x = [[0, 1], [2, 0]]
```
This corresponds to:

-   Sequence 1: ["dog", "cat"]
-   Sequence 2: ["fish", "dog"]

The forward pass will map each word to its embedding, and the output might look like:
```python
[ [[0.1, -0.2, 0.5, 1.0], [-0.3, 0.6, -0.7, 0.2]],
  [[0.8, -1.2, 0.3, -0.9], [0.1, -0.2, 0.5, 1.0]] ]
```
Where each row corresponds to the embedding vector for the words in the sequence.

#### In Summary:

-   The **`Embedding`** layer converts word IDs into dense vectors (embeddings) using a matrix `self.weight`, where each row is the embedding for a word in the vocabulary.
-   During the **forward pass**, the word IDs are mapped to their respective embedding vectors using one-hot encoding and matrix multiplication.

### Explanation of `class LanguageModel`

This class implements a word-level language model using an embedding layer, a recurrent neural network (RNN or LSTM), and a final linear layer to predict the next word in a sequence.

The model consists of:

-   **Embedding Layer**: Converts word indices into dense vectors.
-   **RNN or LSTM Layer**: Processes the sequence of word embeddings, learning dependencies between them.
-   **Linear Layer**: Maps the RNN or LSTM's hidden state output to the vocabulary size, outputting probabilities for the next word in the sequence.

```python
class LanguageModel(nn.Module):
    def __init__(self, embedding_size, output_size, hidden_size, num_layers=1,
                 seq_model='rnn', device=None, dtype="float32"):
        super(LanguageModel, self).__init__()
```

#### Parameters in `__init__`:

1.  **`embedding_size`**: The size of the embedding vector for each word (e.g., if we represent each word with a 100-dimensional vector, `embedding_size = 100`).
2.  **`output_size`**: The number of unique words in the vocabulary (e.g., if there are 10,000 unique words, `output_size = 10000`).
3.  **`hidden_size`**: The size of the hidden state vector in the RNN or LSTM. This determines the model's capacity to learn patterns in the data.
4.  **`num_layers`**: The number of RNN or LSTM layers stacked on top of each other (default is 1).
5.  **`seq_model`**: A string that indicates whether to use an RNN or LSTM (default is `'rnn'`).
6.  **`device`**: Specifies whether to run the model on a CPU or GPU.
7.  **`dtype`**: The data type used for the model (default is `float32`).

#### Components of the Model

```python
self.embedding = nn.Embedding(output_size, embedding_size, device=device, dtype=dtype)
```
- The **embedding layer** converts word indices into dense vectors of size `embedding_size`. For example, if `embedding_size = 100`, each word in the vocabulary is represented by a 100-dimensional dense vector.

```python
if seq_model == 'rnn':
    self.model = nn.RNN(embedding_size, hidden_size, num_layers, device=device, dtype=dtype)
elif seq_model == 'lstm':
    self.model = nn.LSTM(embedding_size, hidden_size, num_layers, device=device, dtype=dtype)
```
- Depending on whether `'rnn'` or `'lstm'` is passed, an RNN or LSTM layer is initialized. Both models take in the embeddings as input and return hidden states of size `hidden_size`.

```python
self.linear = nn.Linear(hidden_size, output_size, device=device, dtype=dtype)
```

- The **linear layer** takes the hidden states from the RNN or LSTM and maps them to the size of the vocabulary (`output_size`). This layer's output will be used to predict the next word.

#### Forward Pass

```python
def forward(self, x, h=None):
    x = self.embedding(x)  # (seq_len, bs, embedding_size)
    out, h = self.model(x, h)
    seq_len, bs, hidden_size = out.shape
    out = out.reshape((seq_len * bs, hidden_size))
    out = self.linear(out)
    return out, h
```

#### Steps in the `forward` Method:

1.  **Embedding Layer**:
```python
x = self.embedding(x)
```
-   The input `x` is a batch of word indices (integers), with shape `(seq_len, batch_size)`. The embedding layer converts these indices into dense vectors, resulting in an output with shape `(seq_len, batch_size, embedding_size)`.
    
2.   **RNN/LSTM Layer**:
```python
out, h = self.model(x, h)
```
-   The embeddings are passed through the RNN or LSTM. The output `out` contains the hidden states for each time step, with shape `(seq_len, batch_size, hidden_size)`. The hidden states capture the relationships between words in the sequence.
    
3.   **Reshaping**:
```python
out = out.reshape((seq_len * bs, hidden_size))
```
-   The hidden states are reshaped to a 2D tensor with shape `(seq_len * batch_size, hidden_size)` in preparation for the final linear layer.
    
4.  **Linear Layer**:
```python
out = self.linear(out)
```
-   The reshaped hidden states are passed through the linear layer, which maps the `hidden_size` to `output_size` (the size of the vocabulary). The result is a tensor of shape `(seq_len * batch_size, output_size)` containing the logits (un-normalized probabilities) for each word in the vocabulary.
    
5.  **Return Output and Hidden States**:
```python
return out, h
```
The model returns both the output (logits for predicting the next word) and the hidden state `h` (to allow the model to remember information across batches).


