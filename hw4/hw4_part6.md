## Part 6: Penn Treebank dataset [10 points]

In word-level language modeling tasks, the model predicts the probability of the next word in the sequence, based on the words already observed in the sequence. You will write support for the Penn Treebank dataset, which consists of stories from the Wall Street Journal, to train and evaluate a language model on word-level prediction.

In `python/needle/data/datasets/ptb_dataset.py`, start by implementing the `Dictionary` class, which creates a dictionary from a list of words, mapping each word to a unique integer.

Next, we will use this `Dictionary` class to create a corpus from the train and test txt files in the Penn Treebank dataset that you downloaded at the beginning of the notebook. Implement the `tokenize` function in the `Corpus` class to do this.

In order to prepare the data for training and evaluation, you will next implement the `batchify` function. Starting from sequential data, batchify arranges the dataset into columns. For instance, with the alphabet as the sequence and batch size 4, we'd get

```
┌ a g m s ┐

│ b h n t │

│ c i o u │

│ d j p v │

│ e k q w │

└ f l r x ┘

```
  
These columns are treated as independent by the model, which means that the dependence of e. g. 'g' on 'f' cannot be learned, but allows more efficient batch processing.

Next, implement the `get_batch` function. `get_batch` subdivides the source data into chunks of length `bptt`. If source is equal to the example output of the batchify function, with a bptt-limit of 2, we'd get the following two Variables for i = 0:

```
┌ a g m s ┐ ┌ b h n t ┐

└ b h n t ┘ └ c i o u ┘
```

Note that despite the name of the function, the subdivison of data is not done along the batch dimension (i.e. dimension 1), since that was handled by the batchify function. The chunks are along dimension 0, corresponding to the seq_len dimension in the LSTM or RNN.

**Code Implementation**
```python
class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        ### BEGIN YOUR SOLUTION
        if self.word2idx.get(word) is None:
            self.word2idx[word] = len(self.idx2word)
            self.idx2word.append(word)
        return self.word2idx[word]
        ### END YOUR SOLUTION

    def __len__(self):
        ### BEGIN YOUR SOLUTION
        return len(self.idx2word)
        ### END YOUR SOLUTION
```

```python
class Corpus(object):
    def __init__(self, base_dir, max_lines=None):
        self.dictionary = Dictionary()
        self.train = self.tokenize(os.path.join(base_dir, 'train.txt'), max_lines)
        self.test = self.tokenize(os.path.join(base_dir, 'test.txt'), max_lines)

    def tokenize(self, path, max_lines=None):
        ### BEGIN YOUR SOLUTION
        with open(path, 'r') as f:
            ids = []
            line_idx = 0
            for line in f:
                if max_lines is not None and line_idx >= max_lines:
                    break
                words = line.split() + ['<eos>']
                for word in words:
                    ids.append(self.dictionary.add_word(word))
                line_idx += 1
        return ids
        ### END YOUR SOLUTION


def batchify(data, batch_size, device, dtype):
    ### BEGIN YOUR SOLUTION
    data_len = len(data)
    nbatch = data_len // batch_size
    data = data[:nbatch * batch_size]
    return np.array(data).reshape(batch_size, -1).T
    ### END YOUR SOLUTION


def get_batch(batches, i, bptt, device=None, dtype=None):
    ### BEGIN YOUR SOLUTION
    data = batches[i: i + bptt, :]
    target = batches[i + 1: i + 1 + bptt, :]
    return Tensor(data, device=device, dtype=dtype), Tensor(target.flatten(), device=device, dtype=dtype)
    ### END YOUR SOLUTION
```
___

### Explain Class `Dictionary`

The `Dictionary` class is responsible for creating a mapping between words and unique integer indices, which is useful for converting text data (words) into a format (numbers) that can be processed by machine learning models.

#### `__init__` Method
```python
def __init__(self):
    self.word2idx = {}
    self.idx2word = []
```
-   **`self.word2idx`**: This is a dictionary where each word (key) is mapped to a unique index (value). It will store the mapping from a word to its corresponding index.
-   **`self.idx2word`**: This is a list that stores words where each position in the list corresponds to the index of the word. The word at position `i` in this list has the index `i`.

#### Example after `__init__`:

If no words are added yet:
```python
word2idx = {}         # Empty dictionary
idx2word = []         # Empty list
```
#### `add_word` Method
```python
def add_word(self, word):
    if self.word2idx.get(word) is None:
        self.word2idx[word] = len(self.idx2word)
        self.idx2word.append(word)
    return self.word2idx[word]
```

-   **`if self.word2idx.get(word) is None`:** This checks if the word is already in the dictionary `word2idx`. If `word2idx.get(word)` returns `None`, it means the word does not yet exist in the dictionary.
    
-   **`self.word2idx[word] = len(self.idx2word)`:** If the word is not already in `word2idx`, it assigns the current length of `idx2word` as the index for that word. This works because the length of the list represents the next available index (since list indices start from 0). For example, if `idx2word` contains 3 words, then the next word will be given index 3.
    
-   **`self.idx2word.append(word)`:** Adds the word to the `idx2word` list. The position in the list corresponds to the index assigned to the word.
    
-   **`return self.word2idx[word]`:** After adding the word, it returns the index assigned to the word.
    

#### Example after `add_word`:

Let’s say we start adding words:

1.  Add the word `"cat"`:
```python
word2idx = {"cat": 0}    # 'cat' gets index 0
idx2word = ["cat"]       # 'cat' is added at index 0
```
-   Now, if you call `add_word("cat")`, it returns `0` (the index of `"cat"`).
    
2.    Add the word `"dog"`:
```python
word2idx = {"cat": 0, "dog": 1}    # 'dog' gets index 1
idx2word = ["cat", "dog"]          # 'dog' is added at index 1
```
Now, if you call `add_word("dog")`, it returns `1` (the index of `"dog"`).

#### `__len__` Method

```python
def __len__(self):
    return len(self.idx2word)
```
-   **`len(self.idx2word)`:** This returns the number of unique words in the dictionary, which is simply the length of the `idx2word` list (since each unique word is added to this list exactly once).

#### Example for `__len__`:

-   After adding `"cat"` and `"dog"`:
```python
len(dictionary)  # Returns 2 because 'cat' and 'dog' are added
```
#
### Full Example:

Let’s walk through an example from the beginning:

1.  **Initialize the Dictionary**:
```python
dictionary = Dictionary()
```
At this point:
```python
word2idx = {}
idx2word = []
```
2. **Add a Word**:
```python
dictionary.add_word("cat")
```
After adding `"cat"`:
```python
word2idx = {"cat": 0}
idx2word = ["cat"]
```
-   The function returns `0`, which is the index assigned to `"cat"`.
    
3.   **Add Another Word**:
```python
dictionary.add_word("dog")
```
After adding `"dog"`:
```python
word2idx = {"cat": 0, "dog": 1}
idx2word = ["cat", "dog"]
```
-   The function returns `1`, which is the index assigned to `"dog"`.
    
4.    **Add a Word that Already Exists**:
```python
dictionary.add_word("cat")
```
-   Since `"cat"` is already in the dictionary, it simply returns the existing index `0`.
    
5.   **Get the Length of the Dictionary**:
```python
len(dictionary)
```
This returns `2` because there are two unique words in the dictionary: `"cat"` and `"dog"`.

### Class: `Corpus`

The `Corpus` class is responsible for reading and processing the text data, such as the Penn Treebank dataset, and converting it into a sequence of word indices. It uses the `Dictionary` class to handle word-to-index mapping.

#### `__init__` Method
```python
def __init__(self, base_dir, max_lines=None):
    self.dictionary = Dictionary()
    self.train = self.tokenize(os.path.join(base_dir, 'train.txt'), max_lines)
    self.test = self.tokenize(os.path.join(base_dir, 'test.txt'), max_lines)
```
-   **`self.dictionary = Dictionary()`**: This initializes a `Dictionary` object, which will be used to convert words into unique indices.
    
-   **`self.train = self.tokenize(...)`**: This calls the `tokenize` method to process the training data from `train.txt`, converting it into a list of word indices.
    
-   **`self.test = self.tokenize(...)`**: Similarly, this tokenizes the test data from `test.txt` and converts it into a list of indices.
    
-   **`max_lines=None`**: This optional argument allows you to limit the number of lines to process from the text files, which is useful for debugging or training with smaller datasets.

#### `tokenize` Method
```python
def tokenize(self, path, max_lines=None):
    with open(path, 'r') as f:
        ids = []
        line_idx = 0
        for line in f:
            if max_lines is not None and line_idx >= max_lines:
                break
            words = line.split() + ['<eos>']
            for word in words:
                ids.append(self.dictionary.add_word(word))
            line_idx += 1
    return ids
```
-   **`with open(path, 'r') as f:`**: Opens the file at `path` (e.g., `train.txt` or `test.txt`) for reading.
    
-   **`ids = []`**: Initializes an empty list `ids` to store the word indices.
    
-   **`for line in f:`**: Iterates through each line of the file.
    
-   **`if max_lines is not None and line_idx >= max_lines: break`**: If `max_lines` is specified and we have processed the specified number of lines, stop the loop. This is useful to limit the size of the dataset.
    
-   **`words = line.split() + ['<eos>']`:** This splits each line of text into a list of words and adds an end-of-sentence token (`<eos>`) to signify the end of the line.
    
-   **`for word in words: ids.append(self.dictionary.add_word(word))`:** This loops through the words, converts each word to its index using `add_word` from the `Dictionary` class, and appends the index to the `ids` list.
    
-   **`line_idx += 1`:** Increments the line counter after processing each line.
    
-   **`return ids`:** Returns the list of word indices representing the entire file.

##### Example of `tokenize`:

Assume `train.txt` contains:

```python
the cat sat on the mat
the dog barked loudly
```
After tokenization, the result could be:
```python
ids = [0, 1, 2, 3, 0, 4, '<eos>_idx', 0, 5, 6, 7, '<eos>_idx']
```
Where:

-   `'the' = 0`
-   `'cat' = 1`
-   `'sat' = 2`
-   `'on' = 3`
-   `'mat' = 4`
-   `'dog' = 5`
-   `'barked' = 6`
-   `'loudly' = 7`
-   `'<eos>'` is the index for the end-of-sentence token.

#### `batchify` Function

This function divides the sequence of data into multiple batches for parallel processing by the model.

```python
def batchify(data, batch_size, device, dtype):
    data_len = len(data)
    nbatch = data_len // batch_size
    data = data[:nbatch * batch_size]
    return np.array(data).reshape(batch_size, -1).T
```

-   **`data_len = len(data)`**: This calculates the total length of the data (i.e., the number of word indices).
    
-   **`nbatch = data_len // batch_size`:** This calculates how many full batches can be created from the data. The `//` operator ensures integer division, meaning any remaining data that doesn't fit into a full batch is discarded.
    
-   **`data = data[:nbatch * batch_size]`:** This trims the data to ensure that it can be evenly divided into `batch_size`. Any extra words at the end are discarded.
    
-   **`np.array(data).reshape(batch_size, -1).T`:** This reshapes the trimmed data into a 2D array where:
    
    -   Each **column** represents one batch.
    -   Each **row** represents the next step in the sequence for a batch.
    -   `.T` transposes the array so that each column represents an independent sequence of word indices.

#### `get_batch` Function

The `get_batch` function takes the batched data and extracts a segment (or chunk) of length `bptt` (backpropagation through time). This chunk will be used as input for the model, and the corresponding next word will be used as the target label.

```python
def get_batch(batches, i, bptt, device=None, dtype=None):
    data = batches[i: i + bptt, :]
    target = batches[i + 1: i + 1 + bptt, :]
    return Tensor(data, device=device, dtype=dtype), Tensor(target.flatten(), device=device, dtype=dtype)
```
-   **`data = batches[i: i + bptt, :]`:** This selects a segment of `bptt` length from the batched data. This is the input for the model.
    
-   **`target = batches[i + 1: i + 1 + bptt, :]`:** This selects the next segment (shifted by 1 word), which will serve as the target for prediction. The model will try to predict this based on the input `data`.
    
-   **`Tensor(data, device=device, dtype=dtype)`**: Converts the `data` into a tensor for use in the model, allowing for GPU acceleration if needed.
    
-   **`Tensor(target.flatten(), device=device, dtype=dtype)`**: Converts the `target` into a flattened tensor. `flatten()` ensures that the target labels are in a single dimension rather than a matrix, as they will be predicted individually by the model.

##### Example of `get_batch`:

Assume the batched data looks like this:
```python
┌ a e i ┐
│ b f j │
│ c g k │
└ d h l ┘
```
With `bptt = 2` and starting at `i = 0`, the input (`data`) and target (`target`) will be:
```css
data  = ┌ a e i ┐
        └ b f j ┘

target = ┌ b f j ┐
         └ c g k ┘
```
The target is shifted by one position from the input data. So, for the input `[a, b]`, the model will try to predict `[b, c]`.

#### Summary:

-   **Corpus Class**: Reads the dataset and converts the text into a sequence of word indices using a `Dictionary`.
-   **`batchify`**: Divides the sequence into equal-length batches for parallel training.
-   **`get_batch`**: Extracts a chunk of the sequence for the model to process, with the next word in the sequence being the target for prediction.

### **Backpropagation Through Time**(BPTT)

**bptt** stands for **Backpropagation Through Time**, which is a method used for training sequence models like RNNs or LSTMs.

In sequence models, you typically process sequences of data step by step, but instead of performing gradient descent after every single time step (which is inefficient), you process chunks of data (sequences) and then perform backpropagation over these chunks.

-   **`bptt`**: This parameter controls how long each chunk of the sequence should be. If `bptt = 30`, for example, you will take chunks of 30 time steps from the sequence, and the model will process these 30 steps together before performing backpropagation.

The value of `bptt` directly affects how much context the model will see at once. A larger `bptt` means the model can learn dependencies across longer sequences, but it also increases memory usage and computational cost.
#### Summary of Processing:

-   For each chunk (e.g., `batches[0:10, :]`), the model:
    -   **Performs a forward pass** over the input data to make predictions.
    -   **Computes the loss** by comparing the predictions to the target (e.g., `batches[1:11, :]`).
    -   **Performs a backward pass** (backpropagation) over the chunk to update the model's weights based on the error.

After processing one chunk (both forward and backward passes), the model moves to the next chunk (e.g., from `batches[0:10, :]` to `batches[10:20, :]`), and the process repeats.
