
### LayerNorm1d

`needle.nn.LayerNorm1d(dim, eps=1e-5, device=None, dtype="float32")`

Applies layer normalization over a mini-batch of inputs as described in the paper [Layer Normalization](https://arxiv.org/abs/1607.06450).

  
$$y = w \circ \frac{x_i - \textbf{E}[x]}{((\textbf{Var}[x]+\epsilon)^{1/2})} + b$$


where $\textbf{E}[x]$ denotes the empirical mean of the inputs, $\textbf{Var}[x]$ denotes their empirical variance (note that here we are using the "biased" estimate of the variance, i.e., dividing by $N$ rather than by $N-1$), and $w$ and $b$ denote learnable scalar weights and biases respectively. Note you can assume the input to this layer is a 2D tensor, with batches in the first dimension and features in the second. You might need to broadcast the weight and bias before applying them.

  
##### Parameters

- `dim` - number of channels

- `eps` - a value added to the denominator for numerical stability.

  

##### Variables

- `weight` - the learnable weights of size `dim`, elements initialized to 1.

- `bias` - the learnable bias of shape `dim`, elements initialized to 0.


Code Implementation:
```python
```

___
