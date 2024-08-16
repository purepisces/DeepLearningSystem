## Question 2

In this question, you will implement additional modules in `python/needle/nn/nn_basic.py`. Specifically, for the following modules described below, initialize any variables of the module in the constructor, and fill out the `forward` method. **Note:** Be sure that you are using the `init` functions that you just implemented to initialize the parameters, and don't forget to pass the `dtype` argument.
___

### Linear
`needle.nn.Linear(in_features, out_features, bias=True, device=None, dtype="float32")`

Applies a linear transformation to the incoming data: $y = xA^T + b$. The input shape is $(N, H_{in})$ where $H_{in}=\text{infeatures}$. The output shape is $(N, H_{out})$ where $H_{out}=\text{outfeatures}$.

Be careful to explicitly broadcast the bias term to the correct shape -- Needle does not support implicit broadcasting.

**Note:** for all layers including this one, you should initialize the weight Tensor before the bias Tensor, and should initialize all Parameters using only functions from `init`.

##### Parameters
- `in_features` - size of each input sample
- `out_features` - size of each output sample
- `bias` - If set to `False`, the layer will not learn an additive bias.

##### Variables
- `weight` - the learnable weights of shape (`in_features`, `out_features`). The values should be initialized with the Kaiming Uniform initialization with `fan_in = in_features`
- `bias` - the learnable bias of shape (1, `out_features`). The values should be initialized with the Kaiming Uniform initialize with `fan_in = out_features`. **Note the difference in fan_in choice, due to their relative sizes**. 
