## Question 3: Topological sort [20 pts]

Now your system is capable of performing operations on tensors which builds up a computation graph. Next you will write one of the main utilities needed for automatic differentiation - the [topological sort](https://en.wikipedia.org/wiki/Topological_sorting). This will allow us to traverse through (forward or backward) the compuatation graph, computing gradients along the way. Furthermore, the previously built components will allow for the operations we perform during this reverse topological traversal to further add to our computation graph (as discussed in lecture), and will therefore give us higher-order differentiation "for free." 

Fill out the `find_topo_sort` method and the `topo_sort_dfs` helper method (in `python/needle/autograd.py`) to perform this topological sorting. 

#### Hints: 
- Ensure that you do a post-order depth-first search, otherwise the test cases will fail. 
- The `topo_sort_dfs` method is not required, but we find it useful to use this as a recursive helper function. 
- The "Reverse mode AD by extending computational graph" section of the Lecture 4 slides contains walks through an example of the proper node ordering. 
- We will be traversing this sorting backwards in later parts of this homework, but the `find_topo_sort` should return the node ordering in the forward direction. 
