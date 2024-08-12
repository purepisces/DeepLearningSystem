## Question 3: Topological sort [20 pts]

Now your system is capable of performing operations on tensors which builds up a computation graph. Next you will write one of the main utilities needed for automatic differentiation - the [topological sort](https://en.wikipedia.org/wiki/Topological_sorting). This will allow us to traverse through (forward or backward) the compuatation graph, computing gradients along the way. Furthermore, the previously built components will allow for the operations we perform during this reverse topological traversal to further add to our computation graph (as discussed in lecture), and will therefore give us higher-order differentiation "for free." 

Fill out the `find_topo_sort` method and the `topo_sort_dfs` helper method (in `python/needle/autograd.py`) to perform this topological sorting. 

#### Hints: 
- Ensure that you do a post-order depth-first search, otherwise the test cases will fail. 
- The `topo_sort_dfs` method is not required, but we find it useful to use this as a recursive helper function. 
- The "Reverse mode AD by extending computational graph" section of the Lecture 4 slides contains walks through an example of the proper node ordering. 
- We will be traversing this sorting backwards in later parts of this homework, but the `find_topo_sort` should return the node ordering in the forward direction.

--------------------------------------------
Refer to leetcode: https://leetcode.com/problems/course-schedule-ii/

Topological sorting is a concept used in computer science to arrange the nodes (or vertices) of a directed graph in a specific order. This order is such that for every directed edge connecting two nodes, the node from which the edge starts (let's call it node u) appears before the node where the edge ends (let's call it node v) in the sequence.

Imagine you have a series of tasks to complete, where some tasks must be done before others. For example, you can't bake a cake until you've mixed the ingredients. In this scenario, topological sorting would give you a valid order to complete all tasks, ensuring that you follow the necessary prerequisites.

However, this type of sorting is only possible if the graph doesn't contain any cycles (loops), meaning you don't have a situation where a task depends on itself either directly or indirectly. If there are no cycles, the graph is called a Directed Acyclic Graph (DAG), and you can always find at least one valid topological order.

To find a topological order, algorithms exist that can do this efficiently, meaning they can process the graph in time proportional to the number of nodes and edges.

Topological sorting is widely used in various applications, such as scheduling tasks, resolving dependencies in programming (like determining the order in which to compile files), and ranking problems. Even if the graph has separate, unconnected parts, topological sorting can still be applied to each part independently.

```python
def find_topo_sort(node_list: List[Value]) -> List[Value]:
    """Given a list of nodes, return a topological sort list of nodes ending in them.

    A simple algorithm is to do a post-order DFS traversal on the given nodes,
    going backwards based on input edges. Since a node is added to the ordering
    after all its predecessors are traversed due to post-order DFS, we get a topological
    sort.
    """
    ### BEGIN YOUR SOLUTION
    visited = set()
    topo_order = []

    for node in node_list:
        topo_sort_dfs(node, visited, topo_order)
        
    return topo_order
    ### END YOUR SOLUTION


def topo_sort_dfs(node, visited, topo_order):
    """Post-order DFS"""
    ### BEGIN YOUR SOLUTION
    if node in visited:
        return
    visited.add(node)
    for input_node in node.inputs:
        topo_sort_dfs(input_node, visited, topo_order)
    topo_order.append(node)
    ### END YOUR SOLUTION
```
## Question 4: Implementing reverse mode differentiation [25 pts]

Once you have correctly implemented the topological sort, you will next leverage it to implement reverse mode automatic differentiation. As a recap from last lecture, we will need to traverse the computational graph in reverse topological order, and construct the new adjoint nodes. For this question, implement the Reverse AD algorithm in the `compute_gradient_of_variables` function in `python/needle/autograd.py`. This will enable use of the `backward` function that computes the gradient and stores the gradient in the `grad` field of each input `Tensor`. With this completed, our reverse model autodifferentiation engine is functional. We can check the correctness of our implementation in much the same way that we numerically checked the individual backward gradients, by comparing the numerical gradient to the computed one, using the function `gradient_check` in `tests/test_autograd.py`.


As discussed in lecture the result of reverse mode AD is still a computational graph. We can extend that graph further by composing more operations and run reverse mode AD again on the gradient (the last two tests of this problem). 


