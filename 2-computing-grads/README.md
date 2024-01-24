# Computing grads from basic operations

Given matrices $A$ and $B$, the following operations and their gradients are:

- $C = A + B \implies \frac{\partial C}{\partial A} = \frac{\partial C}{\partial B} = I$
- $C = A - B \implies \frac{\partial C}{\partial A} = I, \frac{\partial C}{\partial B} = -I$
- $C = A \cdot B \implies \frac{\partial C}{\partial A} = B^T, \frac{\partial C}{\partial B} = A^T$

# Operation tree

Following the rules above, I'm going to build a tree. The tree will have as root the final result, and the leaves will be the inputs. 

Edges will be labeled with the value of the gradient.

For example, if we have $C = A + B$, the tree will be:

```
Root: C
C -> A: I
C -> B: I
```

Or if we have $C = X \cdot W + B$:

```
Root: C
C -> (X \cdot W)
C -> B: I
(X \cdot W) -> X: W^T
(X \cdot W) -> W: X^T
```
