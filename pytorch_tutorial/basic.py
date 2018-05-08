import torch
import numpy as np

x = torch.Tensor(2, 3)          # An un-initialized Tensor object with size 2x3

print(x)                        # [torch.FloatTensor of size 2x3]

# 0.0000e+00 -1.5846e+29  2.4140e-35
# 1.0845e-19  2.2335e+08  2.2369e+08
# [torch.FloatTensor of size 2x3]

### Create a Tensor

v = torch.Tensor(2, 3)          # An un-initialized torch.FloatTensor of size 2x3
v = torch.Tensor([[1,2],[4,5]]) # A Tensor initialized with a specific array
v = torch.LongTensor([1,2,3])   # A Tensor of type Long

v = torch.rand(2, 3)            # Initialize with random number (uniform distribution)
v = torch.randn(2, 3)           # With normal distribution (SD=1, mean=0)
v = torch.randperm(4)           # Size 4. Random permutation of integers from 0 to 3

eye = torch.eye(3)              # Create an identity 3x3 tensor

v = torch.ones(10)              # A tensor of size 10 containing all ones
v = torch.ones(2, 1, 2, 1)      # Size 2x1x2x1
v = torch.ones_like(eye)        # A tensor with same shape as eye. Fill it with 1.

v = torch.zeros(10)             # A tensor of size 10 containing all zeros

v = torch.arange(5)             # similar to range(5) but creating a Tensor
v = torch.arange(0, 5, step=1)  # Size 5. Similar to range(0, 5, 1)

v = torch.linspace(1, 10, steps=10) # Create a Tensor with 10 linear points for (1, 10) inclusively
v = torch.logspace(start=-10, end=10, steps=5) # Size 5: 1.0e-10 1.0e-05 1.0e+00, 1.0e+05, 1.0e+10

# 1  1  1
# 2  2  2
# 3  3  3
v = torch.ones(3, 3)
v[1].fill_(2)
v[2].fill_(3)

# Conversion

a = np.array([1, 2, 3])
v = torch.from_numpy(a)         # Convert a numpy array to a Tensor

b = v.numpy()                   # Tensor to numpy
b[1] = -1                       # Numpy and Tensor share the same memory
assert(a[1] == b[1])            # Change Numpy will also change the Tensor

### Basic Tensor operation

x.size()                        # torch.Size([2, 3])
torch.numel(x)                  # 6: number of elements in x

### Tensor resizing
x = torch.randn(2, 3)            # Size 2x3
y = x.view(6)                    # Resize x to size 6
z = x.view(-1, 2)                # Size 3x2

### Operations
x = torch.eye(3)
y = torch.ones_like(eye)

z1 = x + y
z2 = torch.add(x, y)             # Same as above

result = torch.Tensor(2, 3)

# All operations have an "out" parameter to store the result
torch.add(x, y, out=result)      # result = x + y

# All in-place operator ends with "_"
y.add_(x)                        # y = y + x

# Index
x[:, 1]                          # Can use numpy type indexing
x[:, 0] = 0                      # For assignment

### Indexing, Slicing, Joining, Mutating Ops

# Concatenation
torch.cat((x, x, x), 0)          # Concatenate in the 0 dimension

# 0 1 2
# 3 4 5
# 6 7 8
v = torch.arange(9)
v = v.view(3, 3)

# Gather element
# torch.gather(input, dim, index, out=None)
# out[i][j][k] = input[index[i][j][k]][j][k]  # if dim == 0
# out[i][j][k] = input[i][index[i][j][k]][k]  # if dim == 1
# out[i][j][k] = input[i][j][index[i][j][k]]  # if dim == 2

# 0  1
# 4  3
# 8  7
r = torch.gather(v, 1, torch.LongTensor([[0,1],[1,0],[2,1]]))

# Split an array into 3 chunks
# (
# 0  1  2
# [torch.FloatTensor of size 1x3]
# ,
# 3  4  5
# [torch.FloatTensor of size 1x3]
# ,
# 6  7  8
# [torch.FloatTensor of size 1x3]
# )
r = torch.chunk(v, 3)

# Index select
# 0 2
# 3 5
# 6 8
indices = torch.LongTensor([0, 2])
r = torch.index_select(v, 1, indices) # Select element 0 and 2 for each dimension 1.

# Masked select
# 0  0  0
# 1  1  1
# 1  1  1
mask = v.ge(3)

# Size 6: 3 4 5 6 7 8
r = torch.masked_select(v, mask)

# Non-zero
# [torch.LongTensor of size 8x2]
# [i, j] index for non-zero elements
#    0     1
#    0     2
#    1     0
#    1     1
#    1     2
#    2     0
#    2     1
#    2     2
r = torch.nonzero(v)

# Split an array into chunks of at most size 2
# (
# 0  1  2
# 3  4  5
# [torch.FloatTensor of size 2x3]
# ,
# 6  7  8
# [torch.FloatTensor of size 1x3]
# )
r = torch.split(v, 2)

t = torch.ones(2,1,2,1) # Size 2x1x2x1
r = torch.squeeze(t)     # Size 2x2
r = torch.squeeze(t, 1)  # Squeeze dimension 1: Size 2x2x1

# Stack
r = torch.stack((v, v))

# Flatten a TensorFlow and return elements with given indexes
# Size 3: 0, 4, 2
r = torch.take(v, torch.LongTensor([0, 4, 2]))

# Transpose dim 0 and 1
r = torch.transpose(v, 0, 1)

# Un-squeeze a dimension
x = torch.Tensor([1, 2, 3])
r = torch.unsqueeze(x, 0)       # Size: 1x3
r = torch.unsqueeze(x, 1)       # Size: 3x1

c = torch.ByteTensor([0, 1, 1, 0])

### Distribution

# Uniform distributed

# 2x2: A uniform distributed random matrix with range [0, 1]
r = torch.Tensor(2, 2).uniform_(0, 1)

# bernoulli
r = torch.bernoulli(r)   # Size: 2x2. Bernoulli with probability p stored in elements of r

# Multinomial
w = torch.Tensor([0, 4, 8, 2]) # Create a tensor of weights
r = torch.multinomial(w, 4, replacement=True) # Size 4: 3, 2, 1, 2

# Normal distribution
# From 10 means and SD
r = torch.normal(means=torch.arange(1, 11), std=torch.arange(1, 0.1, -0.1)) # Size 10

### Math operations
f= torch.FloatTensor([-1, -2, 3])
r = torch.abs(f)      # 1 2 3

# Add x, y and scalar 10 to all elements
r = torch.add(x, 10)
r = torch.add(x, 10, y)

# Clamp the value of a Tensor
r = torch.clamp(v, min=-0.5, max=0.5)

# Element-wise divide
r = torch.div(v, v+0.03)

# Element-wise multiple
r = torch.mul(v, v)


### Reduction operations

# Accumulate sum
#  0   1   2
#  3   5   7
#  9  12  15
r = torch.cumsum(v, dim=0)

# L-P norm
r = torch.dist(v, v+3, p=2)  # L-2 norm: ((3^2)*9)^(1/2) = 9.0

# Mean
# 1 4 7
r = torch.mean(v, 1)         # Size 3: Mean in dim 1

r = torch.mean(v, 1, True)   # Size 3x1 since keep dimension = True

# Sum
# 3 12 21
r = torch.sum(v, 1)          # Sum over dim 1

# 36
r = torch.sum(v)

### Comparison
# Size 3x3: Element-wise comparison
r = torch.eq(v, v)

# k-th element (start from 1) ascending order with corresponding index
# (1 4 7
# [torch.FloatTensor of size 3]
# , 1 1 1
# [torch.LongTensor of size 3]
# )
r = torch.kthvalue(v, 2)

# Max element with corresponding index
r = torch.max(v, 1)

# Sort
# (
#  0  1  2
#  3  4  5
#  6  7  8
# [torch.FloatTensor of size 3x3]
# ,
#  0  1  2
#  0  1  2
#  0  1  2
# [torch.LongTensor of size 3x3]
r = torch.sort(v, 1)

# Top k
# (
#  2  5  8
# [torch.FloatTensor of size 3x1]
# ,
#  2  2  2
# [torch.LongTensor of size 3x1]
# )
r = torch.topk(v, 1)

m1 = torch.ones(3, 5)
m2 = torch.ones(3, 5)
v1 = torch.ones(3)

# Cross product
# Size 3x5
r = torch.cross(m1, m2)

# Diagonal matrix
# Size 3x3
r = torch.diag(v1)

# Histogram
# [0, 2, 1, 0]
torch.histc(torch.FloatTensor([1, 2, 1]), bins=4, min=0, max=3)

# Renormalize
# Renormalize for L-1 at dim 0 with max of 1
# 0.0000  0.3333  0.6667
# 0.2500  0.3333  0.4167
# 0.2857  0.3333  0.3810
r = torch.renorm(v, 1, 0, 1)


### Matrix, vector products

# Matrix X vector
# Size 2x4
mat = torch.randn(2, 4)
vec = torch.randn(4)
r = torch.mv(mat, vec)

# Matrix + Matrix X vector
# Size 2
M = torch.randn(2)
mat = torch.randn(2, 3)
vec = torch.randn(3)
r = torch.addmv(M, mat, vec)

# Matrix x Matrix
# Size 2x4
mat1 = torch.randn(2, 3)
mat2 = torch.randn(3, 4)
r = torch.mm(mat1, mat2)

# Matrix + Matrix X Matrix
# Size 3x4
M = torch.randn(3, 4)
mat1 = torch.randn(3, 2)
mat2 = torch.randn(2, 4)
r = torch.addmm(M, mat1, mat2)

# Dot product of 2 tensors
r = torch.dot(torch.Tensor([4, 2]), torch.Tensor([3, 1])) # 14

# Outer product of 2 vectors
# Size 3x2
v1 = torch.arange(1, 4)    # Size 3
v2 = torch.arange(1, 3)    # Size 2
r = torch.ger(v1, v2)

# Add M with outer product of 2 vectors
# Size 3x2
vec1 = torch.arange(1, 4)  # Size 3
vec2 = torch.arange(1, 3)  # Size 2
M = torch.zeros(3, 2)
r = torch.addr(M, vec1, vec2)

# Batch Matrix x Matrix
# Size 10x3x5
batch1 = torch.randn(10, 3, 4)
batch2 = torch.randn(10, 4, 5)
r = torch.bmm(batch1, batch2)

# Batch Matrix + Matrix x Matrix
# Performs a batch matrix-matrix product
# 3x4 + (5x3x4 X 5x4x2 ) -> 5x3x2
M = torch.randn(3, 2)
batch1 = torch.randn(5, 3, 4)
batch2 = torch.randn(5, 4, 2)
r = torch.addbmm(M, batch1, batch2)

# Move Tensors to GPU
if torch.cuda.is_available():
    x = x.cuda()
    y = y.cuda()
    x + y

print(r)
