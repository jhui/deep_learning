import torch
from torch.autograd import Variable

# Variables wrap a Tensor
x = Variable(torch.ones(2, 2), requires_grad=True)
# Variable containing:
# 1  1
# 1  1
# [torch.FloatTensor of size 2x2]

# Define operations:

y = x + 2            # Create y from an operation
# Variable containing:
# 3  3
# 3  3
# [torch.FloatTensor of size 2x2]

print(y.grad_fn)     # The Function that create the Variable y
# <AddBackward0 object at 0x102995438>
print(x.grad_fn)     # None

z = y * y * 2

out = z.mean()
# Variable containing:
# 2
# [torch.FloatTensor of size 1]

out.backward()

print(x.grad)
# Variable containing:
# 3 3
# 3 3
# [torch.FloatTensor of size 2x2]


x = torch.randn(3)
x = Variable(x, requires_grad=True)

y = x * 2
while y.data.norm() < 100:
    y = y * 2

print(y)
# Variable containing:
# 70.1318
# 149.2312
# -78.4707
# [torch.FloatTensor of size 3]

gradients = torch.FloatTensor([0.1, 1.0, 0.0001])
y.backward(gradients)

print(x.grad)
# Variable containing:
#  6.4000
# 64.0000
#  0.0064
# [torch.FloatTensor of size 3]