import torch
from torch.autograd import Variable

# Initialize

x = Variable(torch.ones(2, 2))

x = torch.Tensor(2, 3)  # An un-initialized Tensor object
y = torch.rand(2, 3)    # Initialize with random values

z1 = x + y

result = torch.Tensor(2, 3)
torch.add(x, y, out=result)

print(x)                        # [torch.FloatTensor of size 2x3]
