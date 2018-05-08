import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution kernel
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)

        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))

        # 2 is ame as (2, 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)

        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:       # Get the products
            num_features *= s
        return num_features


net = Net()
print(net)
# Net(
#  (conv1): Conv2d (1, 6, kernel_size=(5, 5), stride=(1, 1))
#  (conv2): Conv2d (6, 16, kernel_size=(5, 5), stride=(1, 1))
#  (fc1): Linear(in_features=400, out_features=120)
#  (fc2): Linear(in_features=120, out_features=84)
#  (fc3): Linear(in_features=84, out_features=10)
#)


params = list(net.parameters())
print(len(params))

print(params[0].size())  # conv1's .weight

input = Variable(torch.randn(1, 1, 32, 32))
out = net(input)  # out's size: 1x10.
print(out)

net.zero_grad()
out.backward(torch.randn(1, 10))

output = net(input)
target = Variable(torch.arange(1, 11))   # Create a dummy true label Size 10.
criterion = nn.MSELoss()

loss = criterion(output, target)         # Size 1
print(loss)

print(loss.grad_fn)                      # <MseLossBackward object at 0x10d729908>
print(loss.grad_fn.next_functions[0][0]) # <AddmmBackward object at 0x10d729400>
print(loss.grad_fn.next_functions[0][0].next_functions[0][0])  # <ExpandBackward object at 0x10fd39e48>

net.zero_grad()     # zeroes the gradient buffers of all parameters

print('conv1.bias.grad before backward')
print(net.conv1.bias.grad)
# Variable containing:
# 0
# 0
# 0
# 0
# 0
# 0
# [torch.FloatTensor of size 6]

loss.backward()

print('conv1.bias.grad after backward')
print(net.conv1.bias.grad)
# Variable containing:
# -0.0007
# -0.0400
# 0.0184
# 0.1273
# -0.0080
# 0.0387
# [torch.FloatTensor of size 6]

import torch.optim as optim

# Create a SGD optimizer for gradient descent
optimizer = optim.SGD(net.parameters(), lr=0.01)

# Inside the training loop
# ...
optimizer.zero_grad()   # zero the gradient buffers
output = net(input)
loss = criterion(output, target)
loss.backward()
optimizer.step()        # Perform the training parameters update

