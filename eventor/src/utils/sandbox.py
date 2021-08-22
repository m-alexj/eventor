from torch import nn, autograd
import torch

m = nn.LogSoftmax()
loss = nn.NLLLoss()
# input is of size nBatch x nClasses = 3 x 5
input = autograd.Variable(torch.FloatTensor([0.0, 1.0]), requires_grad=True)
# each element in target has to have 0 <= value < nclasses
target = autograd.Variable(torch.LongTensor([1]))
input = m(input)
print(input)
output = loss(input, target)
output.backward()
print(output)