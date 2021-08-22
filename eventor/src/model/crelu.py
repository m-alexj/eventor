'''
Created on Oct 20, 2017

@author: judeaax
'''
from torch import nn
import torch
import torch.nn.functional as F


class CReLU(nn.Module):

    def __init__(self):
        super(CReLU, self).__init__()

    def forward(self, x):
        return torch.cat((F.relu(x), F.relu(-x)), 1)