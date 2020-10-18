#!coding=utf-8
import torch
import torch.nn as nn
from torch.nn import functional as F

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
       
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(21*2, 256, bias=True),
            nn.ReLU(),
            nn.Linear(256, 256, bias=True),
            nn.ReLU(),
            nn.Linear(256, 5, bias=True),
            nn.Softmax(dim=1),
        )

    def forward(self, X):  
        X = self.classifier(X)
        return X
 