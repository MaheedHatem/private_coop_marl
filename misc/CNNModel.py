import torch
import torch.nn as nn
import numpy as np

class CNNModel(nn.Module):
    def __init__(self, in_dim, filters, sizes, activation=nn.ReLU(), output_activation=nn.Identity()):
        super().__init__()
        layers = []
        for j in range(len(filters)):
            layers += [nn.Conv2d(*filters[j]), activation, nn.MaxPool2d(2,2)]
        layers += [nn.Flatten()]
        for j in range(len(sizes)-1):
            act = activation if j < len(sizes)-2 else output_activation
            layers += [nn.Linear(sizes[j], sizes[j+1]), act]
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        return self.model(x)