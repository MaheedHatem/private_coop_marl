import torch
import torch.nn as nn
from typing import Tuple, List
import numpy as np

class LinearModel(nn.Module):
    def __init__(self, in_dim: Tuple, sizes: List, activation=nn.ReLU(), output_activation=nn.Identity()):
        super().__init__()
        layers = [nn.Flatten()]
        sizes = [np.product(in_dim)] + sizes
        for j in range(len(sizes)-1):
            act = activation if j < len(sizes)-2 else output_activation
            layers += [nn.Linear(sizes[j], sizes[j+1]), act]
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)