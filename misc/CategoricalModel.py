import torch
import torch.nn as nn
import numpy as np
from torch.distributions import Categorical
from .LinearModel import LinearModel

class CategoricalModel(LinearModel):
    def __init__(self, in_dim, sizes, activation=nn.ReLU(), output_activation=nn.Identity()):
        super().__init__(in_dim, sizes, activation, output_activation)
        
    
    def get_distribution(self, x):
        logits = self.model(x)
        return Categorical(logits=logits)

    def get_act_prob(self, x, act):
        dist = self.get_distribution(x)
        return dist.log_prob(act), dist.entropy().mean()