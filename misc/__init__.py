from .LinearModel import LinearModel
from .CNNModel import CNNModel
from .ReplayBuffer import ReplayBuffer
from .CategoricalModel import CategoricalModel
import torch.nn as nn
from typing import Tuple, List

def get_model(in_dim: Tuple, sizes: List, activation=nn.ReLU(), output_activation=nn.Identity(), cnn: bool = False):
    if(cnn != None):
        return CNNModel(in_dim, cnn, sizes, activation, output_activation)
    return LinearModel(in_dim, sizes, activation, output_activation)