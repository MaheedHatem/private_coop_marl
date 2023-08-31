from .LinearModel import LinearModel
from .CNNModel import CNNModel
from .ReplayBuffer import ReplayBuffer
from .CategoricalModel import CategoricalModel
from .wrappers import *
import torch.nn as nn

def get_model(in_dim, sizes, activation=nn.ReLU(), output_activation=nn.Identity(), cnn= False):
    if(cnn != None):
        return CNNModel(in_dim, cnn, sizes, activation, output_activation)
    return LinearModel(in_dim, sizes, activation, output_activation)