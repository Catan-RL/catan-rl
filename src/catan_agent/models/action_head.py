import math
import numpy as np
import torch
import torch.nn as nn

from catan_env.game.enums import ActionTypes
from catan_agent.distributions import Categorical

class ActionHead(nn.Module):
    def __init__(self, input_dim, output_dim, mlp_size=128, id = None):
        super(ActionHead, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.mlp_size = mlp_size
        self.id = id

        self.mlp_1 = nn.Linear(self.input_dim, mlp_size)
        self.mlp_2 = nn.Linear(mlp_size, mlp_size)
        self.norm = nn.LayerNorm(mlp_size)
        self.relu = nn.ReLU()

        self.distribution = Categorical(mlp_size, 10)

        # nn.init.orthogonal_(self.weight.data, gain = 0.01)
        # nn.init.constnat(self.bias.data)
        # self.linear = nn.Linear(mlp_size, output_dim)

    def forward(self, input, mask):
        input = self.mlp_2(self.relu(self.norm(self.mlp_1(input))))

        return self.distribution(input, mask)
    
