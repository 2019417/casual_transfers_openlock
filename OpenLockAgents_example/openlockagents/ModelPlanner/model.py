#!/usr/bin/env python
# -*- coding:UTF-8 -*-
# File Name : model.py
# Creation Date : 16-07-2018
# Created By : Jeasine Ma [jeasinema[at]gmail[dot]com]

import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable

from .utils import *

class World(nn.Module):
    def __init__(self, num_actions, num_states, hidden_size=(128, 128), activation = "tanh"):
        super().__init__()
        if activation == "tanh":
            self.activation = F.tanh
        elif activation == "relu":
            self.activation = F.relu
        elif activation == "sigmoid":
            self.activation = F.sigmoid
        self.layers = nn.ModuleList()
        last_dim = 1 + num_states
        for nh in hidden_size:
            self.layers.append(nn.Linear(last_dim, nh))
            last_dim = nh
        self.reward_head = nn.Linear(last_dim, 1)
        self.state_head = nn.Linear(last_dim, num_states)
        self.reward_head.weight.data.mul_(0.1)
        self.reward_head.bias.data.mul_(0.0)
        self.state_head.weight.data.mul_(0.1)
        self.state_head.bias.data.mul_(0.0)
        


    def forward(self, action, state):
        #print(action.shape, state.shape)
        x = torch.cat((state, action), dim=1)
        for layer in self.layers:
            x = self.activation(layer(x))
        return self.state_head(x)+state, self.reward_head(x), 



if __name__ == "__main__":
    pass
