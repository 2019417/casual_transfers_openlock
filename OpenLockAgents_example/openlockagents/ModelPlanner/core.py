#!/usr/bin/env python
# -*- coding:UTF-8 -*-
# File Name : core.py
# Creation Date : 16-07-2018
# Created By : Jeasine Ma [jeasinema[at]gmail[dot]com]

import numpy as np
import torch
from torch.autograd import Variable
import torch.nn.functional as F

from .utils import *


def update_params(
    world,
    optimizer_world,
    batch,
    tensor_type,
    action_tensor_type,
    epsilon,
    l2_reg,
    i_iter,
    scenario,
    i_trial,
):
    states = tensor_type(batch.state)
    actions = action_tensor_type(batch.action)
    rewards = tensor_type(batch.reward)
    masks = tensor_type(batch.mask)
    next_states = tensor_type(batch.next_state)
    simulated_states ,simulated_rewards = world(actions.unsqueeze(1), states)
    rewards_loss = F.mse_loss(rewards.unsqueeze(1), simulated_rewards)
    states_loss = F.mse_loss(next_states, simulated_states, reduction = 'sum')
    total_loss = epsilon*rewards_loss + states_loss
    for param in world.parameters():
        total_loss += param.pow(2).sum() * l2_reg
    print(f'iter{i_iter}scene{scenario}{i_trial}: total loss:{total_loss}, rewards_loss:{rewards_loss}, states_loss:{states_loss}')
    optimizer_world.zero_grad()
    total_loss.backward()
    optimizer_world.step()
    return total_loss, rewards_loss, states_loss


if __name__ == "__main__":
    pass
