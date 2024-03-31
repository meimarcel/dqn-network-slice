import itertools
import numpy as np
import torch.nn as nn
import math

from config import *


def initialize_weights(net):
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            m.weight.data.normal_(0, 0.1)
            m.bias.data.zero_()
        elif isinstance(m, nn.ConvTranspose2d):
            m.weight.data.normal_(0, 1)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            m.bias.data.zero_()


def generate_actions():
    tmp = list(itertools.product(range(6), repeat=3))

    result = []

    for value in tmp:
        if sum(value) == 5:
            result.append(list(value))

    return result


def normalize_state(state):
    if state is None:
        return None
    
    discrete_state = np.zeros(len(state))

    if np.all(state) == 0:
        return discrete_state

    for i in range(len(state)):
        discrete_state[i] = state[i]

    discrete_state = (discrete_state - discrete_state.mean()) / discrete_state.std()

    return discrete_state.tolist()


def get_epsilon(episode):
    return EPSILON_END + (EPSILON_START - EPSILON_END) * math.exp(-1.0 * episode / EPSILON_DECAY)
