import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from copy import deepcopy
import gym
from stable_baselines3 import PPO

class Model(nn.Module):
    def __init__(self, in_dim=1, out_dim=1, hidden_dim=256, softmax=False):
        super(Model, self).__init__()
        self.l1 = nn.Linear(in_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.l3 = nn.Linear(hidden_dim, out_dim)
        self.softmax = softmax

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        if self.softmax:
            return F.softmax(self.l3(x))
        return self.l3(x)

class LinearModel(nn.Module):
    def __init__(self, in_dim=1, out_dim=1, bias=False):
        super(LinearModel, self).__init__()
        self.l1 = nn.Linear(in_dim, out_dim, bias=bias)

    def forward(self, x):
        return self.l1(x)

def init_weights(m):
    if type(m) == nn.Linear:
        m.weight.data.uniform_(-1.0, 1.0)
        m.bias.data.uniform_(-1.0, 1.0)