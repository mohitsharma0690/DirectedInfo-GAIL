import copy
import numpy as np

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F

def square(a):
    return torch.pow(a, 2.)

class Policy(nn.Module):

    def __init__(self, state_size, action_size, latent_size, output_size, hidden_size, output_activation=None):
        super(Policy, self).__init__()
    
        self.input_size = state_size + action_size + latent_size
        self.state_size = state_size
        self.action_size = action_size
        self.latent_size = latent_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.output_activation = output_activation

        self.affine1 = nn.Linear(self.input_size, self.hidden_size)
        self.affine2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.action_mean = nn.Linear(self.hidden_size, self.output_size)
        self.action_mean.weight.data.mul_(0.1)
        self.action_mean.bias.data.mul_(0.0)
        self.action_log_std = nn.Parameter(torch.zeros(1, self.output_size))


    def forward(self, x, old=False):

        x = F.relu(self.affine1(x))
        x = F.relu(self.affine2(x))

        action_mean = self.action_mean(x)
        if self.output_activation == 'sigmoid':
            action_mean = F.sigmoid(self.action_mean(x))
        elif self.output_activation == 'tanh':
            action_mean = F.tanh(self.action_mean(x))
        action_log_std = self.action_log_std.expand_as(action_mean)
        action_std = torch.exp(action_log_std)

        return action_mean, action_log_std, action_std

class Posterior(nn.Module):

    def __init__(self, state_size, action_size, latent_size, hidden_size):
        super(Posterior, self).__init__()
        
        self.input_size = state_size + action_size + latent_size
        self.state_size = state_size
        self.action_size = action_size
        self.latent_size = latent_size
        self.hidden_size = hidden_size
        
        self.affine1 = nn.Linear(self.input_size, self.hidden_size)
        self.affine2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.affine31 = nn.Linear(self.hidden_size, 1)
        self.affine32 = nn.Linear(self.hidden_size, 1)

    def forward(self, x):
        h1 = F.relu(self.affine1(x))
        h2 = F.relu(self.affine2(h1))

        return self.affine31(h2), self.affine32(h2)
            

class Value(nn.Module):
    def __init__(self, num_inputs, hidden_size=100):
        super(Value, self).__init__()
        self.affine1 = nn.Linear(num_inputs, hidden_size)
        self.affine2 = nn.Linear(hidden_size, hidden_size)
        self.value_head = nn.Linear(hidden_size, 1)
        self.value_head.weight.data.mul_(0.1)
        self.value_head.bias.data.mul_(0.0)

    def forward(self, x):
        x = F.relu(self.affine1(x))
        x = F.relu(self.affine2(x))

        state_values = self.value_head(x)
        return state_values

class Reward(nn.Module):
    def __init__(self, state_size, action_size, latent_size, hidden_size=100):
        super(Reward, self).__init__()

        self.input_size = state_size + action_size + latent_size
        self.state_size = state_size
        self.action_size = action_size
        self.latent_size = latent_size
        self.hidden_size = hidden_size

        self.affine1 = nn.Linear(self.input_size, hidden_size)
        self.affine2 = nn.Linear(hidden_size, hidden_size)
        self.reward_head = nn.Linear(hidden_size, 1)
        #self.reward_head.weight.data.mul_(0.1)
        #self.reward_head.bias.data.mul_(0.0)

    def forward(self, x):
        x = F.relu(self.affine1(x))
        x = F.relu(self.affine2(x))

        rewards = F.sigmoid(self.reward_head(x))
        return rewards
