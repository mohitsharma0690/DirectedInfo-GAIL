import math
import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn.init as init


class GRU(nn.Module):
    def __init__(self, state_size, action_size, latent_size, output_size, hidden_size=8, dtype=torch.DoubleTensor, n_layers=2, batch_size=1, policy_flag=1, activation_flag=0):
        super(GRU, self).__init__()
        
        self.input_size = state_size + action_size + latent_size
        self.state_size = state_size
        self.action_size = action_size
        self.latent_size = latent_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.batch_size = batch_size
        self.policy_flag = policy_flag
        self.activation_flag = activation_flag
        self.dtype = dtype

        self.gru1_sa = nn.GRUCell(self.input_size-self.latent_size, self.hidden_size)
        self.gru1_c = nn.GRUCell(self.latent_size, self.hidden_size)
        self.h2o = nn.Linear(self.hidden_size, self.output_size)
        #if policy_flag:
        #    init.uniform(self.h2o.weight,-3e-3, 3e-3)
        #    init.uniform(self.h2o.bias,-3e-3, 3e-3)
        self.h1_sa = Variable(torch.zeros(batch_size, hidden_size), requires_grad=True).type(dtype)
        self.h1_c = Variable(torch.zeros(batch_size, hidden_size), requires_grad=True).type(dtype)
        self.action_log_std = nn.Parameter(torch.zeros(1, output_size).type(dtype))

        if n_layers == 2:
            self.gru2_sa = nn.GRUCell(self.hidden_size, self.hidden_size)
            self.gru2_c = nn.GRUCell(self.hidden_size, self.hidden_size)
            self.h2_sa = Variable(torch.zeros(batch_size, hidden_size), requires_grad=True).type(dtype)
            self.h2_c = Variable(torch.zeros(batch_size, hidden_size), requires_grad=True).type(dtype)


    def forward(self,x):
        x_sa = x[:, 0:self.input_size-self.latent_size]
        x_c = x[:, self.input_size-self.latent_size:]

        self.h1_sa = self.gru1_sa(x_sa, self.h1_sa)
        self.h1_c = self.gru1_c(x_c, self.h1_c)

        if self.n_layers == 2:
            self.h2_sa = self.gru2_sa(self.h1_sa, self.h2_sa)
            self.h2_c = self.gru2_c(self.h1_c, self.h2_c)
            self.h2 = self.h2_sa + self.h2_c

            if self.policy_flag:
                action = F.log_softmax(self.h2o(self.h2))
            else:
                if self.activation_flag == 1:
                    action = F.tanh(self.h2o(self.h2))
                elif self.activation_flag == 2:
                    action = F.sigmoid(self.h2o(self.h2))
                else:
                    action = self.h2o(self.h2)
        else:
            self.h1 = self.h1_sa + self.h1_c
            if self.policy_flag:
                action = F.log_softmax(self.h2o(self.h1))
            else:
                if self.activation_flag == 1:
                    action = F.tanh(self.h2o(self.h1))
                elif self.activation_flag == 2:
                    action = F.sigmoid(self.h2o(self.h1))
                else:
                    action = self.h2o(self.h1)

        return action

    def reset(self, batch_size=1):
        self.h1_sa = Variable(torch.zeros(batch_size, self.hidden_size), requires_grad=True).type(self.dtype)
        self.h1_c = Variable(torch.zeros(batch_size, self.hidden_size), requires_grad=True).type(self.dtype)
        if self.n_layers == 2:
            self.h2_sa = Variable(torch.zeros(batch_size, self.hidden_size), requires_grad=True).type(self.dtype)
            self.h2_c = Variable(torch.zeros(batch_size, self.hidden_size), requires_grad=True).type(self.dtype)
