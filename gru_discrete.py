import math
import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn.init as init


class GRU(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=8, dtype=torch.DoubleTensor, n_layers=2, batch_size=1, policy_flag=1, activation_flag=0):
        super(GRU, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.batch_size = batch_size
        self.policy_flag = policy_flag
        self.activation_flag = activation_flag
        self.dtype = dtype

        self.gru1 = nn.GRUCell(self.input_size, self.hidden_size)
        self.h2o = nn.Linear(self.hidden_size, self.output_size)
        #if policy_flag:
        #    init.uniform(self.h2o.weight,-3e-3, 3e-3)
        #    init.uniform(self.h2o.bias,-3e-3, 3e-3)
        self.h1 = Variable(torch.zeros(batch_size, hidden_size), requires_grad=True).type(dtype)
        self.action_log_std = nn.Parameter(torch.zeros(1, output_size).type(dtype))

        if n_layers == 2:
            self.gru2 = nn.GRUCell(self.hidden_size, self.hidden_size)
            self.h2 = Variable(torch.zeros(batch_size, hidden_size), requires_grad=True).type(dtype)


    def forward(self,x):
        self.h1 = self.gru1(x, self.h1)
        if self.n_layers == 2:
            self.h2 = self.gru2(self.h1, self.h2)
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
        self.h1 = Variable(torch.zeros(batch_size, self.hidden_size), requires_grad=True).type(self.dtype)
        if self.n_layers == 2:
            self.h2 = Variable(torch.zeros(batch_size, self.hidden_size), requires_grad=True).type(self.dtype)
