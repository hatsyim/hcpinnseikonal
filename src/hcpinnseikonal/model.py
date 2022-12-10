import random
import numpy as np
import torch
import torch.nn as nn
from torch.nn import Linear

def activation(act_fun='leakyrelu'):

    if isinstance(act_fun, str):
        if act_fun == 'leakyrelu':
            return nn.LeakyReLU(0.2, inplace='y')
        elif act_fun == 'elu':
            return nn.ELU()
        elif act_fun == 'relu':
            return nn.ReLU()
        elif act_fun == 'tanh':
            return nn.Tanh()
        elif act_fun == 'swish':
            return Swish()
        else:
            raise ValueError(f'{act_fun} is not an activation function...')
    else:
        return act_fun

def layer(lay='linear'):
    if isinstance(lay, str):
        if lay == 'linear':
            return lambda x,y: nn.Linear(x, y)
        
        elif lay == 'adaptive':
            return lambda x,y: AdaptiveLinear(x,y,
                                              adaptive_rate=0.1,
                                              adaptive_rate_scaler=10.)
        else:
            raise ValueError(f'{lay} is not a layer type...')
    else:
        return lay

class ResidualNetwork(torch.nn.Module):
    def __init__(self, num_input, num_output, num_layers=10, num_neurons=10,
                 lay='linear', act='relu', last_act=None, last_multiplier=1):
        
            super(ResidualNetwork, self).__init__()
            
            self.act = activation(act)
            self.last_act = activation(last_act)
            self.last_multiplier = last_multiplier

            # Input Structure
            self.fc0  = Linear(num_input,num_neurons)
            self.fc1  = Linear(num_neurons,num_neurons)

            # Resnet Block
            self.rn_fc1 = torch.nn.ModuleList([Linear(num_neurons, num_neurons) for i in range(num_layers)])
            self.rn_fc2 = torch.nn.ModuleList([Linear(num_neurons, num_neurons) for i in range(num_layers)])
            self.rn_fc3 = torch.nn.ModuleList([Linear(num_neurons, num_neurons) for i in range(num_layers)])

            # Output structure
            self.fc8  = Linear(num_neurons,num_neurons)
            self.fc9  = Linear(num_neurons,num_output)

    def forward(self,x):
        x = self.act(self.fc0(x))
        x = self.act(self.fc1(x))
        for j in range(len(self.rn_fc1)):
            x0 = x
            x  = self.act(self.rn_fc1[j](x))
            x  = self.act(self.rn_fc3[j](x)+self.rn_fc2[j](x0))

        x = self.act(self.fc8(x))
        x = self.fc9(x)
        
        if self.last_act is not None:
            x = self.last_act(x)
        x = x * self.last_multiplier
        return x

class FullyConnectedNetwork(nn.Module):
    
    def __init__(self, num_input, num_output, n_hidden=[16, 32],
                 lay='linear', act='tanh', last_act=None, last_multiplier=1):
        super(FullyConnectedNetwork, self).__init__()
        self.lay = lay
        self.act = act
        self.last_multiplier = last_multiplier
        self.last_act = last_act

        act = activation(act)
        lay = layer(lay)
        if last_act == 'sigmoid':
            self.model = nn.Sequential(
                nn.Sequential(lay(num_input, n_hidden[0]), act),
                *[nn.Sequential(lay(n_hidden[i], n_hidden[i + 1]),act) for i in range(len(n_hidden) - 1)],
                lay(n_hidden[-1], num_output),
                nn.Sigmoid()
            )
        elif last_act == 'relu':
            self.model = nn.Sequential(
                nn.Sequential(lay(num_input, n_hidden[0]), act),
                *[nn.Sequential(lay(n_hidden[i], n_hidden[i + 1]),act) for i in range(len(n_hidden) - 1)],
                lay(n_hidden[-1], num_output),
                nn.ReLU()
            )
        elif last_act == 'tanh':
            self.model = nn.Sequential(
                nn.Sequential(lay(num_input, n_hidden[0]), act),
                *[nn.Sequential(lay(n_hidden[i], n_hidden[i + 1]),act) for i in range(len(n_hidden) - 1)],
                lay(n_hidden[-1], num_output),
                nn.Tanh()
            )
        else:
            self.model = nn.Sequential(
                nn.Sequential(lay(num_input, n_hidden[0]), act),
                *[nn.Sequential(lay(n_hidden[i], n_hidden[i + 1]),act) for i in range(len(n_hidden) - 1)],
                lay(n_hidden[-1], num_output)
            )

    def forward(self, x):
        x = self.model(x) #/ (1-0.9999)
        return x*self.last_multiplier