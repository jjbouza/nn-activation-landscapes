import torch
import torch.nn as nn
import torch.nn.functional as F

from ripser import Rips

class Net(nn.Module):
    """
    MLP for simple classification task.
    """
    def __init__(self):
        super(Net, self).__init__()
        self.fc_first = nn.Linear(2, 15)
        self.middle_fc = [nn.Linear(15, 15) for _ in range(9)]
        self.fc_last = nn.Linear(15, 2)

        self.fc_layers = [self.fc_first]+self.middle_fc
        self.layers = nn.ModuleList([nn.Sequential(fc_layer, nn.Tanh()) for fc_layer in
                                     self.fc_layers]+[self.fc_last])


    def forward(self, x, n=None):
        """
        Net.forward(x, n) means return the nth layer activation of Net given input x.
        If instead Net.forward(x) is called then a list with all of the hidden layer 
        activations is returned.

        NOTE: TO WORK WITH TDA-NN THE FORWARD FUNCTION OF YOUR MODEL MUST ACCEPT THE 'n' PARAMATER.
        """
        if n == 'all':
            activation_list = [x]
            for f in self.layers:
                activation_list.append(f(activation_list[-1]))
            return activation_list

        if n == None:
            n = len(self.layers)

        for f in self.layers[:n]:
            x = f(x)

        return x

