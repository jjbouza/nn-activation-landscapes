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
        self.fc1 = nn.Linear(2, 8)
        self.fc2 = nn.Linear(8, 8)
        self.fc3 = nn.Linear(8, 8)
        self.fc4 = nn.Linear(8, 2)

        self.layer1 = nn.Sequential(self.fc1, nn.ReLU())
        self.layer2 = nn.Sequential(self.fc2, nn.ReLU())
        self.layer3 = nn.Sequential(self.fc3, nn.ReLU())
        self.layer4 = self.fc4

        self.layers = [self.layer1,
                       self.layer2,
                       self.layer3,
                       self.layer4]


    def forward(self, x, n=None):
        """
        Net.forward(x, n) means return the nth layer activation of Net given input x.
        If instead Net.forward(x) is called then a list with all of the hidden layer 
        activations is returned.

        NOTE: TO WORK WITH TDA-NN THE FORWARD FUNCTION OF YOUR MODEL MUST ACCEPT THE 'n' PARAMATER.
        """
        if n == 'all':
            activation_list = [x]
            for f in self.layers[:n]:
                activation_list.append(f(activation_list[-1]))
            return activation_list

        if n == None:
            n = len(self.layers)

        for f in self.layers[:n]:
            x = f(x)

        return x

