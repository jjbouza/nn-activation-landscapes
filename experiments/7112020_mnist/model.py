import torch
import torch.nn as nn
import torch.nn.functional as F

from ripser import Rips

class Net(nn.Module):
    """
    Simple classifier.
    """
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

        self.layers = [self.conv1,
                       F.relu,
                       self.conv2,
                       F.relu,
                       lambda x: F.max_pool2d(x, 2),
                       lambda x: torch.flatten(x, 1),
                       self.fc1,
                       F.relu,
                       self.fc2,
                       lambda x: F.log_softmax(x, dim=1)]

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


