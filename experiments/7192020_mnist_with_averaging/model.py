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
        
        self.layer1 = nn.Sequential(self.conv1, nn.ReLU())
        self.layer2 = nn.Sequential(self.conv2, nn.ReLU())
        self.layer3 = lambda x : torch.flatten(F.max_pool2d(x, 2), 1)
        self.layer4 = nn.Sequential(self.fc1, nn.ReLU())
        self.layer5 = self.fc2

        self.layers = [self.layer1,
                       self.layer2,
                       self.layer3,
                       self.layer4,
                       self.layer5,
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


