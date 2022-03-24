import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Sequential(nn.Linear(784, 128),
                      nn.ReLU())
        self.fc2 = nn.Sequential(nn.Linear(128, 64),
                      nn.ReLU())
        self.fc_middle = [nn.Sequential(nn.Linear(64, 64),
                          nn.ReLU()) for _ in range(4)]
        self.fc_last = nn.Linear(64, 10)
        self.softmax = nn.Softmax()

        self.layers = nn.ModuleList([self.fc1, self.fc2]+self.fc_middle+[self.fc_last])

    def forward(self, x, n=None):
        x = x.view(x.shape[0], -1)

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
