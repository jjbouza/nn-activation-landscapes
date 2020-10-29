import torch
from skorch.net import NeuralNet

from utils import *

def compute_activations(net, 
                        data, 
                        layers,
                        device='cpu'):
    if isinstance(net, NeuralNet):
        module = net.module_.to(device)
    elif isinstance(net, torch.nn.Module):
        module = net.to(device)
    else:
        error("Error: invalid network type {}".format(net))
    
    activations = []
    for n in layers:
        xn = module(data, n)
        activations.append(xn)

    return activations

