from diagram import *
from landscape import *

import argparse
import warnings
import dill
import pickle

import torch
from torchvision import datasets, transforms
import numpy as np

from ripser import Rips


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Compute persistence landscapes at layers of a trained model.')
    parser.add_argument('model', action='store')
    parser.add_argument('--maxdim', type=int, nargs='+', default=2)  
    parser.add_argument('--threshold', type=float, nargs='+', default=10)  
    parser.add_argument('--n', type=int, nargs='+', default=1)
    parser.add_argument('--data_samples', type=int, default=1000)
    parser.add_argument('--dx', type=float, default=0.1)
    parser.add_argument('--min_x', type=float, default=0)
    parser.add_argument('--max_x', type=float, default=10)
    parser.add_argument('--save', default=None)

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    args = parser.parse_args()


    net = torch.load(args.model, pickle_module=dill)
    net.eval()
    net = net.to(device)
    print("Model Loaded")

    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])


    test_data = datasets.MNIST('./data', train=False, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.data_samples)

    data = next(iter(test_loader))[0].to(device)
    print("Data loaded")

    landscapes, diagrams = landscapes_diagrams_from_model(net, data, args.maxdim, args.threshold, args.n, args.dx, args.min_x, args.max_x)

    # save landscapes
    if args.save:
        with open(args.save, 'wb') as lfile:
            pickle.dump({'landscapes': landscapes, 'diagrams': diagrams}, lfile)
