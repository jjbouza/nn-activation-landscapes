import argparse
import warnings

import torch
from torchvision import datasets, transforms
import numpy as np

from ripser import Rips
from model import Net

def compute_diagram(data, rips):
    # compute and return diagram
    data_cpu = data.detach().cpu().numpy()
    samples = data_cpu.reshape(data.shape[0], -1)
    with warnings.catch_warnings():
        # supress ripser warnings
        warnings.simplefilter("ignore")
        pd = rips.fit_transform(samples)

    return pd

def compute_diagram_n(model, data, rips, n):    
    xn = model(data, n)
    return compute_diagram(xn, rips)

def compute_diagram_all(model, data, rips):
    activations = model(data)
    diagrams = [compute_diagram(activation, rips) for activation in activations]
    return diagrams


# Command line interface:

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compute persistence diagrams at layers of a trained model.')
    parser.add_argument('model', action='store')
    parser.add_argument('--maxdim', type=int, nargs='+', default=2)  
    parser.add_argument('--threshold', type=float, nargs='+', default=10)  
    parser.add_argument('--n', type=int, nargs='+', default=1)
    parser.add_argument('--data_samples', type=int, default=100)
    parser.add_argument('--save', action='store_true', default=True)

    args = parser.parse_args()

    assert len(args.maxdim) == len(args.threshold) and len(args.threshold) == len(args.n), "maxdim, threshold and n must be same size."
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # initialize a trained model.
    net = Net()
    net.load_state_dict(torch.load(args.model))
    net.eval()
    net = net.to(device)
    print("Model Loaded")

    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])

    test_data = datasets.MNIST('./data', train=False, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=100)

    data = next(iter(test_loader))[0].to(device)
    print("Data loaded")

    # initialize rips diagrams
    pds = []
    for maxdim, threshold, n in zip(args.maxdim, args.threshold, args.n):
        print("Processing layer {} with {} dimensions and threshold of {}".format(n, maxdim, threshold))
        rips = Rips(maxdim=maxdim,
                    thresh=threshold,
                    verbose=False)

        pds.append(compute_diagram_n(net, data, rips, n))

    # save diagrams
    if args.save:
        np.savez("{}_{}_{}".format(n, maxdim, threshold), *pds)
