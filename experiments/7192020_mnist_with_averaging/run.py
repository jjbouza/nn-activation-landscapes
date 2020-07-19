import torch
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import pickle
import dill

from model import Net
from landscape import *

import argparse

def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()


def test(model, device, test_loader, id=0):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('Network {} Status: Test set average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        id, test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def main():
    parser = argparse.ArgumentParser(description='PyTorch landscape computations')
    # Training settings
    parser.add_argument('--iterations', type=int, default=16)
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=2, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')

    # diagram and landscape computation settings
    parser.add_argument('--maxdim', type=int, nargs='+', default=2,
                        help='List of maxdims to compute diagrams and landscapes at for each layer.')  
    parser.add_argument('--threshold', type=float, nargs='+', default=10,
                        help='List of thresholds to compute diagrams at for each layer.')  
    parser.add_argument('--n', type=int, nargs='+', default=1,
                        help='List of which layers to compute landscapes at.')  
    parser.add_argument('--data_samples', type=int, default=1000,
                        help='Number of data samples to pass through network to get per-layer activations.')  
    parser.add_argument('--dx', type=float, default=0.1,
                        help='x-spacing for landscape sampling')  
    parser.add_argument('--min_x', type=float, default=0,
                        help='min x to sample landscape')  
    parser.add_argument('--max_x', type=float, default=10,
                        help='max x to sample landscape')  
    parser.add_argument('--save', default=None,
                        help='Save output landscapes to this file')  

    args = parser.parse_args()
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    kwargs = {'batch_size': args.batch_size, 'shuffle': True}

    # preprocessing for MNIST data.
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))])

    dataset1 = datasets.MNIST('./data', train=True, download=True,
                       transform=transform)
    dataset2 = datasets.MNIST('./data', train=False,
                       transform=transform)

    train_loader = torch.utils.data.DataLoader(dataset1,**kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **kwargs)
    landscape_loader = torch.utils.data.DataLoader(dataset2, batch_size=args.data_samples)
    
    landscapes_per_network = []
    for it in range(args.iterations):
        print('Beginning training of network {}'.format(it))

        model = Net().to(device)
        optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

        scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
        for epoch in range(1, args.epochs + 1):
            train(model, device, train_loader, optimizer, epoch)
            test(model, device, test_loader, it)
            scheduler.step()

        print('Beginning landscape computation for network {}'.format(it))
        data = next(iter(landscape_loader))[0].to(device)
        landscapes_per_network.append(landscapes_diagrams_from_model(model, data, args.maxdim, args.threshold, args.n, args.dx, args.min_x, args.max_x, it)[0])

    # average across networks
    # landscapes_per_network: network x layer x n

    landscape_averages = []
    for layer_it in range(len(landscapes_per_network[0])):
        landscape_averages_layer = []

        for H_degree_it in range(len(landscapes_per_network[0][0])):
            landscape_degree_layer = [landscape[layer_it][H_degree_it] for landscape in landscapes_per_network]
            landscape_averages_layer.append(average(landscape_degree_layer))

        landscape_averages.append(landscape_averages_layer)

    # layer x n x landscape
    if args.save:
        with open(args.save, 'wb') as lfile:
            pickle.dump(landscape_averages, lfile)

if __name__ == '__main__':
    main()
