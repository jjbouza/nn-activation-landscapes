import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import pickle
import dill
import tqdm
import os

from csv_loader import CSVDataset, extract_class
from model import Net
from landscape import *

import argparse
import math


def save_activations(model, device, data_loader, dname):
    os.makedirs(dname, exist_ok=True)
    model.eval()
    with torch.no_grad():
        for i in range(len(model.layers)+1):
            layer = []

            for data, target in data_loader:
                data, target = data.to(device), target.to(device)
                outputi = model(data, i)
                layer.append(torch.cat([target[:, None], outputi.view(outputi.shape[0], -1)], dim=1))
            np.savetxt(os.path.join(dname, "layer{}.csv".format(i)), 
                    torch.cat(layer, dim=0).detach().cpu().numpy(),
                    delimiter=',')


def main():
    parser = argparse.ArgumentParser(description='PyTorch landscape computations')
    # Training settings
    parser.add_argument('--iterations', type=int, default=16)
    parser.add_argument('--csv_file', type=str, default='disk6.csv')
    parser.add_argument('--training_threshold', type=float, nargs='+', default=math.inf,
                        help='Training accuracy threshold (stop training at this accuracy).')
    parser.add_argument('--batch-size', type=int, default=1024, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=2, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=0.02, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')

    # diagram and landscape computation settings
    parser.add_argument('--landscape_class', type=int, default=0,
                        help='Class of samples to run landscape evaluation on.')
    parser.add_argument('--pd_metric', type=str, default='L2',
                        help='Persistence Diagram metric. Options: L2, GG (graph geodesic), SN (scale normalized L2)')
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
    parser.add_argument('--save_csv', default=True,
                        help='Save output csv files.')  

    args = parser.parse_args()
    torch.manual_seed(args.seed)

    kwargs = {'batch_size': args.batch_size, 'shuffle': True}
    dataset = CSVDataset(args.csv_file)
    class_dataset = extract_class(dataset, args.landscape_class)

    train_loader = torch.utils.data.DataLoader(dataset,**kwargs)
    landscape_loader = torch.utils.data.DataLoader(class_dataset, batch_size=args.data_samples, shuffle=True)
    
    landscapes_per_network = []

    # Train the networks:
    for it in range(args.iterations):
        print('Beginning training of network {}'.format(it))
        print('Running on device: {}'.format(device))

        model = Net().to(device)
        optimizer = optim.Adam(model.parameters(), lr=args.lr)

        training_threshold = args.training_threshold[-1] if it >= len(args.training_threshold) else args.training_threshold[it]
        epochs = tqdm.tqdm(range(1, args.epochs+1))
        
        for epoch in epochs:
            acc = train(model, device, train_loader, evaluation_loader, optimizer, epoch, it, training_threshold)
            epochs.set_description('Evaluation Accuracy: {}'.format(acc))

            if acc >= training_threshold:
                print("Network {} Status: Early terminated after passing training threshold of {} with {}".format(it, training_threshold, acc))
                break

        
        save_activations(model, device, evaluation_loader, "./activations/network{}/".format(it))

        print('Beginning landscape computation for network {}'.format(it))
        data = next(iter(landscape_loader))[0].to(device)
        landscapes_per_network.append(landscapes_diagrams_from_model(model, 
                                    data, 
                                    args.maxdim, 
                                    args.threshold, 
                                    args.n, 
                                    args.dx, 
                                    args.min_x, 
                                    args.max_x, 
                                    it, 
                                    mode='efficient',
                                    pd_metric=args.pd_metric)[0])

    # average across networks
    # landscapes_per_network: network x layer x dims
    # out: layer x dims x landscape
    def save_landscape(landscape, dirname):
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        for layer_id, layer in enumerate(landscape):
            for dim_id, dim in enumerate(layer):
                name = os.path.join(dirname, "layer{}dim{}.csv".format(layer_id, dim_id))
                np.savetxt(name, dim[1], delimiter=',')

    landscape_averages = average_across_networks(landscapes_per_network)
    if args.save_csv:
        for landscape_id, landscape in enumerate(landscapes_per_network):
            save_landscape(landscape, './landscapes_csv/network{}/'.format(landscape_id))

        save_landscape(landscape_averages, './landscapes_csv/average/')

    if args.save:
        with open(args.save, 'wb') as lfile:
            pickle.dump(landscape_averages, lfile)

if __name__ == '__main__':
    main()
