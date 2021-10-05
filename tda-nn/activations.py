import torch
from skorch.net import NeuralNet
from scipy.special import softmax

from utils import *
import numpy as np
import os
import sys

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
    module.eval()
    with torch.no_grad():
        for n in layers:
            xn = module(data, n)
            activations.append(xn)
    return activations

def save_activations(activations, dname):
    os.makedirs(dname, exist_ok=True)
    for i, to_save in enumerate(activations):
        activation_id = int2str_with_leading_zero(i, len(activations))
        np.savetxt(os.path.join(dname, "layer{}.csv".format(activation_id)), 
                to_save.detach().cpu().numpy(),
                delimiter=',', fmt='%g')

if __name__=='__main__':
    import argparse


    def load_data(fname):
        if os.path.splitext(fname)[1] == '.npy':
            data_numpy = np.load(fname)
            np.random.shuffle(data_numpy)
            data = torch.from_numpy(data_numpy).float()
        elif os.path.splitext(fname)[1] == '.csv':
            data_numpy = np.loadtxt(fname, delimiter=',')
            np.random.shuffle(data_numpy)
            data = torch.from_numpy(data_numpy).float()
        else:
            error("Error: invalid file extension {}, this script only support numpy datasets.".format(os.path.splitext(fname)[1]))
            quit()

        return data

    parser = argparse.ArgumentParser(description='Compute activations given a network and data.')
    parser.add_argument('--network', type=str)
    parser.add_argument('--input_data', type=str)
    parser.add_argument('--sample-count', type=int)
    parser.add_argument('--persistence-class', type=int)
    parser.add_argument('--keep_class', action='store_true')
    parser.add_argument('--layers', type=int, nargs='+')
    parser.add_argument('--do_softmax', action='store_true')
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--output_dir', type=str)
    parser.add_argument('--models_dir', type=str)

    args = parser.parse_args()

    # load network and data
    sys.path.append(args.models_dir)
    model = torch.load(args.network, map_location=torch.device(args.device)).to(args.device)
    data = load_data(args.input_data).to(args.device)
    # data preprocessing
    if args.persistence_class != -1:
        class_data = data[data[:,-1]==args.persistence_class]
    else:
        shuffle_idx = torch.randperm(data.shape[0])
        class_data = data[shuffle_idx]

    final_data = class_data[:args.sample_count, :-1]
    classes = class_data[:args.sample_count, -1].unsqueeze(-1)

    # save activations
    activations = compute_activations(model, final_data, args.layers, args.device)
    if args.do_softmax:
        activations.append(softmax(activations[-1], axis=1))
        
    if args.keep_class:
        activations = [torch.cat([activation, classes], axis=-1) for activation in activations]

    save_activations(activations, args.output_dir)
