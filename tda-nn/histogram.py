import numpy as np
from utils import *
from visualize import plot_histogram
import os

def compute_histograms(activations, centers, bins, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    center_list = [None for _ in activations] if centers is None else centers
    for layer, (activation, center) in enumerate(zip(activations, center_list)):
        hist, output_bins = compute_histogram(activation, center, bins)
        plot_histogram(hist, output_bins, save=os.path.join(output_dir, "layer{}.png".format(layer)))

def compute_histogram(activation, center, bins):
    if center:
        activation = activation-np.mean(activation, axis=0)
    norms = np.linalg.norm(activation, axis=1)
    hist, output_bins = np.histogram(norms, bins=bins)
    return hist, output_bins

if __name__=='__main__':
    import argparse

    def load_data(fnames):
        data = []
        for fname in sorted(os.listdir(fnames)):
            path = os.path.join(fnames, fname)
            if os.path.splitext(path)[1] == '.csv':
                data.append(np.loadtxt(path, delimiter=','))
            else:
                error("Error: invalid file extension {}, this script only support CSV datasets.".format(os.path.splitext(fname)[1]))
                quit()

        return data

    parser = argparse.ArgumentParser(description='Computes histogram of norms of activations vectors, with optional centering.')
    parser.add_argument('--activations', type=str)
    parser.add_argument('--layers', type=int, nargs='+')
    parser.add_argument('--center', type=int, nargs='+', default=None)
    parser.add_argument('--bins', type=int, default=50)
    parser.add_argument('--output-dir', type=str)

    args = parser.parse_args()
    data_ = load_data(args.activations)
    data = [data_[layer] for layer in args.layers]
    center = [bool(c) for c in args.center] if args.center is not None else args.center
    
    compute_histograms(data, center, args.bins, args.output_dir)