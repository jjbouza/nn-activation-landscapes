import numpy as np
from utils import *

import math
import os
import re

#from gudhi.representations.vector_methods import Landscape

#def landscape(diagram, dx=0.1, min_x= 0, max_x=10, threshold=-1):
#    """
#    Simple interface to tda-tools for DISCRETE landscapes.
#    """
#    landscape = Landscape(num_landscapes=diagram.shape[0], resolution=1//dx)
#
#    if diagram.shape[0] == 0:
#        warning("WARNING: Empty diagram detected")
#        return np.zeros([1, math.floor((max_x-min_x)/dx)+1, 2])
#    return landscape(diagram)

# import Rpy modules and ignore warnings...
import warnings 
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import rpy2
    from rpy2.robjects.packages import importr
    from rpy2.robjects import numpy2ri
    numpy2ri.activate()

def compute_landscapes(diagrams, dx=0.1, min_x=0, max_x=10, thresholds=None):
    def one_x_axis(landscape):
        x_axis = landscape[0,:,0]
        y_axis = landscape[:,:,1]
        return (x_axis, y_axis)

    if thresholds is None:
        thresholds = [-1 for _ in diagrams]
    
    landscapes = []
    for diagram, threshold in zip(diagrams, thresholds):
        landscape_layer = []
        for diagram_dim in diagram:
            landscape = compute_landscape(diagram_dim, dx, min_x, max_x, threshold)
            landscape_layer.append(one_x_axis(landscape))
        landscapes.append(landscape_layer)

    return landscapes

def compute_landscape(diagram, dx=0.1, min_x= 0, max_x=10, threshold=-1):
    """
    Simple interface to tda-tools for DISCRETE landscapes.
    """
    if diagram.shape[0] == 0:
        warning("WARNING: Empty diagram detected")
        return np.zeros([1, math.floor((max_x-min_x)/dx)+1, 2])
    return np.array(compute_landscape.tdatools.landscape_discrete(diagram, dx, min_x, max_x, threshold))
compute_landscape.tdatools = importr('tdatools')

def average_across_networks(landscapes_per_network):
    """
    Inputs: 
    lanscapes_per_network: networks x layers x degree x landscape_array

    Outputs:
    landscape_averages: layers x degree x landscape_array
    """
    landscape_averages = []
    for layer_it in range(len(landscapes_per_network[0])):
        landscape_averages_layer = []

        for H_degree_it in range(len(landscapes_per_network[0][0])):
            landscape_degree_layer = [landscape[layer_it][H_degree_it] for landscape in landscapes_per_network]
            landscape_averages_layer.append(average(landscape_degree_layer))

        landscape_averages.append(landscape_averages_layer)
    
    return landscape_averages

def average(landscapes):
    """
    Inputs: 
    landscapes: List[ndarray[levels, samples, 2]]

    Outputs:
    mean: average landscape array

    Description:
    samples dimension must have same size for all landscapes.

    * NOTE: if we have overflow issues here at some point we could do the average differently (divide at each step)
    """
    max_levels = max([landscape[1].shape[0] for landscape in landscapes])
    samples = landscapes[0][0].shape[0]
    mean = np.zeros([max_levels, samples])

    for landscape in landscapes:
        # zero pad to match dimensions. Equivalent to saying that non-present levels
        # are constant zero functions.
        mean += np.pad(landscape[1], ((0, max_levels-landscape[1].shape[0]), (0,0)) )

    return landscapes[0][0], mean/len(landscapes)

def load_landscape(dirname):
    max_layer, max_dim = 0, 0
    for landscape_fname in os.listdir(dirname):
        matches = re.findall("\d+", landscape_fname)
        layer, dim = int(matches[0]), int(matches[1])
        if max_layer < layer:
            max_layer = layer
        if max_dim < dim:
            max_dim = dim

    landscapes = [[None for _ in range(max_dim+1)] for _ in range(max_layer+1)]
    for landscape_fname in os.listdir(dirname):
        matches = re.findall("\d+", landscape_fname)
        layer, dim = int(matches[0]), int(matches[1])
        landscape = np.loadtxt(os.path.join(dirname, landscape_fname), delimiter=',')
        landscapes[layer][dim] = landscape

    return landscapes

def average_from_disk(dirname):
    mean = np.zeroes([0,0])
    for network_dir in os.listdir(dirname):
        network_landscapes = load_landscape(os.path.join(dirname, network_dir))
        max_levels = max([l.shape[0] for z in network_landscapes for l in z])


def save_landscape(landscape, dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    for layer_id, layer in enumerate(landscape):
        for dim_id, dim in enumerate(layer):
            name = os.path.join(dirname, "layer{}dim{}.csv".format(layer_id, dim_id))
            np.savetxt(name, dim[1], delimiter=',')

if __name__=='__main__':
    import argparse
    import warnings
    import re
    warnings.simplefilter("ignore", UserWarning)

    def load_data(fnames):
        data = {}
        max_layers, max_dims = 0,0
        for fname in os.listdir(fnames):
            fname_meta = re.findall("[0-9]+", fname)
            layer, dim = int(fname_meta[0]), int(fname_meta[1])
            if layer > max_layers:
                max_layers = layer
            if dim > max_dims:
                max_dims = dim
            path = os.path.join(fnames, fname)
            if os.path.splitext(path)[1] == '.csv':
                network_data = np.loadtxt(path, delimiter=',')
                if len(network_data.shape) == 1:
                    data[(layer, dim)] = np.zeros([0,2])
                else:
                    data[(layer, dim)] = network_data
            else:
                error("Error: invalid file extension {}, this script only support CSV datasets.".format(os.path.splitext(fname)[1]))
                quit()
        
        data_final = [[None for j in range(max_dims)] for i in range(max_layers)]
        for i in range(max_layers):
            for j in range(max_dims):
                data_final[i][j] = data[(i,j)]
        
        return data_final

    parser = argparse.ArgumentParser(description='Compute landscapes given diagrams.')
    parser.add_argument('--diagrams', type=str)
    parser.add_argument('--landscape-dx', type=float)
    parser.add_argument('--landscape-min-x', type=float)
    parser.add_argument('--landscape-max-x', type=float)
    parser.add_argument('--output-dir', type=str)

    args = parser.parse_args()
   

    diagrams = load_data(args.diagrams)
    landscapes = compute_landscapes(diagrams, args.landscape_dx, args.landscape_min_x, args.landscape_max_x)
    save_landscape(landscapes, args.output_dir)

