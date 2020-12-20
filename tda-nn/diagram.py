import warnings
import os

from ripser import ripser
import numpy as np
import torch

import visualize
# some useful computational geometry tools
from sklearn.neighbors import NearestNeighbors
import sklearn.utils.graph_shortest_path as gp
import scipy.spatial

from utils import *

def compute_diagrams(data, maxdims, thresholds, metric='L2', k=12, save_GG_activations_plots=None):
    if save_GG_activations_plots is not None:
        if not os.path.exists(save_GG_activations_plots):
            os.makedirs(save_GG_activations_plots)

    diagrams = []
    for id, (activation, dim, threshold) in enumerate(zip(data, maxdims, thresholds)):
        act_plot = None if save_GG_activations_plots is None else os.path.join(save_GG_activations_plots, 'layer{}.png'.format(id))
        diag = compute_diagram(activation, 
                               dim, 
                               threshold, 
                               metric=metric, 
                               k=k,
                               save_GG_activations_plots=act_plot)
        diagrams.append(diag)

    return diagrams

def compute_diagram(data, maxdim, threshold, metric='L2', k=12, save_GG_activations_plots=None):
    # compute and return diagram
    if isinstance(data, torch.Tensor):
        data_cpu = data.cpu().detach().numpy()
    elif isinstance(data, np.ndarray):
        data_cpu = data
    else:
        error("Unsupported data type: {} for compute_diagram".format(type(data)))
        quit()

    X = data_cpu.reshape(data.shape[0], -1)

    with warnings.catch_warnings():
        # supress ripser warnings
        warnings.simplefilter("ignore")
        if metric == 'L2':
            pd = ripser(X, maxdim, threshold)['dgms']
        elif metric == 'GG' or metric == 'graph geodesic':
            adjacency_matrix = graph_geodesic_adjacency(X, k)
            graph_geodesic_dm = graph_geodesic_metric(adjacency_matrix)
            if save_GG_activations_plots is not None:
                visualize.plot_graph(X, adjacency_matrix, save=save_GG_activations_plots)

            pd = ripser(graph_geodesic_dm, maxdim, threshold, distance_matrix=True)['dgms']
        elif metric == 'SN' or metric == 'scale normalized':
            normalized_X = scale_normalize(X)
            pd = ripser(X, maxdim, threshold)['dgms']
        else:
            error("Error: Unknown metric: ".format(metric))
            quit()

    return pd

def graph_geodesic_adjacency(data, k=12):
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(data)
    adjacency_matrix = nbrs.kneighbors_graph(data).toarray()
    return adjacency_matrix

def graph_geodesic_metric(adjacency_matrix):
    '''
    Input: Adjacency matrix: [N,N]
    Output: Graph geodesic distance matrix: [N, N]
    '''
    distance_matrix = gp.graph_shortest_path(adjacency_matrix, directed=False, method='auto')
    distance_matrix[distance_matrix==0] = np.inf
    np.fill_diagonal(distance_matrix, 0)

    return distance_matrix

def scale_normalize(data, p=2):
    distance_matrix = scipy.spatial.distance_matrix(data, data, p)
    mean_distance = np.mean(distance_matrix)
    normalized_data = data/mean_distance
    return normalized_data

def save_diagram(diagram, dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    for layer_id, layer in enumerate(diagram):
        for dim_id, dim in enumerate(layer):
            layer_id_zfill = int2str_with_leading_zero(layer_id, len(diagram))
            dim_id_zfill = int2str_with_leading_zero(dim_id, len(layer))
            name = os.path.join(dirname, "layer{}dim{}.csv".format(layer_id_zfill, dim_id_zfill))
            np.savetxt(name, dim, delimiter=',')

if __name__ == '__main__':
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

    parser = argparse.ArgumentParser(description='Compute diagrams given a network and data.')
    parser.add_argument('--activations', type=str)
    parser.add_argument('--max-diagram-dimension', type=int, nargs='+')
    parser.add_argument('--diagram-threshold', type=float, nargs='+')
    parser.add_argument('--persistence-layers', type=int, nargs='+')
    parser.add_argument('--diagram-metric', type=str)
    parser.add_argument('--nn-graph-k', type=int)
    parser.add_argument('--save-gg-diagram-plots', default=None)
    parser.add_argument('--output-dir', type=str)

    args = parser.parse_args()
    
    # take first persistence-data-samples rows from activations
    data_ = load_data(args.activations)
    data = [data_[layer] for layer in args.persistence_layers]
    diagrams = compute_diagrams(data, 
                                args.max_diagram_dimension, 
                                args.diagram_threshold, 
                                args.diagram_metric, 
                                args.nn_graph_k, 
                                args.save_gg_diagram_plots)
    save_diagram(diagrams, args.output_dir)
