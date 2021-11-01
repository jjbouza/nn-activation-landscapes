import warnings
import os

#from gph.python import ripser_parallel
from ripser import ripser
import gudhi

import numpy as np
import torch

import visualize
# some useful computational geometry tools
from sklearn.neighbors import NearestNeighbors
import sklearn.utils.graph_shortest_path as gp
import scipy.spatial
import resource

from utils import *

def compute_diagrams(data, maxdims, thresholds, metric='L2', k=12, percentiles=None, centers=None, save_activations_plots=None):
    if save_activations_plots is not None:
        if not os.path.exists(save_activations_plots):
            os.makedirs(save_activations_plots)

    center_list = [None for _ in maxdims] if centers is None else centers
    percentile_list = [None for _ in maxdims] if percentiles is None else percentiles

    diagrams = []
    for id, (activation, dim, threshold, center, percentile) in enumerate(zip(data, maxdims, thresholds, center_list, percentile_list)):
        act_plot = None if save_activations_plots is None else os.path.join(save_activations_plots, 'layer{}.png'.format(id))
        diag = compute_diagram(activation, 
                               dim, 
                               threshold, 
                               metric=metric, 
                               k=k,
                               percentile=percentile,
                               center=center,
                               save_activations_plots=act_plot)
        diagrams.append(diag)

    return diagrams

def compute_diagram(data, maxdim, threshold, metric='L2', k=12, percentile=0.9, center=None, save_activations_plots=None):
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
            pd = ripser(X, maxdim=maxdim, thresh=threshold, n_threads=-1)['dgms']
            if save_activations_plots is not None:
                visualize.plot_activations(X, None, save=save_activations_plots)
        elif metric == 'GG' or metric == 'graph geodesic':
            adjacency_matrix = graph_geodesic_adjacency(X, k)
            graph_geodesic_dm = graph_geodesic_metric(adjacency_matrix)
            if save_activations_plots is not None:
                visualize.plot_activations(X, adjacency_matrix, save=save_activations_plots)

            pd = ripser(graph_geodesic_dm, maxdim=maxdim, thresh=threshold, metric='precomputed', n_threads=-1)['dgms']
        elif metric == 'SN' or metric == 'scale normalized':
            normalized_X = scale_normalize(X)
            pd = ripser(normalized_X, maxdim=maxdim, thresh=threshold, n_threads=-1)['dgms']
            if save_activations_plots is not None:
                visualize.plot_activations(normalized_X, None, save=save_activations_plots)

        elif metric == 'MN' or metric == 'max normalized':
            normalized_X = scale_normalize(X, max=True)
            pd = ripser(normalized_X, maxdim=maxdim, thresh=threshold, n_threads=-1)['dgms']
            if save_activations_plots is not None:
                visualize.plot_activations(normalized_X, None, save=save_activations_plots)
        
        elif metric == 'PN' or metric == 'percentile normalized':
            normalized_data, percentile_index= percentile_normalize(X, percentile, center)
            distance_matrix = scipy.spatial.distance_matrix(normalized_data, normalized_data)
            # Because we sorted the data, we just need to set the block beyond row and col = percentile_index to 0.
            if percentile_index < distance_matrix.shape[0]:
                distance_matrix[percentile_index:, percentile_index:] = 0.0

            pd = ripser(distance_matrix, maxdim=maxdim, thresh=threshold, distance_matrix=True)['dgms']
            if save_activations_plots is not None:
                visualize.plot_activations(X, None, save=save_activations_plots)

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

def scale_normalize(data, p=2, max=False):
    translated_data = data-np.mean(data, axis=0)
    distance_matrix = scipy.spatial.distance_matrix(translated_data, translated_data, p)
    
    if max:
        scale = np.max(distance_matrix)
    else:
        scale = np.mean(distance_matrix)

    if scale == 0:
        normalized_data = translated_data
    else:
        normalized_data = translated_data/scale

    return normalized_data

def percentile_normalize(data, percentile, center=None, p=2):
    if center:
        data = data-np.mean(data, axis=0)

    distances = np.linalg.norm(data, ord=p, axis=1)

    # reorder data from closest to farthest
    argsort = np.argsort(distances)
    data = data[argsort]
    distances = distances[argsort]

    percentile_index = int(distances.shape[0]*percentile) # index in translated_data where "outlier" region begins. 
    percentile_distance = distances[min(percentile_index, distances.shape[0]-1)]
    normalized_data = data/percentile_distance
    normalized_data[percentile_index:] *= (percentile_distance/distances[percentile_index:])[:,None]

    distances_new = np.linalg.norm(normalized_data, ord=p, axis=1)

    return normalized_data, percentile_index

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
    parser.add_argument('--center', type=int, nargs='+', default=None)
    parser.add_argument('--percentile', type=float, nargs='+')
    parser.add_argument('--save-diagram-plots', default=None)
    parser.add_argument('--output-dir', type=str)

    args = parser.parse_args()
    
    # take first persistence-data-samples rows from activations
    data_ = load_data(args.activations)
    data = [data_[layer] for layer in args.persistence_layers]
    center = [bool(c) for c in args.center] if args.center is not None else args.center
    diagrams = compute_diagrams(data, 
                                args.max_diagram_dimension, 
                                args.diagram_threshold, 
                                args.diagram_metric, 
                                args.nn_graph_k, 
                                args.percentile,
                                center,
                                args.save_diagram_plots)
    save_diagram(diagrams, args.output_dir)
