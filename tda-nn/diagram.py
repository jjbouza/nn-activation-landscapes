import warnings
import os

from ripser import Rips
import numpy as np

import visualize
# some useful computational geometry tools
from sklearn.neighbors import NearestNeighbors
import sklearn.utils.graph_shortest_path as gp
import scipy.spatial

from utils import *

def compute_diagram(data, rips, id, metric='L2', k=12):
    # compute and return diagram
    data_cpu = data.cpu().detach().numpy()
    samples = data_cpu.reshape(data.shape[0], -1)

    with warnings.catch_warnings():
        # supress ripser warnings
        warnings.simplefilter("ignore")
        if metric == 'L2':
            pd = rips.fit_transform(samples)
        elif metric == 'GG' or metric == 'graph geodesic':
            adjacency_matrix = graph_geodesic_adjacency(samples, k)
            graph_geodesic_dm = graph_geodesic_metric(adjacency_matrix)
            visualize.plot_graph(samples, adjacency_matrix, save=id)
            pd = rips.fit_transform(graph_geodesic_dm, distance_matrix=True)
        elif metric == 'SN' or metric == 'scale normalized':
            normalized_data = scale_normalize(samples)
            pd = rips.fit_transform(samples)
        else:
            error("Error: Unknown metric: ".format(metric))
            quit()

    return pd

def compute_diagram_n(model, data, rips, n, metric, k=12, dirname='./activation_visualizations/'):    
    mod = model.module_.to('cpu')
    xn = mod(data, n)

    if not os.path.exists("{}/network{}/".format(dirname, model.id)):
        os.makedirs("{}/network{}/".format(dirname, model.id))

    return compute_diagram(xn, rips, "{}/network{}/layer{}.png".format(dirname, model.id, n), metric=metric, k=k)

def graph_geodesic_adjacency(data, k=12):
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(data)
    adjacency_matrix = nbrs.kneighbors_graph(data).toarray()
    return adjacency_matrix

def graph_geodesic_metric(adjacency_matrix):
    '''
    Input: Point Cloud - [N, d]
    Output: Graph geodesic distance matrix: [N, N]
    '''
    distance_matrix = gp.graph_shortest_path(adjacency_matrix, directed=False, method='auto')
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
            name = os.path.join(dirname, "layer{}dim{}.csv".format(layer_id, dim_id))
            np.savetxt(name, dim, delimiter=',')

