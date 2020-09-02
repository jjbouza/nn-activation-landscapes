import warnings
from ripser import Rips
import numpy as np

# some useful computational geometry tools
from sklearn.neighbors import NearestNeighbors
import sklearn.utils.graph_shortest_path as gp
import scipy.spatial


def compute_diagram(data, rips, metric='L2', k=10):
    # compute and return diagram
    data_cpu = data.detach().cpu().numpy()
    samples = data_cpu.reshape(data.shape[0], -1)

    with warnings.catch_warnings():
        # supress ripser warnings
        warnings.simplefilter("ignore")
        if metric == 'L2':
            pd = rips.fit_transform(samples)
        elif metric == 'GG' or metric == 'graph geodesic':
            graph_geodesic_dm = graph_geodesic_metric(samples, k)
            pd = rips.fit_transform(graph_geodesic_dm, distance_matrix=True)
        elif metric == 'SN' or metric == 'scale normalized':
            normalized_data = scale_normalize(samples)
            pd = rips.fit_transform(samples)
        else:
            print("Error: Unknown metric: ".format(metric))
            quit()

    return pd

def compute_diagram_n(model, data, rips, n, metric):    
    xn = model(data, n)
    return compute_diagram(xn, rips, metric=metric)

def compute_diagram_all(model, data, rips):
    activations = model(data)
    diagrams = [compute_diagram(activation, rips) for activation in activations]
    return diagrams

def graph_geodesic_metric(data, k=5):
    '''
    Input: Point Cloud - [N, d]
    Output: Graph geodesic distance matrix: [N, N]
    '''
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(data)
    adjacency_matrix = nbrs.kneighbors_graph(data).toarray()
    distance_matrix = gp.graph_shortest_path(adjacency_matrix, method='auto', directed=False)
    print(distance_matrix)
    return distance_matrix


def scale_normalize(data, p=2):
    distance_matrix = scipy.spatial.distance_matrix(data, data, p)
    mean_distance = np.mean(distance_matrix)
    normalized_data = data/mean_distance
    return normalized_data




