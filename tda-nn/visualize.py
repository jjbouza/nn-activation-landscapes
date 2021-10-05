import numpy as np
import sys
import os
import matplotlib as mpl
mpl.use('Agg')
from matplotlib import pyplot as plt
from persim import visuals, plot_diagrams

from sklearn.decomposition import PCA
from utils import int2str_with_leading_zero, warning
import scipy.spatial

def indices(arr):
    if len(arr.shape) == 0:
        return 0,0
    first_nz = arr.shape[0]
    last_nz = 0

    for en in range(arr.shape[0]):
        if arr[en] > 1e-5:
            last_nz = en

            if first_nz > en:
                first_nz = en

    return first_nz, last_nz

def save_diagram_plots(diagrams, dirname, include=None):
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    fig, ax = plt.subplots()  # Create a figure and an axes.
    for layer, diagram in enumerate(diagrams):
        plot_only = [d for d in diagram if d.shape[0] != 0]
        if any([d.shape[0] == 0 for d in diagram]):
            warning("Not plotting some diagrams with 0 entries.")
        if plot_only != []:
            plot_diagrams(plot_only, show=False, ax=ax, plot_only=include)
            layer_id = int2str_with_leading_zero(layer, len(plot_only))
            fig.savefig(os.path.join(dirname, 'layer{}.png'.format(layer)))
            plt.cla()
    plt.close('all')

def save_landscape_plots(landscapes, dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    fig, ax = plt.subplots()  # Create a figure and an axes.
    for layer, landscape_ in enumerate(landscapes):
        for homology, landscape in enumerate(landscape_):
            plot_landscape(landscape, np.linspace(0, 1, num=landscape.shape[1]), ax=ax)
            layer_id = int2str_with_leading_zero(layer, len(landscapes))
            homology_id = int2str_with_leading_zero(homology, len(landscape_))
            fig.savefig(os.path.join(dirname, 'layer{}dim{}'.format(layer_id, homology_id)))
            plt.cla()
    plt.close('all')

def plot_landscape(landscapes, x_axis, ax):
    # landscapes: np.array[levels, points, 1]
    #starts = []
    #ends = []
    #for level in landscapes:
    #    start, end = indices(level)
    #    starts.append(start)
    #    ends.append(end)

    #start = min(starts)
    #end = max(ends)+2

    #for level in landscapes:
    #    ax.plot(x_axis[:start+end], level[:start+end])
    for level in landscapes:
        ax.plot(level)

def plot_activations(data, adjacency_matrix, save=None):
    # run PCA on data
    plt.clf()
    pca = PCA(n_components=2)
    new_data = pca.fit_transform(data)
    distance_matrix = scipy.spatial.distance_matrix(new_data, new_data, 2)

    plt.scatter(new_data[:,0], new_data[:,1])
    if adjacency_matrix is not None:
        for i in range(adjacency_matrix.shape[0]):
            for j in range(adjacency_matrix.shape[1]):
                if adjacency_matrix[i, j] == 1:
                    plt.plot((new_data[i, 0], new_data[j, 0]), (new_data[i, 1], new_data[j, 1]))

    if save is None:
        plt.show()
    else:
        plt.savefig(save)

def plot_histogram(histogram, bins, save=None):
    plt.clf()
    width = 0.7 * (bins[1] - bins[0])
    center = (bins[:-1] + bins[1:]) / 2
    plt.bar(center, histogram, align='center', width=width)
    if save is None:
        plt.show()
    else:
        plt.savefig(save)