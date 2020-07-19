import numpy as np
from diagram import compute_diagram_n
from ripser import Rips

import rpy2
from rpy2.robjects.packages import importr
from rpy2.robjects import numpy2ri
numpy2ri.activate()

def landscape(diagram, dx=0.1, min_x= 0, max_x=10, threshold=-1):
    """
    Simple interface to tda-tools for DISCRETE landscapes.
    """

    return np.array(landscape.tdatools.landscape_discrete(diagram, dx, min_x, max_x, threshold))

landscape.tdatools = importr('tdatools')

def landscapes_diagrams_from_model(net, data, maxdims, thresholds, ns, dx, min_x, max_x, id=None):
    landscapes = []
    diagrams = []
    for maxdim, threshold, n in zip(maxdims, thresholds, ns):
        if id is None:
            print("Processing layer {} with {} dimensions and threshold of {}".format(n, maxdim, threshold))
        else:
            print("Network {} Status: Processing layer {} with {} dimensions and threshold of {}".format(id, n, maxdim, threshold))

        rips = Rips(maxdim=maxdim,
                    thresh=threshold,
                    verbose=False)

        diagrams_all = compute_diagram_n(net, data, rips, n)
        diagrams.append(diagrams_all)
        landscapes_layer = [landscape(diag, dx, min_x, max_x) for diag in diagrams_all]
        landscapes.append(landscapes_layer)

    return landscapes, diagrams

def average(landscapes):
    """
    landscapes: List[ndarray[levels, samples, 2]]
    samples dimension must have same size for all landscapes.

    * NOTE: if we have overflow issues here at some point we could do the average differently (divide at each step)
    """
    max_levels = max([landscape.shape[0] for landscape in landscapes])
    samples = landscapes[0].shape[1]
    mean = np.zeros([max_levels, samples, 2])

    for landscape in landscapes:
        # zero pad to match dimensions. Equivalent to saying that non-present levels
        # are constant zero functions.
        mean += np.pad(landscape, ((0, max_levels-landscape.shape[0]), (0,0), (0,0)) )

    return mean/len(landscapes)