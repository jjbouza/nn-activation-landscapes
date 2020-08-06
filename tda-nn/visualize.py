import numpy as np
import sys
from matplotlib import pyplot as plt
from persim import visuals

def indices(arr):
    first_nz = len(arr)
    last_nz = 0

    for en in range(len(arr)):
        if arr[en] > 1e-5:
            last_nz = en

            if first_nz > en:
                first_nz = en

    return first_nz, last_nz


def plot_landscape(ax, x_axis, landscapes):
    # landscapes: np.array[levels, points, 2]
    starts = []
    ends = []
    for level in landscapes:
        start, end = indices(level)
        starts.append(start)
        ends.append(end)

    start = min(starts)
    end = max(ends)+2
    

    for level in landscapes:
        ax.plot(x_axis[:start+end], level[:start+end])


