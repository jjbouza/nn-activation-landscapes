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


def plot_landscape(landscapes):
    # landscapes: np.array[levels, points, 2]
    starts = []
    ends = []
    for level in landscapes:
        start, end = indices(level[:,1])
        starts.append(start)
        ends.append(end)

    start = min(starts)
    end = max(ends)

    for level in landscapes:
        plt.plot(level[:start+end,0], level[:start+end,1])


