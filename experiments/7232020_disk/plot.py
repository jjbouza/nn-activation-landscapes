from visualize import plot_landscape
import sys
import pickle
import argparse
from persim import plot_diagrams
from matplotlib import pyplot as plt



if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Plot landscapes and diagrams')
    parser.add_argument('data', action='store')
    parser.add_argument('--dim', type=int, default=1)  
    parser.add_argument('--n', type=int, nargs='+', default=1)
    args = parser.parse_args()

    with open(args.data, 'rb') as data_f:
        landscape = pickle.load(data_f)

    i = 0
    landscape_ylim = [None, None]
    landscape_xlim = [None, None]

    fig, axes = plt.subplots(len(args.n), 1)
    for i, n in enumerate(args.n):
        # plot landscape
        plot_landscape(axes[i], landscape[n][args.dim][0], landscape[n][args.dim][1])
        
    min_xlim = min([ax.get_xlim()[0] for ax in axes])
    max_xlim = max([ax.get_xlim()[1] for ax in axes])

    min_ylim = min([ax.get_ylim()[0] for ax in axes])
    max_ylim = max([ax.get_ylim()[1] for ax in axes])
    
    for ax in axes:
        ax.set_xlim((min_xlim, max_xlim))
        ax.set_ylim((min_ylim, max_ylim))

    plt.show()
