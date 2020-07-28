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
        data = pickle.load(data_f)

    i = 0
    landscape_ylim = None
    landscape_xlim = None

    for n in args.n:
        # plot landscape
        i += 1
        landscape = data
        plt.subplot(len(args.n), 1, i)
        plot_landscape(landscape[n][args.dim])

        if landscape_ylim == None:
            landscape_ylim = plt.gca().get_ylim()
            landscape_xlim = plt.gca().get_xlim()

        plt.ylim(landscape_ylim)
        plt.xlim(landscape_xlim)

    plt.show()
