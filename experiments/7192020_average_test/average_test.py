from visualize import plot_landscape
import sys
import pickle
import argparse
from persim import plot_diagrams
from matplotlib import pyplot as plt

from landscape import average

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Plot landscapes and diagrams')
    parser.add_argument('data', action='store')
    args = parser.parse_args()

    with open(args.data, 'rb') as data_f:
        data = pickle.load(data_f)

    flatten = lambda l: [item for sublist in l for item in sublist]
    mean = average( flatten(data['landscapes'])[:2] )
    plot_landscape(mean)
    plt.show()
