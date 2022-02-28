import os

from sklearn.decomposition import PCA

import numpy as np

# some useful computational geometry tools
from sklearn.neighbors import NearestNeighbors
import sklearn.utils.graph_shortest_path as gp

from utils import *

def save_activations(base_dir, data, activations, logits_layer):
    data_map = {"ground_truth_class": "__scalar__", "infer_class": "__scalar__", "images": "__greyscale_image__"}
    for activation_id, activation in enumerate(activations):
        sample_data = {}
        data_activation = data[:activation.shape[0]]
        classes = data_activation[:,-1]
        data_activation_images = data_activation[:,:-1]
        data_activation_images = (data_activation_images+1)*255

        pca = PCA(n_components=3)
        activation_pca = pca.fit_transform(activation)

        sample_data["__data__"] = activation_pca
        sample_data["__data_map__"] = data_map
        sample_data["ground_truth_class"] = classes.astype('int')
        sample_data["infer_class"] = np.argmax(logits_layer, axis=-1).astype('int')
        sample_data["__color__"] = (sample_data["ground_truth_class"] == sample_data["infer_class"]).astype('int')
        sample_data["images"] = data_activation_images

        filename = os.path.join(base_dir, "activation{}.npz".format(activation_id))
        np.savez_compressed(filename, **sample_data)



if __name__ == '__main__':
    import argparse

    def load_activations(fnames):
        data = []
        for fname in sorted(os.listdir(fnames)):
            path = os.path.join(fnames, fname)
            if os.path.splitext(path)[1] == '.csv':
                data.append(np.loadtxt(path, delimiter=','))
            else:
                error("Error: invalid file extension {}, this script only support CSV datasets.".format(os.path.splitext(fname)[1]))
                quit()

        return data

    def load_data(fname):
        if os.path.splitext(fname)[1] == '.npy':
            data_numpy = np.load(fname)
        elif os.path.splitext(fname)[1] == '.csv':
            data_numpy = np.loadtxt(fname, delimiter=',')
        else:
            error("Error: invalid file extension {}, this script only support numpy or CSV datasets.".format(os.path.splitext(fname)[1]))
            quit()

        return data_numpy

    parser = argparse.ArgumentParser(description='Plot activations.')
    parser.add_argument('--activations', type=str)
    parser.add_argument('--data', type=str)
    parser.add_argument('--persistence-layers', type=int, nargs='+')
    parser.add_argument('--logit-layer', type=int)
    parser.add_argument('--output-dir', type=str)

    args = parser.parse_args()
    
    # take first persistence-data-samples rows from activations
    activations = load_activations(args.activations)
    activations = [activations[layer] for layer in args.persistence_layers]
    logits = activations[args.logit_layer]
    data = load_data(args.data)
    save_activations(args.output_dir, data, activations, logits)
