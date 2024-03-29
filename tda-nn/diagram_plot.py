from visualize import save_diagram_plots
import numpy as np

if __name__=='__main__':
    import argparse
    import warnings
    import os
    import re
    warnings.simplefilter("ignore", UserWarning)

    def load_data(fnames):
        data = {}
        max_layers, max_dims = 0,0
        for fname in sorted(os.listdir(fnames)):
            fname_meta = re.findall("[0-9]+", fname)
            layer, dim = int(fname_meta[0]), int(fname_meta[1])
            if layer > max_layers:
                max_layers = layer
            if dim > max_dims:
                max_dims = dim
            path = os.path.join(fnames, fname)
            if os.path.splitext(path)[1] == '.csv':
                network_data = np.loadtxt(path, delimiter=',')
                if network_data.shape[0] == 0:
                    data[(layer, dim)] = np.zeros([0,2])
                elif len(network_data.shape) == 1:
                    data[(layer, dim)] = np.reshape(network_data, [1, network_data.shape[0]])
                else:
                    data[(layer, dim)] = network_data
            else:
                error("Error: invalid file extension {}, this script only support CSV datasets.".format(os.path.splitext(fname)[1]))
                quit()
        
        data_final = [[None for j in range(max_dims+1)] for i in range(max_layers+1)]
        for i in range(max_layers+1):
            for j in range(max_dims+1):
                data_final[i][j] = data[(i,j)]
        
        return data_final

    parser = argparse.ArgumentParser(description='Plot diagrams.')
    parser.add_argument('--diagrams', type=str)
    parser.add_argument('--includes', type=int, nargs='+', default=None)
    parser.add_argument('--output-dir', type=str)

    args = parser.parse_args()
    diagrams = load_data(args.diagrams)
    save_diagram_plots(diagrams, args.output_dir, include=args.includes)
