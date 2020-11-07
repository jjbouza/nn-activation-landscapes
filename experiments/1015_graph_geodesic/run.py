import argparse
import importlib
import os

from csv_loader import CSVDataset

import numpy as np

import torch
import trainer
from train.extract_data import extract_data
from activations import compute_activations, save_activations
from landscape import compute_landscapes, save_landscape
from diagram import compute_diagrams, save_diagram
from visualize import save_diagram_plots, save_landscape_plots

from utils import status

import time

def generate_cli_parser():
    parser = argparse.ArgumentParser(description='PyTorch landscape computations')
    # Training settings
    parser.add_argument('--output_folder', type=str, default='./')
    parser.add_argument('--network-count', type=int, default=16)
    parser.add_argument('--csv-file', type=str, default='disk6.csv')
    parser.add_argument('--model', type=str, default='model',
                        help='Model to load.')
    parser.add_argument('--training-threshold', type=float, nargs='+', default=1.0,
                        help='Training accuracy threshold (stop training at this accuracy).')
    parser.add_argument('--batch-size', type=int, default=1024, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--max-epochs', type=int, default=18000, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--learning-rate', type=float, default=0.05, metavar='LR',
                        help='learning rate (default: 0.02)')
    parser.add_argument('--ignore-failed', default=True, action='store_true',
                        help='Do not count failed training (less than training threshold)')

    # diagram and landscape computation settings
    parser.add_argument('--diagram-metric', type=str, default='L2',
                        help='Persistence Diagram metric. Options: L2, GG (graph geodesic), SN (scale normalized L2)')
    parser.add_argument('--nn-graph-k', type=int, default=12)
    parser.add_argument('--max-diagram-dimension', type=int, nargs='+', default=2,
                        help='List of maxdims to compute diagrams and landscapes at for each layer.')  
    parser.add_argument('--diagram-threshold', type=float, nargs='+', default=10,
                        help='List of thresholds to compute diagrams at for each layer.')  
    parser.add_argument('--persistence-layers', type=int, nargs='+', default=1,
                        help='List of which layers to compute landscapes at.')  
    parser.add_argument('--persistence-data-samples', type=int, default=2000,
                        help='Number of data samples to pass through network to get per-layer activations.')  
    parser.add_argument('--landscape-dx', type=float, default=0.1,
                        help='x-spacing for landscape sampling')  
    parser.add_argument('--landscape-min-x', type=float, default=0,
                        help='min x to sample landscape')  
    parser.add_argument('--landscape-max-x', type=float, default=10,
                        help='max x to sample landscape')  

    parser.add_argument('--save-landscape', default=True, action='store_true',
                        help='Save landscape to output csv files.')  
    parser.add_argument('--save-diagram', default=True, action='store_true',
                        help='Save diagrams to output csv files.')
    parser.add_argument('--save-activations', default=True, action='store_true',
                        help='Save network activations (landscape-class controls class to save activations of).')
    parser.add_argument('--save-diagram-plots', default=True, action='store_true',
                        help='Save plots of persistence diagrams')
    parser.add_argument('--save-landscape-plots', default=True, action='store_true',
                        help='Save plots of persistence landscapes')
    parser.add_argument('--save-mean-landscapes', default=True, action='store_true',
                        help='Compute and save mean persistence landscape')

    parser.add_argument('--persistence-class', type=int, default=1,
                        help='Class of samples to run persistence evaluation on. (-1 for all classes)')


    return parser

def main(args):
    network = __import__(args.model).Net
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # load data
    dataset = CSVDataset(args.csv_file)

    print("Network: {}".format(network()))
    print("Dataset Samples: {}".format(len(dataset)))

    network_id = 0
    while network_id <= args.network_count:
        status('STATUS: Beginning training of network {}'.format(network_id))
        print('Running on device: {}'.format(device))
        
        net = trainer.train(args.model,
                            dataset,
                            args.training_threshold,
                            args.max_epochs,
                            args.batch_size, 
                            args.learning_rate)

        net.id = network_id

        if net.history[-1, 'Training_Accuracy'] >= args.training_threshold or args.ignore_failed == False:
            status('STATUS: Starting persistence computation for network {}'.format(network_id))
            landscape_data = extract_data(dataset, args.persistence_data_samples, args.persistence_class)
            # compute landscape statistics

            status("STATUS: Computing activations for network {}".format(network_id))
            start = time.time()
            activations = compute_activations(net, 
                                              landscape_data,
                                              layers=args.persistence_layers)
            end = time.time()
            status("STATUS: Done computing activations for network {}. Time elapsed: {} s".format(network_id, end-start))
            
            status("STATUS: Computing diagrams for network {}".format(network_id))
            start = time.time()
            diagrams = compute_diagrams(activations, 
                                      maxdims=args.max_diagram_dimension,
                                      thresholds=args.diagram_threshold,
                                      metric=args.diagram_metric,
                                      k=args.nn_graph_k,
                                      save_GG_activations_plots=os.path.join(args.output_folder, 'activation_visualizations/network{}/'.format(network_id)))
            end = time.time()
            status("STATUS: Done computing diagrams for network {}. Time elapsed: {} s".format(network_id, end-start))

            status("STATUS: Computing landscapes for network {}".format(network_id))
            start = time.time()
            landscapes = compute_landscapes(diagrams,
                                            args.landscape_dx,
                                            args.landscape_min_x,
                                            args.landscape_max_x,
                                            thresholds=args.diagram_threshold)
            end = time.time()
            status("STATUS: Done computing landscapes for network {}. Time elapsed: {} s".format(network_id, end-start))
            
            if args.save_diagram_plots:
                status("STATUS: Saving diagram plots for network {}".format(network_id))
                save_diagram_plots(diagrams, os.path.join(args.output_folder, './diagram_plots/network{}'.format(net.id)))

            # save diagrams
            if args.save_diagram:
                status("STATUS: Saving diagrams for network {}".format(network_id))
                save_diagram(diagrams, os.path.join(args.output_folder, './diagrams_csv/network{}'.format(net.id)))

            # save landscapes
            if args.save_landscape:
                status("STATUS: Saving landscapes for network {}".format(network_id))
                save_landscape(landscapes, os.path.join(args.output_folder, './landscapes_csv/network{}'.format(net.id)))
            
            if args.save_activations:
                status("STATUS: Saving activations for network {}".format(network_id))
                # TODO: use activations from above
                save_activations(activations, os.path.join(args.output_folder, './activations/network{}'.format(net.id)))

            if args.save_landscape_plots:
                status("STATUS: Saving landscape plots for network {}".format(network_id))
                save_landscape_plots(landscapes, os.path.join(args.output_folder, './landscape_plots/network{}'.format(net.id)))

            network_id += 1

        else:
            status("STATUS: Training failed for network {}, not counting this network.".format(network_id))

        # compute the mean landscape using CONSTANT MEMORY and save.
        #if args.save_mean_landscapes:
        #    average_from_disk(os.path.join(args.output_folder, './landscapes_csv'))


if __name__=='__main__':
    args = generate_cli_parser().parse_args()
    main(args)
