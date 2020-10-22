import argparse
import importlib
import os

from csv_loader import CSVDataset, extract_class

import numpy as np

import torch
from skorch.net import NeuralNet
from skorch import NeuralNetClassifier
from skorch.callbacks import EpochScoring, LRScheduler
from train.callbacks import TrainingThreshold
from train.extract_data import extract_data
from train.utils import save_activations
from landscape import landscapes_diagrams_from_model, save_landscape, average_from_disk
from diagram import save_diagram
from visualize import save_diagram_plots, save_landscape_plots

from utils import status

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
    y_train = np.array([y for X, y in iter(dataset)])

    # Prepare training callback
    training_accuracy = EpochScoring('accuracy',
                                     lower_is_better=False,
                                     on_train=True,
                                     name='Training_Accuracy')
    test_accuracy = EpochScoring('accuracy',
                                 lower_is_better=False,
                                 on_train=False,
                                 name='Test_Accuracy')

    training_threshold = TrainingThreshold(training_threshold=args.training_threshold,
                                           on_training=True)

    callbacks = [training_accuracy,
                 test_accuracy,
                 training_threshold]

    print("Network: {}".format(network()))
    print("Dataset Samples: {}".format(len(dataset)))

    network_id = 0
    while network_id <= args.network_count:
        status('STATUS: Beginning training of network {}'.format(network_id))
        print('Running on device: {}'.format(device))
        
        # initalize the network
        net = NeuralNetClassifier(
            network,
            criterion=torch.nn.CrossEntropyLoss,
            max_epochs=args.max_epochs,
            batch_size=args.batch_size,
            optimizer=torch.optim.Adam,
            lr=args.learning_rate,
            callbacks=callbacks,
            device=device)

        net.set_params(callbacks__valid_acc=None)
        net.id = network_id


        net.fit(X=dataset, y=y_train)

        if net.history[-1, 'Training_Accuracy'] >= args.training_threshold or args.ignore_failed == False:
            status('STATUS: Starting landscape computation for network {}'.format(network_id))
            landscape_data = extract_data(dataset, args.persistence_data_samples, args.persistence_class)
            # compute landscape statistics
            landscapes, diagrams = landscapes_diagrams_from_model(net,
                                                                landscape_data,
                                                                maxdims=args.max_diagram_dimension,
                                                                thresholds=args.diagram_threshold,
                                                                ns=args.persistence_layers,
                                                                dx=args.landscape_dx,
                                                                min_x=args.landscape_min_x,
                                                                max_x=args.landscape_max_x,
                                                                pd_metric=args.diagram_metric,
                                                                k=args.nn_graph_k,
                                                                activations_dirname=os.path.join(args.output_folder, './activations_visualizations/'))
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
                save_activations(net, dataset, os.path.join(args.output_folder, './activations/network{}'.format(net.id)))

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
