import numpy as np

import torch
from skorch.net import NeuralNet
from skorch import NeuralNetClassifier
from skorch.callbacks import EpochScoring, LRScheduler
from train.callbacks import TrainingThreshold

import os
import sys
from csv_loader import CSVDataset

def train(model_fname,
          dataset,
          training_threshold,
          testing_threshold,
          max_epochs,
          batch_size,
          learning_rate):
    sys.path.append(os.path.dirname(model_fname))
    network = __import__(os.path.basename(model_fname)).Net
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # load data
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

    training_threshold = TrainingThreshold(training_threshold=training_threshold,
                                           on_training=True)
    testing_threshold  = TrainingThreshold(training_threshold=testing_threshold,
                                           on_training=False)

    callbacks = [training_accuracy,
                 test_accuracy,
                 training_threshold,
                 testing_threshold]
    
    # initalize the network
    net = NeuralNetClassifier(
        network,
        criterion=torch.nn.CrossEntropyLoss,
        max_epochs=max_epochs,
        batch_size=batch_size,
        optimizer=torch.optim.Adam,
        lr=learning_rate,
        callbacks=callbacks,
        iterator_train__shuffle=True,
        device=device)

    net.set_params(callbacks__valid_acc=None)
    net.fit(X=dataset, y=y_train)

    return net

if __name__=='__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Compute activations given a network and data.')
    parser.add_argument('--model', type=str, default='model',
                        help='Model to load.')
    parser.add_argument('--csv-file', type=str, default='disk6.csv')
    parser.add_argument('--training-threshold', type=float, nargs='+', default=1.0,
                        help='Training accuracy threshold (stop training at this accuracy).')
    parser.add_argument('--testing-threshold', type=float, nargs='+', default=1.0,
                        help='Testing accuracy threshold (stop training at this accuracy).')
    parser.add_argument('--batch-size', type=int, default=1024, metavar='N',
                        help='input batch size for training (default: 1024)')
    parser.add_argument('--max-epochs', type=int, default=1024, metavar='N',
                        help='number of epochs to train (default: 1024)')
    parser.add_argument('--learning-rate', type=float, default=0.02, metavar='LR',
                        help='learning rate (default: 0.02)')
    parser.add_argument('--output-name', type=str)
    parser.add_argument('--log-name', type=str)
    
    args = parser.parse_args()
    dataset = CSVDataset(args.csv_file)

    net = train(args.model,
                dataset,
                args.training_threshold,
                args.testing_threshold,
                args.max_epochs,
                args.batch_size,
                args.learning_rate)
    final_training_acc = net.history[-1, 'Training_Accuracy']
    final_testing_acc = net.history[-1, 'Test_Accuracy']

    torch.save(net.module_, args.output_name)
    with open(args.log_name, 'w') as log_handle:
        log_handle.write('Final Training Accuracy: {} \nFinal Test Accuracy: {}\n'.format(final_training_acc,
                                                                                         final_testing_acc))
    
