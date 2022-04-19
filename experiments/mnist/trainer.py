import numpy as np

import torch
from skorch.net import NeuralNet
from skorch import NeuralNetClassifier
from skorch.callbacks import BatchScoring, EpochScoring, LRScheduler
from train.callbacks import TrainingThreshold
from train.losses import MSELossClassification, SphereLoss

import os
import sys
from csv_loader import CSVDataset

def train(model_fname,
          dataset,
          training_threshold,
          testing_threshold,
          max_epochs,
          batch_size,
          learning_rate,
          output_prefix,
          log_file,
          loss_fn):

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
    testing_accuracy = EpochScoring('accuracy',
                                     lower_is_better=False,
                                     on_train=False,
                                     name='Test_Accuracy')

    training_threshold = TrainingThreshold(training_threshold=training_threshold,
                                           testing_threshold=testing_threshold,
                                           output_prefix=output_prefix,
                                           log_file=log_file)

    callbacks = [training_accuracy,
                 testing_accuracy,
                 training_threshold]
    
    if loss_fn == 'MSELoss':
        loss = MSELossClassification
    elif loss_fn == 'SphereLoss':
        loss = SphereLoss
    elif loss_fn == 'CrossEntropyLoss':
        loss = torch.nn.CrossEntropyLoss
    else:
        print("ERROR: Un-supported loss function: {}".format(loss_fn))
        exit()
    print(loss)
    # initalize the network
    net = NeuralNetClassifier(
        network,
        criterion=loss,
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

    parser = argparse.ArgumentParser(description='Trains a network.')
    parser.add_argument('--model', type=str, default='model',
                        help='Model to load.')
    parser.add_argument('--csv-file', type=str, default='disk6.csv')
    parser.add_argument('--training-threshold', type=float, nargs='+', default=[1.0],
                        help='Training accuracy threshold (stop training at this accuracy).')
    parser.add_argument('--testing-threshold', type=float, nargs='+', default=[1.0],
                        help='Testing accuracy threshold (stop training at this accuracy).')
    parser.add_argument('--batch-size', type=int, default=1024, metavar='N',
                        help='input batch size for training (default: 1024)')
    parser.add_argument('--max-epochs', type=int, default=1024, metavar='N',
                        help='number of epochs to train (default: 1024)')
    parser.add_argument('--learning-rate', type=float, default=0.02, metavar='LR',
                        help='learning rate (default: 0.02)')
    parser.add_argument('--loss_fn', type=str, default="CrossEntropyLoss",
                        help='Loss function string.')
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
                args.learning_rate,
                args.output_name,
                args.log_name,
                args.loss_fn)

    final_training_acc = net.history[-1, "batches", -1, 'Testing_Accuracy']
