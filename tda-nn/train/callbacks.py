import torch
from skorch.callbacks import Callback

from utils import bcolors

class TrainingThreshold(Callback):
    def __init__(self, 
                 training_threshold=[1.0],
                 testing_threshold=[1.0],
                 output_prefix='network',
                 log_file='log.csv',
                 sink=print):

        self._training_threshold = training_threshold
        self._testing_threshold = testing_threshold
        self._output_prefix = output_prefix
        self._log_handle = open(log_file, "w")
        self._log_handle.write("network_name,training_accuracy,test_accuracy,epoch_number\n")
        self._sink = sink

    def on_epoch_end(self, net, dataset_train, dataset_valid):
        current_score = net.history[-1, "Training_Accuracy"]
        # check if current_score greater than any in training_threshold
        for threshold in self._training_threshold:
            if current_score >= threshold:
                network_name = "{}_{}.pt".format(self._output_prefix, threshold)
                torch.save(net.module_, network_name)
                self._log_handle.write("{},{},{},{}\n".format(network_name, net.history[-1, "Training_Accuracy"],\
                                                         net.history[-1, "Test_Accuracy"], len(net.history)))

                if threshold == self._training_threshold[-1]:
                    self._sink((bcolors.BOLD+"STATUS: Stopping network training early due to passing of final threshold accuracy of {} with an accuracy of {}"+bcolors.ENDC).format(threshold, current_score))
                    raise KeyboardInterrupt
                else:
                    self._sink((bcolors.BOLD+"STATUS: Saving network snapshot due to passing of threshold accuracy of {} with an accuracy of {}"+bcolors.ENDC).format(threshold, current_score))
                    self._training_threshold.remove(threshold)

        current_score = net.history[-1, "Test_Accuracy"]
        # check if current_score greater than any in testing_threshold
        for threshold in self._testing_threshold:
            if current_score >= threshold:
                network_name = "{}_{}.pt".format(self._output_prefix, threshold)
                torch.save(net.module_, network_name)
                self._log_handle.write("{},{},{},{}\n".format(network_name, net.history[-1, "Training_Accuracy"],\
                                                         net.history[-1, "Test_Accuracy"], len(net.history)))

                if threshold == self._testing_threshold[-1]:
                    self._sink((bcolors.BOLD+"STATUS: Stopping network testing early due to passing of final threshold accuracy of {} with an accuracy of {}"+bcolors.ENDC).format(threshold, current_score))
                    raise KeyboardInterrupt
                else:
                    self._sink((bcolors.BOLD+"STATUS: Saving network snapshot due to passing of threshold accuracy of {} with an accuracy of {}"+bcolors.ENDC).format(threshold, current_score))
                    self._testing_threshold.remove(threshold)

