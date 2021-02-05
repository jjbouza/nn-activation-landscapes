from skorch.callbacks import Callback

from utils import bcolors

class TrainingThreshold(Callback):
    def __init__(self, 
                 training_threshold=1.0,
                 on_training=True,
                 sink=print):

        self._training_threshold = training_threshold
        self._monitor = 'Training_Accuracy' if on_training else 'Test_Accuracy'
        self._sink = sink

    def on_epoch_end(self, net, dataset_train, dataset_valid):
        current_score = net.history[-1, self._monitor]
        if current_score >= self._training_threshold:
            self._sink((bcolors.BOLD+"STATUS: Stopping network training early due to passing of threshold accuracy of {} with an accuracy of {}"+bcolors.ENDC).format(self._training_threshold, current_score))
            raise KeyboardInterrupt

