# tda-nn
tda-nn is a package and set of experiments for analyzing neural networks using TDA, particularly Persistence Landscapes. 

## Quickstart

### Dependencies:
- Python3
- R with [tdatools](https://github.com/jjbouza/tda-tools)
- dill (`pip3 install dill`)
- numpy (`pip3 install numpy`)
- matplotlib (`pip3 install matplotlib`)
- ripser (`pip3 install ripser`)
- rpy2 (`pip3 install rpy2`)
- PyTorch (`pip3 install torch torchvision` or see [here](https://pytorch.org/get-started/locally/) for GPU)

### Code
The core files for computation of diagrams and landscapes are stored under `tda-nn/`. Experiments, along with experiment specific code is stored under `experiments/*`. 

The basic idea for using tda-nn is to define a PyTorch `nn.Module` class that implements a forward function with the signature `forward(data, n)`, where `n` means return the nth layer activation. To get landscapes at each layer of the network, you would then call `landscape.landscapes_diagrams_from_model(net, data, ...)`. This function will evaluate the network at each layer and use the activations to compute persistence diagrams and landscapes. See `experiments/7112020_mnist/compute_landscapes.py` for a full example. 

### Experiments

- `experiments/7112020_mnist`: Train an MNIST classifier and get the persistence landscapes at each layer.
- `experiments/7192020_mnist_with_averaging`: Train multiple MNIST classifiers and average the persistence landscapes at each layer for all of them.


## v0.1 TODO:
- Plot across layers as well (currently only plotting one PD and landscape for one layer). ✔
- x and y axes of each plot should be the same. ✔
- Write function to average PLs. ✔
- Train multiple networks and average. ✔
- Do multiple network training in parallel.
