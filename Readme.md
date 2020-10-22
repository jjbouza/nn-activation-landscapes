# tda-nn
tda-nn is a package and set of experiments for analyzing neural networks using TDA, particularly Persistence Landscapes. 

![Average Landscapes](https://github.com/jjbouza/tda-nn/blob/master/experiments/7192020_mnist_with_averaging/result1.png)

## Quickstart

### Dependencies:
- Python3
- R with [tdatools](https://github.com/jjbouza/tda-tools)
- numpy (`pip3 install numpy`)
- matplotlib (`pip3 install matplotlib`)
- ripser (`pip3 install ripser`)
- rpy2 (`pip3 install rpy2`)
- PyTorch (`pip3 install torch torchvision` or see [here](https://pytorch.org/get-started/locally/) for GPU)

### Code
The core library files for computation of diagrams and landscapes are stored under `tda-nn/`. Experiments, along with experiment specific code is stored under `experiments/*`. 

The basic idea for using tda-nn is to define a PyTorch `nn.Module` class that implements a forward function with the signature `forward(data, n)`, where `n` means return the nth layer activation. 


### Experiments
Previous experiments removed for new version. Current experiments are 

- `experiments/1015_graph_geodesic: Aimed at reproducing synthetic MLP results from original LHL paper from UChicago.

### Anatomy of 1015_reproduce Call
The 1015_graph_geodesic experiment is extremely versatile. Here is part of the example run.sh file included under experiments/1015_geodesic/run.sh:


```
  PYTHONPATH=../../tda-nn python3 ./run.py \
      --output_folder $OUTPUT_FOLDER \
      --model model \
      --network-count 30 \
      --training-threshold 0.995 \
      --batch-size 2560 \
      --max-epochs 8000 \
      --learning-rate 0.01 \
      --diagram-metric GG \
      --max-diagram-dimension 1 1 1 1 1 1 1 1 1 1 1 1\
      --diagram-threshold 100000 100000 100000 100000 100000 100000 100000 100000 100000 100000 100000 100000\
      --persistence-layers 0 1 2 3 4 5 6 7 8 9 10 11\
      --persistence-data-samples 4000 \
      --landscape-dx 0.001 \
      --landscape-min-x 0 \
      --landscape-max-x 5 \
      --persistence-class 1 \
      --save-landscape \
      --save-diagram \
      --save-activations \
      --save-mean-landscapes \
```

The relevant arguments are described below:

- **--output_folder:** Base directory for saving experiment output data. 
- **--model:** Name of model file. For example, `--model my_model` means import the network from `my_model.py`
- **--network-count:** Number of networks to train and compute landscapes on.
- **--training-threshold:** Stop training when training set accuracy reaches this level.
- **--batch-size:** Batch size for training.
- **--max_epochs:** Max number of epochs to train a network. If network training accuracy is below the specified training threshold and `--ignore-failed` is not set then this network will not count towards the `--network-count` (i.e. a new network will be initialized and attempted to be trained to the specified training threshold).
- **--learning-rate:** Training learning rate (an Adam optimizer is used internally).
- **--diagram_metric:** Persistence Diagram metric. Options: L2, GG (graph geodesic), SN (scale normalized L2). GG is recommended. 
- **--max-diagram-dimension:** Max dimension to compute homology in for each of the networks for each layer of network.
- **--diagram-threshold:** Persistence threshold for each layer of network.
- **--persistence-layers:** Layers to compute persistence at. The number of elements in this argument should match the previous two.
- **--persistence-data-samples:** Number of data samples to run persistence algorithm on (a subset of the full dataset).
- **--persistence-class:** When computing activation homology, we sometimes want to only compute homology for activations that come from a certain class. This class can be specified here. To do both classes set this to -1. 
