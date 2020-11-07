import torch
import numpy as np

import os

def dataset_to_tensor(dataset):
    inputs = []
    outputs = []
    for entry in dataset:
        inputs.append(entry[0])
        outputs.append(entry[1])

    return torch.stack(inputs, dim=0), torch.stack(outputs, dim=0)


