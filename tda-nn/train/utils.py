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


def save_activations(network, dataset, dname):
    os.makedirs(dname, exist_ok=True)
    network.module_.eval()
    model = network.module_.to('cpu')
    inp, out = dataset_to_tensor(dataset)
    with torch.no_grad():
        for i in range(len(network.module_.layers)+1):
            output = model(inp, i)
            to_save = torch.cat([out[..., None], output], dim=1)
            np.savetxt(os.path.join(dname, "layer{}.csv".format(i)), 
                    to_save.detach().cpu().numpy(),
                    delimiter=',')
