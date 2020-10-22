import torch
import random

def extract_data(dataset, samples, of_class=None):
    index_list = []
    for i in range(len(dataset)):
        if of_class is -1 or dataset[i][1].item() == of_class:
            index_list.append(i)

    random.shuffle(index_list)

    tensor_list = []
    for i in range(samples):
        tensor_list.append(dataset[index_list[i]][0].unsqueeze(0))

    return torch.cat(tensor_list)

