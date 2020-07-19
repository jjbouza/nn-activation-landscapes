import warnings
from ripser import Rips

def compute_diagram(data, rips):
    # compute and return diagram
    data_cpu = data.detach().cpu().numpy()
    samples = data_cpu.reshape(data.shape[0], -1)
    with warnings.catch_warnings():
        # supress ripser warnings
        warnings.simplefilter("ignore")
        pd = rips.fit_transform(samples)

    return pd

def compute_diagram_n(model, data, rips, n):    
    xn = model(data, n)
    return compute_diagram(xn, rips)

def compute_diagram_all(model, data, rips):
    activations = model(data)
    diagrams = [compute_diagram(activation, rips) for activation in activations]
    return diagrams


