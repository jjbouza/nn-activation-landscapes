import numpy as np
import scipy

_EPS_=1e-6
def _to_sphere_sqrt_(x):
    """
    map x to hypersphere.
    """
    y = np.sqrt(np.clip(x, 0, None))
    return y

def sphere_distance(x, y):
    x_ = _to_sphere_sqrt_(scipy.special.softmax(x, axis=-1))
    y_ = _to_sphere_sqrt_(scipy.special.softmax(y, axis=-1))
    
    z = np.clip(np.sum(x_*y_, axis=-1)[..., None], _EPS_-1, 1-_EPS_)
    return z
