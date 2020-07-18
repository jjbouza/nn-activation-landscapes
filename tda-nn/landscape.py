import numpy as np
import rpy2
from rpy2.robjects.packages import importr

def landscape(diagram, dx=0.1,  min_x= 0, max_x=10, threshold=-1):
    """
    Simple interface to tda-tools for DISCRETE landscapes.
    """

    return np.array(landscale.tdatools(diagram, dx, min_x, max_x, threshold))

landscape.tdatools = importr('tdatools')
