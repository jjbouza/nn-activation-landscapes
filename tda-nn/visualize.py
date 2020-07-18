import numpy as np
import sys
from matplotlib import pyplot as plt
from persim import visuals

if __name__ == '__main__':
    pds = np.load(sys.argv[1], allow_pickle=True)
    pds = [pds[f] for f in pds.files]
    visuals.plot_diagrams(pds[2][1])
    plt.show()