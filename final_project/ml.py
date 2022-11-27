"""
Some helper functions to create the initial dataset for ML.
TODO: Eventually, integrate into wedap.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import h5py

### helper functions normally avail in wedap ###
def get_parents(f, walker_tuple):
    it, wlk = walker_tuple
    parent = f[f"iterations/iter_{it:08d}"]["seg_index"]["parent_id"][wlk]
    return it-1, parent

def trace_walker(walker_tuple):
    # Unroll the tuple into iteration/walker 
    it, wlk = walker_tuple
    # Initialize our path
    path = [(it,wlk)]
    # And trace it
    while it > 1: 
        it, wlk = get_parents((it, wlk))
        path.append((it,wlk))
    return np.array(sorted(path, key=lambda x: x[0]))

def get_coords(path, data_name, data_index):
    # Initialize a list for the pcoords
    coords = []
    # Loop over the path and get the pcoords for each walker
    # for it, wlk in path:
    #     coords.append(self._get_data_array(data_name, data_index, it)[wlk][::10])
    return np.array(coords)

def create_ml_input(h5="data/ctd_ub_1d_v00.h5", last_iter=137):
    """
    Need a dataset from west.h5 as follows:
    rows: 1 for each segment of --last-iter, traced back to bstate to rep each iteration
          so if --last-iter is 100 and has 127 segments, 127 rows
    cols: 1 for each ∆iteration, last_frame - first_frame, with col for each feature
          so n iteations by n features
    
    Parameters
    ----------
    h5 : str
        Path to west.h5 file.
    last_iter : int
        The iteration to consider for data extraction.
        Using 137 based on the ctd_ub_1d_v00 dataset.

    Returns
    -------
    ml_input : 2d array
    """
    # import west.h5 data file
    f = h5py.File(h5, "r")

    # n segments per iteration
    n_particles = f["summary"]["n_particles"]
    # for the specfied iteration
    segs_in_last = n_particles[last_iter-1]
    # correctly choosen
    #print(segs_in_last)

    # n features per iteration (must be constant for all iterations)
    n_features = len(list(f[f"iterations/iter_{last_iter:08d}/auxdata"]))
    # and every pcoord tracked (n depth)
    n_features += np.atleast_3d(np.array(f[f"iterations/iter_{last_iter:08d}/pcoord"])).shape[2]

    # initialize correctly shaped array
    ml_input = np.zeros((segs_in_last, last_iter * n_features))
    #print(ml_input.shape)

    


if __name__ == "__main__":
    create_ml_input()