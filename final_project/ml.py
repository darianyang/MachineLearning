"""
Some helper functions to create the initial dataset for ML.
TODO: Eventually, integrate into wedap.
"""

import numpy as np
import h5py

from tqdm.auto import tqdm
import sys

### helper functions normally avail in wedap ###
def get_parents(f, walker_tuple):
    it, wlk = walker_tuple
    parent = f[f"iterations/iter_{it:08d}"]["seg_index"]["parent_id"][wlk]
    return it-1, parent

def trace_walker(f, walker_tuple):
    # Unroll the tuple into iteration/walker 
    it, wlk = walker_tuple
    # Initialize our path
    path = [(it,wlk)]
    # And trace it
    while it > 1: 
        it, wlk = get_parents(f, (it, wlk))
        path.append((it,wlk))
    return np.array(sorted(path, key=lambda x: x[0]))

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
    # TODO: option to skip certain auxdata features (e.g. secondary structure)

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

    # save tau value (columns of pcoord array)
    tau = np.atleast_3d(np.array(f[f"iterations/iter_{last_iter:08d}/pcoord"])).shape[1]

    # n features per iteration (must be constant for all iterations)
    n_features = len(list(f[f"iterations/iter_{last_iter:08d}/auxdata"]))
    # and every pcoord tracked (n depth)
    n_pcoords = np.atleast_3d(np.array(f[f"iterations/iter_{last_iter:08d}/pcoord"])).shape[2]
    n_features += n_pcoords

    # get the names of each feature
    feat_names = [f"pcoord_{dim}" for dim in range(n_pcoords)] + \
                 list(f[f"iterations/iter_{last_iter:08d}/auxdata"])
    #print(feat_names)

    # initialize empty but correctly shaped array
    ml_input = np.zeros((segs_in_last, last_iter * n_features))
    # as expected, it is 24 by (137i * (47 aux + 2 pcoord) features) = 6713 rows
    #print(ml_input.shape)

    # testing trace output
    #print(trace_walker(f, (last_iter, 2))[0])

    ### fill the empty array ###
    # loop each segment index of the last_iter choosen
    for seg_i in tqdm(range(segs_in_last), desc="Creating initial input array"):
        # get trace path back to bstate
        trace = trace_walker(f, (last_iter, seg_i))
        
        # take trace path and loop each (it, wlk) item (begins at i1)
        for it, wlk in trace:
            
            # loop each feature name
            for feat_i, feat in enumerate(feat_names):
                
                # need to account for pcoords before auxdata
                if feat[:-2] == "pcoord":
                    feat_name = "pcoord"
                    feat_depth = int(feat[-1:])
                else:
                    feat_name = f"auxdata/{feat}"
                    # TODO: eventually can account for multi depth dim auxdata
                    feat_depth = 0

                ### calc the ∆feat value of each feature of current (it, wlk) ###
                # first grab the correctly indexed (it, wlk, feat) array from h5 file
                it_wlk_feat_data = np.atleast_2d(f[f"iterations/iter_{it:08d}/{feat_name}"][wlk])
                # properly shape the array for ndims of auxdata
                if feat_name[:7] == "auxdata":
                    # (if tau=11) goes from row to column (1, 11) to (11, 1); pcoord is (11, n)
                    it_wlk_feat_data = it_wlk_feat_data.reshape(tau, 1)

                # ∆feat = |last frame - first frame|
                d_feat = np.absolute(it_wlk_feat_data[:, feat_depth][-1] - 
                                     it_wlk_feat_data[:, feat_depth][0])

                # assign values of ml_input array (row=seg_i, col=feat_i+((it-1)*n_features))
                ml_input[seg_i, feat_i + ((it - 1) * n_features)] = d_feat

    return ml_input
            

# eventually for optimized weight tracking, make dict of aux names

if __name__ == "__main__":
    # ml_input = create_ml_input()
    # np.savetxt("we_ml_input.tsv", ml_input, delimiter="\t")
    
    ml_input = np.loadtxt("we_ml_input.tsv")
    print(ml_input)