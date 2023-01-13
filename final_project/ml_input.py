"""
Generate input for ML pcoord.
"""

import numpy as np
import pandas as pd
import h5py

import matplotlib.pyplot as plt
import sklearn.metrics
import sklearn.preprocessing
import sklearn.model_selection

from tqdm.auto import tqdm
import sys

plt.style.use("/Users/darian/github/wedap/wedap/styles/default.mplstyle")

# Suppress divide-by-zero in log
np.seterr(divide="ignore", invalid="ignore")

class ML_Input_Gen:
    def __init__(self, h5, first_iter=1, last_iter=None, savefile=None,
                 skip_feats=[], n_succ=0, label_space=None, rand_ml_input=False):
        """
        Methods to generate input for machine learning based pcoord from a west.h5 file.

        # TODO: option to change standardization/normalization type
        #       e.g. max, l1, l2, std, None ; 'delta', 'None'
        Parameters
        ----------
        h5 : str
            Path to west.h5 file.
        first_iter : int
            The lower bound iteration to consider for data extraction.
        last_iter : int
            The upper bound iteration to consider for data extraction.
        savefile : str
            Optional path to save the output ml_input array as a tsv.
            Saves the features as X_{savefile} and classifications as y_{savefile}.
        only_feats : list (TODO)
            List of str feature names, if included, only the listed features will be included.
        skip_feats : list
            List of str feature names to not include in ml_input dataset.
            By default is an empty list, so no features are skipped.
            TODO: option to toggle to select only these feats (inverse this).
        n_succ : int
            Number of (iter, seg) additional pairs in each successfull trace path to label as True.
            Default 0, so only the recycled iteration. Increasing this value and including more
            history may be useful but will need to be optimized on a case-by-case basis.
        label_space : list
            List of 3 elements: [(feat_name), (gt or lt), (float or int)]
            This determines which segments are labeled as True.
            e.g. label_space=["pcoord_0", "gt", 37] (TODO: update to be multi-dim)
                 every seg with pcoord_0 value > 37 will be counted as True.
            With None, use the recycled trajectories from west.h5.
        rand_ml_input : bool
            Default False, if True, ml_input is random values with 50/50 T/F labels.
        """
        # import west.h5 data file
        self.h5 = h5py.File(h5, "r")
        self.first_iter = first_iter
        if last_iter:
            self.last_iter = last_iter
        else:
            self.last_iter = self.h5.attrs["west_current_iteration"] - 1

        # n segments per all iterations
        self.n_particles = self.h5["summary"]["n_particles"]
        # for the specfied iterations
        self.total_segs = np.sum(self.n_particles[self.first_iter-1:self.last_iter])

        # save tau value (columns of pcoord array)
        self.tau = np.atleast_3d(np.array(self.h5[f"iterations/iter_{self.last_iter:08d}/pcoord"])).shape[1]

        # n features per iteration (must be constant for all iterations)
        #self.n_features = len(list(self.h5[f"iterations/iter_{self.last_iter:08d}/auxdata"]))
        self.n_features = 0
        for aux in list(self.h5[f"iterations/iter_{self.last_iter:08d}/auxdata"]):
            # only skip if specified
            if aux not in skip_feats:
                self.n_features += \
                np.atleast_3d(np.array(self.h5[f"iterations/iter_{self.last_iter:08d}/auxdata/{aux}"])).shape[2]
        # and every pcoord tracked (n depth) unless otherwise skipped
        if "pcoord" not in skip_feats:
            n_pcoords = np.atleast_3d(np.array(self.h5[f"iterations/iter_{self.last_iter:08d}/pcoord"])).shape[2]
            self.n_features += n_pcoords

        # TODO: *** adjust this to handle multi-dim aux data
        
        # get the names of each feature (and multiple pcoords)
        self.feat_names = [f"pcoord_{dim}" for dim in range(n_pcoords)] + \
                           list(self.h5[f"iterations/iter_{self.last_iter:08d}/auxdata"])

        # optionally skip certain specified features
        if skip_feats:
            self.skip_feats = skip_feats
            # subtract the skipped feats from total n_features
            self.n_features -= len(skip_feats)
            # remove skipped feats from feature name list
            self.feat_names = [i for i in self.feat_names if i not in self.skip_feats]

        # ml_input array options
        # number of (iter, seg) pairs to count as True
        self.n_succ = n_succ
        # optionally save ml_input
        self.savefile = savefile
        self.label_space = label_space
        self.rand_ml_input = rand_ml_input

    ### trace_walker and get_parents helper methods normally avail in wedap ###
    def get_parents(self, walker_tuple):
        it, wlk = walker_tuple
        parent = self.h5[f"iterations/iter_{it:08d}"]["seg_index"]["parent_id"][wlk]
        return it-1, parent

    def trace_walker(self, walker_tuple):
        # Unroll the tuple into iteration/walker 
        it, wlk = walker_tuple
        # Initialize our path
        path = [(it,wlk)]
        # And trace it (TODO: maybe add option for full trace?)
        #while it > 1: 
        # limit the trace to n_succ
        for _ in range(self.n_succ):
            # added to prevent tracing before iter 1
            if it == 1:
                break
            it, wlk = self.get_parents((it, wlk))
            path.append((it,wlk))
        return np.array(sorted(path, key=lambda x: x[0]))

    def w_succ(self):
        """
        Find and return all successfully recycled (iter, seg) pairs.
        """
        succ = []
        for iter in range(self.last_iter):
            # if the new_weights group exists in the h5 file
            if f"iterations/iter_{iter:08d}/new_weights" in self.h5:
                prev_segs = self.h5[f"iterations/iter_{iter:08d}/new_weights/index"]["prev_seg_id"]
                # append the previous iter and previous seg id recycled
                for seg in prev_segs:
                    succ.append((iter-1, seg))
        # TODO: order this by iter and seg vals? currently segs not sorted
        return succ
            
    def create_ml_input(self, norm=False):
        """
        Generate ml dataset from west.h5.

        Returns
        -------
        ml_input : 2d array
            Array of features for each segment in specified last_iter.
        seg_labels : 1d array
            Array of binary labels with successful trajectories as True.
        norm : bool
            Normalize the final array.
        """
        # initialize empty but correctly shaped array
        # rows are segs for each iter in specified range, cols are n_features
        ml_input = np.zeros((self.total_segs, self.n_features))

        # 1d empty array for binary y labels
        seg_labels = np.zeros((self.total_segs))

        # TODO: right now label_space is not ready for use: 
        #       I have to eventually set this up to be able to use not yet recycled and 
        #       "assigned" space from label_space as True labels
        # might be better to use w_assign, then use that file for this, is there w_assign api?
        # previous code:
            # matching the input feature name for label cutoff
            # if feat == label_space[0]:
            #     # if a certain cutoff is met, set label as true (1), otherwise false (0)
            #     if label_space[1] == "gt":
            #         if np.any(it_wlk_feat_data[:,feat_depth] > label_space[2]):
            #             label = 1
            #     elif label_space[1] == "lt":
            #         if np.any(it_wlk_feat_data[:,feat_depth] < label_space[2]):
            #             label = 1
            #     else:
            #         raise ValueError(f"label_space[1] must be 'gt' or 'lt', not {label_space[1]}")
        # find list of successfully recycled trajectories: (iter,seg) pairs
        if self.label_space is None:
            succ_traces = []
            succ_pairs = self.w_succ()

            # make an array for all succ (iter,seg) traced paths
            for pair in succ_pairs:
                # filter for only iterations considered
                if pair[0] >= self.first_iter and pair[0] <= self.last_iter:
                    trace = self.trace_walker(pair)
                    succ_traces.append(trace)

            # implicit booleanness of list
            if not succ_traces:
                raise NameError("No successfull trajectories found, please input a label_space criterion")

            # unique (iter,seg) pairs from each trace of each succ_traj (iter,seg) pair
            succ_traces = np.unique(np.concatenate(succ_traces), axis=0)
            # needs to be formatted to accommodate the succ traj label lookup
            succ_traces = [(i[0], i[1]) for i in succ_traces]

        else:
            raise ValueError("label_space arg is not working yet...")

        # output a random dataset for testing the optimzation
        if self.rand_ml_input:
            # make labels 50/50 T/F (back half of array as True (1))
            seg_labels[(int(seg_labels.shape[0] / 2)):] = 1

            # put random values for ml_input
            ml_input = np.random.rand(self.total_segs, self.n_features)

        # otherwise make the ml_input array
        else:
            # overall seg row count
            seg_n = 0
            # loop each specified iteration and walker pair to fill array
            for it in tqdm(range(self.first_iter, self.last_iter + 1), desc="Creating input array"):
                for wlk in range(self.n_particles[it - 1]):
                    # labeling as True if apart of succ traj paths
                    # can label only recycled (iter,seg) pairs or n trace pairs
                    if (it, wlk) in succ_traces:
                        label = 1
                    else:
                        label = 0

                    # loop each feature name
                    for feat_i, feat in enumerate(self.feat_names):
                        # need to account for pcoords before auxdata
                        if feat[:-2] == "pcoord":
                            feat_name = "pcoord"
                            feat_depth = int(feat[-1:])
                        else:
                            feat_name = f"auxdata/{feat}"
                            # TODO: eventually can account for multi depth dim auxdata
                            # e.g. for a distance matrix (maybe W184 M1 to all M2 residues and vv)
                            feat_depth = 0

                        ### calc the ∆feat value of each feature of current (it, wlk) ###
                        # first grab the correctly indexed (it, wlk, feat) array from h5 file
                        it_wlk_feat_data = np.atleast_2d(self.h5[f"iterations/iter_{it:08d}/{feat_name}"][wlk])
                        # properly shape the array for ndims of auxdata
                        if feat_name[:7] == "auxdata":
                            # (if tau=11) goes from row to column (1, 11) to (11, 1); pcoord is (11, n)
                            it_wlk_feat_data = it_wlk_feat_data.reshape(self.tau, 1)

                        # ∆feat = |last frame - first frame|
                        d_feat = np.absolute(it_wlk_feat_data[:, feat_depth][-1] - 
                                            it_wlk_feat_data[:, feat_depth][0])

                        # assign values of ml_input array
                        ml_input[seg_n, feat_i] = d_feat

                    # label the segment
                    seg_labels[seg_n] = label
                    # iterate the overall row number
                    seg_n += 1

            # standardize: actually, just norm should be fine since these are |∆values|
            #ml_input = sklearn.preprocessing.StandardScaler().fit_transform(ml_input)
            
            # l2 is sum of squares norm, l1 is sum of abs vector values, max is maximum value norm
            # for me, max value norm is most intuitive here
            # normalizing each feature ∆value (axis=0)
            if norm:
                ml_input = sklearn.preprocessing.normalize(ml_input, norm="max", axis=0)

        # optionally save to file
        if self.savefile:
            # add labels as last col
            ml_input = np.hstack((ml_input, seg_labels.reshape(-1,1)))
            # update feature names list
            col_names = self.feat_names + ["label"]
            # convert to pandas df and save
            df = pd.DataFrame(ml_input, columns=col_names)
            # save tsv and don't include the index/row values
            df.to_csv(self.savefile, sep="\t", index=False)

        return ml_input, seg_labels

if __name__ == "__main__":
    # ml = ML_Input_Gen(h5="data/1d_v06.h5", first_iter=400, last_iter=410,
    #                   skip_feats=["M1W184_M2_DMAT", "M2W184_M1_DMAT"],
    #                   savefile="ml_input/ml_input_v06.tsv")
    # ml.create_ml_input()

    # skip all except dmatrix (TODO: make into an arg)
    skip = ['pcoord', '1_75_39_c2', 'M1E175_M1T148', 'M1E175_M2W184', 'M1M2_COM', 'M1M2_L46', 'M1W184_M2_DMAT', 'M1_E175_chi1', 'M1_E175_chi2', 'M1_E175_chi3', 'M1_E175_phi', 'M1_E175_psi', 'M2E175_M1W184', 'M2E175_M2T148', 'M2_E175_chi1', 'M2_E175_chi2', 'M2_E175_chi3', 'M2_E175_phi', 'M2_E175_psi', 'angle_3pt', 'com_dist', 'inter_nc', 'inter_nnc', 'intra_nc', 'intra_nnc', 'm1_sasa_mdt', 'm2_sasa_mdt', 'min_dist', 'rms_184_185', 'rms_bb_nmr', 'rms_bb_xtal', 'rms_dimer_int_nmr', 'rms_dimer_int_xtal', 'rms_h9m1_nmr', 'rms_h9m1_xtal', 'rms_h9m2_nmr', 'rms_h9m2_xtal', 'rms_heavy_nmr', 'rms_heavy_xtal', 'rms_key_int_nmr', 'rms_key_int_xtal', 'rms_m1_nmr', 'rms_m1_xtal', 'rms_m2_nmr', 'rms_m2_xtal', 'rog', 'rog_cut', 'secondary_struct', 'total_sasa', 'total_sasa_mdt']

    # trying it with W184 distance matrix only
    ml = ML_Input_Gen(h5="data/1d_v06.h5", first_iter=400, last_iter=410,
                      skip_feats=skip,
                      savefile="ml_input/ml_input_v06_dmat.tsv")
    #ml.create_ml_input()
