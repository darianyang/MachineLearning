"""
Some helper functions to create the initial dataset and run ML.
TODO: Eventually, integrate into wedap.
"""

import numpy as np
import h5py

import scipy.optimize
import sklearn.metrics
from tqdm.auto import tqdm
import sys

class ML_Pcoord:
    def __init__(self, h5="data/ctd_ub_1d_v00.h5", last_iter=137, ml_input=None, seg_labels=None):
        """
        Methods to generate weights for a machine learning based pcoord from a west.h5 file.

        Parameters
        ----------
        h5 : str
            Path to west.h5 file. (TODO: update to be general)
        last_iter : int
            The iteration to consider for data extraction.
            Using 137 based on the ctd_ub_1d_v00 dataset. (TODO: update to be general)
        # TODO: option to skip certain auxdata features (e.g. secondary structure)
        ml_input : str
            Path to input data file if already made, if present, does not generate new ml_input.
            Must also have seg_labels input file.
        seg_labels : str
            Path to input label file if already made, if present, does not generate new seg_labels.
            Must also have ml_input input file.
        """
        # import west.h5 data file
        self.h5 = h5py.File(h5, "r")
        self.last_iter = last_iter
        self.ml_input = ml_input
        self.seg_labels = seg_labels

        # n segments per iteration
        self.n_particles = self.h5["summary"]["n_particles"]
        # for the specfied iteration
        self.segs_in_last = self.n_particles[last_iter-1]

        # save tau value (columns of pcoord array)
        self.tau = np.atleast_3d(np.array(self.h5[f"iterations/iter_{last_iter:08d}/pcoord"])).shape[1]

        # n features per iteration (must be constant for all iterations)
        self.n_features = len(list(self.h5[f"iterations/iter_{last_iter:08d}/auxdata"]))
        # and every pcoord tracked (n depth)
        n_pcoords = np.atleast_3d(np.array(self.h5[f"iterations/iter_{last_iter:08d}/pcoord"])).shape[2]
        self.n_features += n_pcoords

        # get the names of each feature
        self.feat_names = [f"pcoord_{dim}" for dim in range(n_pcoords)] + \
                           list(self.h5[f"iterations/iter_{last_iter:08d}/auxdata"])

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
        # And trace it
        while it > 1: 
            it, wlk = self.get_parents((it, wlk))
            path.append((it,wlk))
        return np.array(sorted(path, key=lambda x: x[0]))

    def create_ml_input(self, label_space=["pcoord_0", "gt", 37], savefile=None):
        """
        Need a dataset from west.h5 as follows:
        rows: 1 for each segment of --last-iter, traced back to bstate to rep each iteration
            so if --last-iter is 100 and has 127 segments, 127 rows
        cols: 1 for each ∆iteration, last_frame - first_frame, with col for each feature
            so n iteations by n features

        Parameters
        ----------
        label_space : list
            List of 3 elements: [(feat_name), (gt or lt), (float or int)]
            This determines which segments are labeled as True.
            e.g. label_space=["pcoord_0", "gt", 37] (TODO: update to be general)
                 every seg with pcoord_0 value > 37 will be counted as True.
        savefile : str
            Optional path to save the output ml_input array as a tsv.
            Saves the features as X_{savefile} and classifications as y_{savefile}.

        Returns
        -------
        ml_input : 2d array
            Array of features for each segment in specified last_iter.
        seg_labels : 1d array
            Array of binary labels with successful trajectories as True.
        """
        # initialize empty but correctly shaped array
        ml_input = np.zeros((self.segs_in_last, self.last_iter * self.n_features))
        # as expected, it is 24 by (137i * (47 aux + 2 pcoord) features) = 6713 rows

        # 1d empty array for binary y labels
        seg_labels = np.zeros((self.segs_in_last))

        ### fill the empty array ###
        # loop each segment index of the last_iter choosen
        for seg_i in tqdm(range(self.segs_in_last), desc="Creating input array"):
            # get trace path back to bstate
            trace = self.trace_walker((self.last_iter, seg_i))

            # for labeling as T or F, start as F
            label = 0

            # take trace path and loop each (it, wlk) item (begins at i1)
            for it, wlk in trace:
                # loop each feature name
                for feat_i, feat in enumerate(self.feat_names):
                    
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
                    it_wlk_feat_data = np.atleast_2d(self.h5[f"iterations/iter_{it:08d}/{feat_name}"][wlk])
                    # properly shape the array for ndims of auxdata
                    if feat_name[:7] == "auxdata":
                        # (if tau=11) goes from row to column (1, 11) to (11, 1); pcoord is (11, n)
                        it_wlk_feat_data = it_wlk_feat_data.reshape(self.tau, 1)

                    # matching the input feature name for label cutoff
                    if feat == label_space[0]:
                        # if a certain cutoff is met, set label as true (1), otherwise false (0)
                        if label_space[1] == "gt":
                            if np.any(it_wlk_feat_data[:,feat_depth] > label_space[2]):
                                label = 1
                        elif label_space[1] == "lt":
                            if np.any(it_wlk_feat_data[:,feat_depth] < label_space[2]):
                                label = 1
                        else:
                            raise ValueError(f"label_space[1] must be 'gt' or 'lt', not {label_space[1]}")

                    # ∆feat = |last frame - first frame|
                    d_feat = np.absolute(it_wlk_feat_data[:, feat_depth][-1] - 
                                         it_wlk_feat_data[:, feat_depth][0])

                    # assign values of ml_input array (row=seg_i, col=feat_i+((it-1)*n_features))
                    ml_input[seg_i, feat_i + ((it - 1) * self.n_features)] = d_feat

            # label the segment
            seg_labels[seg_i] = label

        # optionally save to file
        if savefile:
            np.savetxt("X_" + savefile, ml_input, delimiter="\t")
            np.savetxt("y_" + savefile, seg_labels, delimiter="\t")

        # TODO: may need to reshape the seg_labels
        return ml_input, seg_labels

    def calc_seg_scores(self, iter_w, feat_w):
        """
        A score calculation method for min objective functions.
        TODO: eventually can try other segment scoring (e.g. rank instead of average)
              and other loss metrics, e.g. pROCAUC instead of ROCAUC.

        Parameters
        ----------
        iter_w : 1d array
            Weights for each iteration.
        feat_w : 1d array
            Weights for each feature.

        Returns
        -------
        seg_scores : 1d array
            Array of weighted averages for each segment/row.
        """
        # map set of n_features for each iter to n_iter scores using feat_w weights
        # ml_input (n_segs rows and n_iters * n_features cols) --> n_segs rows and n_iters cols

        # weight each iter using iter_w weights
        
        # calculate the weighted average score of each segment/row
        
        # return the 1d array of segment scores
        pass

    def iter_f(self, iter_w, feat_w):
        """
        Objective function to be minimized for iteration weight optimization.

        Parameters
        ----------
        iter_w : 1d array
            Weights to be optimized for each iteration.
            (minimization variable)
        feat_w : 1d array
            Static weights for each feature.

        Returns
        -------
        rocauc : float
            -ROCAUC score using the input weights.
            Negative since being minimized, here ROCAUC must be maximized.
        """
        # map each row of self.ml_input to a weighted average score per row/segment
        seg_scores = self.calc_seg_scores(iter_w, feat_w)

        # calc rocauc value and return the negative (min negative to maximize rocauc)
        rocauc = sklearn.metrics.roc_auc_score(self.seg_labels, seg_scores)
        return -rocauc

    def feat_f(self, feat_w, iter_w):
        """
        Objective function to be minimized for feature weight optimization.

        Parameters
        ----------
        feat_w : 1d array
            Weights to be optimized for each feature.
            (minimization variable)
        iter_w : 1d array
            Static weights for each iteration.

        Returns
        -------
        rocauc : float
            -ROCAUC score using the input weights.
            Negative since being minimized, here ROCAUC must be maximized.
        """
        # map each row of self.ml_input to a weighted average score per row/segment
        seg_scores = self.calc_seg_scores(iter_w, feat_w)

        # calc rocauc value and return the negative (min negative to maximize rocauc)
        rocauc = sklearn.metrics.roc_auc_score(self.seg_labels, seg_scores)
        return -rocauc

    def optimize_pcoord(self, recycle=3):
        """
        Main public class method
        ------------------------
        Optimizes a linear combination of pcoords and returns weights
        for each input feature. These weights can be used to calculate
        a high dimensional pcoord during a westpa run.

        Parameters
        ----------
        recycle : int
            Number of rounds or cycles of iter then feat minimization.

        Returns
        -------
        iter_w : 1d array
            Final optimized weights for each iteration.
        feat_w : 1d array
            Final optimized weights for each feature.
        """
        # don't create the ml_input array if provided
        if self.ml_input is None or self.seg_labels is None:
            self.ml_input, self.seg_labels = self.create_ml_input()
        else:
            self.ml_input = np.loadtxt(self.ml_input)
            self.seg_labels = np.loadtxt(self.seg_labels)

        # get constant starting weights
        iter_w = np.array([1/self.last_iter for _ in range(self.last_iter)])
        feat_w = np.array([1/self.n_features for _ in range(self.n_features)])

        # implement bounds for each scalar in output array (0-1)
        iter_bounds = tuple((0,1) for _ in range(self.last_iter))
        feat_bounds = tuple((0,1) for _ in range(self.n_features))

        # implement equality constraint (equals 0): sum of output array = 1
        constraints = ({"type": "eq", "fun": lambda x: np.sum(x) -1})

        # eps = step size used for estimation of jacobian in minimization
        # eps must be large enough to get out of local mimima for SLSQP
        # TODO: do something like stochastic GD where there is a variety of step sizes
        options = {"eps": 1}
        
        # repeat iter then feat weight minimization n times
        for cycle in range(recycle):
            # first optimize iteration weights
            # SLSQP local minimization method for each scalar in output array 
            iter_min = scipy.optimize.minimize(self.iter_f, iter_w, 
                                               constraints=constraints, args=(feat_w),
                                               bounds=iter_bounds, options=options)
            
            # set new weights var to be output array of minimization
            iter_w = iter_min.x
            # var to compare each scoring function
            iter_loss = iter_min.fun

            # then optimize feature weighs using optimized iteration weights
            # SLSQP local minimization method for each scalar in output array 
            feat_min = scipy.optimize.minimize(self.feat_f, feat_w, 
                                               constraints=constraints, args=(iter_w),
                                               bounds=feat_bounds, options=options)
            
            # set new weights var to be output array of minimization
            feat_w = feat_min.x
            # var to compare each scoring function
            feat_loss = feat_min.fun

            print("-------------------------------------------------------------------")
            print(f"CYCLE: {cycle} | ITER LOSS: {-iter_loss} | FEAT LOSS: {-feat_loss}")
            print("-------------------------------------------------------------------")

        return iter_w, feat_w

# eventually for optimized weight tracking, make dict of aux names
# so that you can output which features or iterations have what weight

if __name__ == "__main__":
    # ml = ML_Pcoord()
    # ml.create_ml_input(savefile="ml_input.tsv")

    ml = ML_Pcoord(ml_input="X_ml_input.tsv", seg_labels="y_ml_input.tsv")
    ml.optimize_pcoord()