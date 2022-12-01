"""
Some helper functions to create the initial dataset and run ML.
TODO: Eventually, integrate into wedap.
"""

import numpy as np
import h5py

import matplotlib.pyplot as plt
import scipy.optimize
import sklearn.metrics
import sklearn.preprocessing
from tqdm.auto import tqdm
import sys

# Suppress divide-by-zero in log
np.seterr(divide="ignore", invalid="ignore")

class ML_Pcoord:
    def __init__(self, h5="data/ctd_ub_1d_v00.h5", first_iter=10, last_iter=160,
                 ml_input=None, seg_labels=None):
        """
        Methods to generate weights for a machine learning based pcoord from a west.h5 file.

        # TODO: option to skip certain auxdata features (e.g. secondary structure)
        Parameters
        ----------
        h5 : str
            Path to west.h5 file. (TODO: update to be general)
        first_iter : int
            The lower bound iteration to consider for data extraction.
        last_iter : int
            The upper bound iteration to consider for data extraction.
            Using 160 based on the ctd_ub_1d_v00 dataset. (TODO: update to be general)
        ml_input : str
            Path to input data file if already made, if present, does not generate new ml_input.
            Must also have seg_labels input file.
        seg_labels : str
            Path to input label file if already made, if present, does not generate new seg_labels.
            Must also have ml_input input file.
        """
        # import west.h5 data file
        self.h5 = h5py.File(h5, "r")
        self.first_iter = first_iter
        self.last_iter = last_iter
        self.ml_input = ml_input
        self.seg_labels = seg_labels

        # n segments per all iterations
        self.n_particles = self.h5["summary"]["n_particles"]
        # for the specfied iterations
        self.total_segs = np.sum(self.n_particles[self.first_iter-1:self.last_iter])

        # save tau value (columns of pcoord array)
        self.tau = np.atleast_3d(np.array(self.h5[f"iterations/iter_{last_iter:08d}/pcoord"])).shape[1]

        # n features per iteration (must be constant for all iterations)
        self.n_features = len(list(self.h5[f"iterations/iter_{last_iter:08d}/auxdata"]))
        # and every pcoord tracked (n depth)
        n_pcoords = np.atleast_3d(np.array(self.h5[f"iterations/iter_{last_iter:08d}/pcoord"])).shape[2]
        self.n_features += n_pcoords

        # get the names of each feature (and multiple pcoords)
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

    def w_succ(self):
        """
        Find and return all successfully recycled (iter, seg) pairs.
        TODO: eventually can use this to plot pdist of succ only trajs
              note that I would have to norm by the overall pmax
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
            
    def create_ml_input(self, label_space=None, savefile=None):
        """
        Need a dataset from west.h5 as follows:
        rows: 1 for each segment of --last-iter, traced back to bstate to rep each iteration
            so if --last-iter is 100 and has 127 segments, 127 rows
        cols: 1 for each ∆iteration, last_frame - first_frame, with col for each feature
            so n iteations by n features
        # TODO: actually, try rows as every segment from each iteration ∆values
                then cols are just the n features, start with just feat_w min then
                if needed can incorporate iter min but would weight each row so might
                not be needed or might not even help.
                Use trace for labeling still? if in iter,seg in trace_path mark as True
                So only trace the feature determining label space.
                Maybe save the trace paths of all recycled trajectories and use these as 
                the reference for which iter,seg to label as successful
                This could be an alternative to using the label_space to set it more manually
                    e.g. for a WE with no recycling, so basically w_succ for equilWE
        TODO: implement --first_iter arg to cut out the initial non-important iter segs

        Parameters
        ----------
        label_space : list
            List of 3 elements: [(feat_name), (gt or lt), (float or int)]
            This determines which segments are labeled as True.
            e.g. label_space=["pcoord_0", "gt", 37] (TODO: update to be general and multi-dim)
                 every seg with pcoord_0 value > 37 will be counted as True.
            With None, use the recycled trajectories from west.h5.
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
        # rows are segs for each iter in specified range, cols are n_features
        ml_input = np.zeros((self.total_segs, self.n_features))

        # 1d empty array for binary y labels
        seg_labels = np.zeros((self.total_segs))

        # TODO: right now label_space is not ready for use: previous code:
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
        if label_space is None:
            succ_traces = []
            succ_pairs = self.w_succ()
            # make an array for all succ (iter,seg) traced paths
            for pair in succ_pairs:
                # filter for only iterations considered
                if pair[0] >= self.first_iter and pair[0] <= self.last_iter:
                    trace = self.trace_walker(pair)
                    # need to filter the trace up to specified first_iter
                    trace = trace[self.first_iter-1:]
                    succ_traces.append(trace)

            # unique (iter,seg) pairs from each trace of each succ_traj (iter,seg) pair
            succ_traces = np.unique(np.concatenate(succ_traces), axis=0)
        else:
            raise ValueError("label_space arg is not working yet...")

        # overall seg row count
        seg_n = 0
        # loop each specified iteration and walker pair to fill array
        for it in tqdm(range(self.first_iter, self.last_iter-1), desc="Creating input array"):
            for wlk in range(self.n_particles[it - 1]):
                # labeling as T if apart of succ traj paths
                # TODO: testing labeling only recycle (iter,seg) or whole trace
                #if [it, wlk] in succ_traces:
                if (it, wlk) in succ_pairs:
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

        # standardize
        ml_input = sklearn.preprocessing.StandardScaler().fit_transform(ml_input)

        # optionally save to file
        if savefile:
            np.savetxt("X_" + savefile, ml_input, delimiter="\t")
            np.savetxt("y_" + savefile, seg_labels, delimiter="\t")

        # TODO: may need to reshape the seg_labels
        return ml_input, seg_labels

    def loss_f(self, feat_w):
        """
        A score calculation method for min objective functions.
        TODO: eventually can try other segment scoring (e.g. rank instead of average)
              and other loss metrics, e.g. pROCAUC instead of ROCAUC.

        Parameters
        ----------
        feat_w : 1d array
            Weights for each feature.

        Returns
        -------
        rocauc : float
            -ROCAUC score using the input weights.
            Negative since being minimized, here ROCAUC must be maximized.
        """
        # calc feature weighted ∆values of ml_input
        self.feat_weighted = np.average(self.ml_input, weights=feat_w, axis=1)

        # TODO: testing adding random noise
        # 0 is the mean of the normal distribution you are choosing from
        # 1 is the standard deviation of the normal distribution
        # 100 is the number of elements you get in array noise
        self.feat_weighted = self.feat_weighted * np.random.normal(np.average(self.feat_weighted),
                                                                   np.std(self.feat_weighted),
                                                                   self.feat_weighted.shape[0])
        
        # calc rocauc value and return the negative (min negative to maximize rocauc)
        score = sklearn.metrics.roc_auc_score(self.seg_labels, self.feat_weighted)
        return -score

    def constraint_f(self, w):
        """
        Sum weights to 1.
        TODO: change to lambda
        """
        return np.sum(w)

    def optimize_pcoord(self, recycle=1, plot=False):
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
        plot : bool
            Whether or not to plot the roc curve.

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
        feat_w = np.array([1/self.n_features for _ in range(self.n_features)])
        # random vs uniform initial guess array?
        #feat_w = np.random.dirichlet(np.ones(self.n_features),size=1).reshape(-1)
        #print("Original Feature Weights:", feat_w)
        #print("Original Iteration Weights:", iter_w)

        # implement bounds for each scalar in output array (0-1)
        feat_bounds = tuple((0,1) for _ in range(self.n_features))

        # implement equality constraint (equals 0): sum of output array = 1
        #constraints = ({"type": "eq", "fun": lambda x: np.sum(x) -1})
        #constraints = scipy.optimize.NonlinearConstraint(lambda x: np.sum(x), -np.inf, 1)
        
        if plot:
            fig, ax = plt.subplots()

        # repeat iter then feat weight minimization n times
        for cycle in range(recycle):
            # eps = step size used for estimation of jacobian in minimization
            # eps must be large enough to get out of local mimima for SLSQP
            # gradually decrease step size per cycle
            #options = {"eps": 10**-cycle}
            #options = {"eps": 1**(-8+cycle)}
            # default
            #options = {"eps": 1.4901161193847656e-08}

            print("--------------------------------------------")
            print(f"CYCLE: {cycle} | PRE LOSS: {-self.loss_f(feat_w)}")
            print("--------------------------------------------")
            if plot:
                fpr, tpr, thresholds = sklearn.metrics.roc_curve(self.seg_labels,
                                                                 self.feat_weighted)
                self.plot_roc_curve(fpr, tpr, -self.loss_f(feat_w), ax)

            # then optimize feature weighs using optimized iteration weights
            # SLSQP local minimization method for each scalar in output array 
            # feat_min = scipy.optimize.minimize(self.loss_f, feat_w, 
            #                                    constraints=constraints,
            #                                    bounds=feat_bounds, options=options)

            # the function isn't smooth so trying a stochastic minimization (non-gradient based)
            feat_min = scipy.optimize.differential_evolution(self.loss_f, x0=feat_w, 
                                                             #constraints=constraints,
                                                             bounds=feat_bounds, maxiter=10)

            # set new weights var to be output array of minimization
            feat_w = feat_min.x
            # var to compare each scoring function
            feat_loss = feat_min.fun

            print(feat_min)
            print("--------------------------------------------")
            print(f"CYCLE: {cycle} | POST LOSS: {-feat_loss}")
            print("--------------------------------------------")
            if plot:
                fpr, tpr, thresholds = sklearn.metrics.roc_curve(self.seg_labels,
                                                                 self.feat_weighted)
                self.plot_roc_curve(fpr, tpr, -feat_loss, ax)

        if plot:
            plt.show()
        return feat_w

    def plot_roc_curve(self, x, y, score, ax=None):
        """
        Function for plotting the reciever operator characteristic curve in 2-D 
        with the X axis as the false positive rate and the y axis as the true 
        positive rate.
        """
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = plt.gcf()
        ax.plot(x, y, label=f"ROC: {score}")
        ax.plot([0, 1], [0, 1], color="k", linestyle="--")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("Receiver Operating Characteristic (ROC) Curve")
        ax.legend()

# eventually for optimized weight tracking, make dict of aux names
# so that you can output which features or iterations have what weight

if __name__ == "__main__":
    # ml = ML_Pcoord()
    # succ = ml.w_succ()
    # print(succ)

    # ml = ML_Pcoord(first_iter=10)
    # ml.create_ml_input(savefile="ml_input.tsv")

    # X = np.loadtxt("X_ml_input.tsv")
    # y = np.loadtxt("y_ml_input.tsv")
    # i10-160 = 61 True segs / 3624 for traced succ label True
    #print(np.count_nonzero(y))

    # # i10-160 = 48 True segs / 3624 for traced succ label True
    # print(np.count_nonzero(y==0))
    # plt.plot(X[120])
    # plt.show()

    ml = ML_Pcoord(ml_input="X_ml_input.tsv", seg_labels="y_ml_input.tsv")
    names = np.array(ml.feat_names)
    fw = ml.optimize_pcoord(plot=True, recycle=1)
    plt.plot(fw)
    plt.show()

    top = np.argpartition(fw, -10)[-10:]
    print(top)
    print(names[top])
    print("weights:", fw[top])
