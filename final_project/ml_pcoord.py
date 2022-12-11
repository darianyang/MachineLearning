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
import sklearn.model_selection

from sklearn import linear_model
from sklearn import ensemble

from tqdm.auto import tqdm
import sys

plt.style.use("/Users/darian/github/wedap/wedap/styles/default.mplstyle")

# Suppress divide-by-zero in log
np.seterr(divide="ignore", invalid="ignore")

class ML_Pcoord:
    def __init__(self, h5=None, first_iter=1, last_iter=None, savefile=None,
                 ml_input=None, seg_labels=None, skip_feats=None):
        """
        Methods to generate weights for a machine learning based pcoord from a west.h5 file.

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
        ml_input : str
            Path to input data file if already made, if present, does not generate new ml_input.
            Must also have seg_labels input file.
        seg_labels : str
            Path to input label file if already made, if present, does not generate new seg_labels.
            Must also have ml_input input file.
        skip_feats : list
            List of str feature names to not include in ml_input dataset.
        """
        # import west.h5 data file
        self.h5 = h5py.File(h5, "r")
        self.first_iter = first_iter
        if last_iter:
            self.last_iter = last_iter
        else:
            self.last_iter = self.h5.attrs["west_current_iteration"] - 1
        self.ml_input = ml_input
        self.seg_labels = seg_labels

        # n segments per all iterations
        self.n_particles = self.h5["summary"]["n_particles"]
        # for the specfied iterations
        self.total_segs = np.sum(self.n_particles[self.first_iter-1:self.last_iter])

        # save tau value (columns of pcoord array)
        self.tau = np.atleast_3d(np.array(self.h5[f"iterations/iter_{self.last_iter:08d}/pcoord"])).shape[1]

        # n features per iteration (must be constant for all iterations)
        self.n_features = len(list(self.h5[f"iterations/iter_{self.last_iter:08d}/auxdata"]))
        # and every pcoord tracked (n depth)
        n_pcoords = np.atleast_3d(np.array(self.h5[f"iterations/iter_{self.last_iter:08d}/pcoord"])).shape[2]
        self.n_features += n_pcoords

        # get the names of each feature (and multiple pcoords)
        self.feat_names = [f"pcoord_{dim}" for dim in range(n_pcoords)] + \
                           list(self.h5[f"iterations/iter_{self.last_iter:08d}/auxdata"])

        # optionally skip certain specified features (TODO)
        if skip_feats:
            self.skip_feats = skip_feats
            # subtract the skipped feats from total n_features
            self.n_features -= len(skip_feats)
            # remove skipped feats from feature name list
            self.feat_names = [i for i in self.feat_names if i not in self.skip_feats]

        # optionally save ml_input
        self.savefile = savefile

        # don't create the ml_input array if provided with filepath str
        if self.ml_input is None or self.seg_labels is None:
            self.ml_input, self.seg_labels = self.create_ml_input()
        elif isinstance(self.ml_input, str):
            self.ml_input = np.loadtxt(self.ml_input)
            self.seg_labels = np.loadtxt(self.seg_labels)

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
            
    def create_ml_input(self, label_space=None, random=False):
        """
        Generate ml dataset from west.h5.

        Parameters
        ----------
        label_space : list
            List of 3 elements: [(feat_name), (gt or lt), (float or int)]
            This determines which segments are labeled as True.
            e.g. label_space=["pcoord_0", "gt", 37] (TODO: update to be multi-dim)
                 every seg with pcoord_0 value > 37 will be counted as True.
            With None, use the recycled trajectories from west.h5.
        random : bool
            Default False, if True, returns an output file of random values with 50/50 T/F labels.

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

            # TODO: if succ_traces is None:
                # raise error "No successfull trajectories found, please input a label_space criterion"
            # unique (iter,seg) pairs from each trace of each succ_traj (iter,seg) pair
            succ_traces = np.unique(np.concatenate(succ_traces), axis=0)
        else:
            raise ValueError("label_space arg is not working yet...")

        # TODO: output a random dataset for testing the optimzation
        if random:
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
                    # TODO: testing labeling only recycle (iter,seg) or whole trace
                    # only recycle seems to be better for now
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
            ml_input = sklearn.preprocessing.normalize(ml_input, norm="max", axis=0)

        # optionally save to file
        if self.savefile:
            np.savetxt("X_" + self.savefile, ml_input, delimiter="\t")
            np.savetxt("y_" + self.savefile, seg_labels, delimiter="\t")

        return ml_input, seg_labels

    def loss_f(self, feat_w):
        """
        A score calculation method for min objective functions.
        TODO: eventually can try other segment scoring (e.g. rank instead of average)
              and other loss metrics, e.g. pROCAUC instead of ROCAUC.
              Maybe use variation from wevo as the loss function
              where variation of the system made by the distance matrix of all features (standardized)
              preweighted before calculating distance matrix. Variation would be maximized.

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

        # TODO: also try time series test train split and eval with rocauc
        # note that splitting with equal weights in test and train may lead to better scores

        # calc rocauc value and return the negative (min negative to maximize rocauc)
        score = sklearn.metrics.roc_auc_score(self.seg_labels, self.feat_weighted)
        return -score

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
        # get constant starting weights
        feat_w = np.array([1/self.n_features for _ in range(self.n_features)])
        # random vs uniform initial guess array?
        #feat_w = np.random.dirichlet(np.ones(self.n_features),size=1).reshape(-1)

        # implement bounds for each scalar in output array (0-1)
        feat_bounds = tuple((0,1) for _ in range(self.n_features))

        # TODO: need to think about this, do I really need sum=1 constraint? Not necessarily
        # implement equality constraint (equals 0): sum of output array = 1
        constraints = ({"type": "eq", "fun": lambda x: np.sum(x) -1})
        #constraints = scipy.optimize.NonlinearConstraint(lambda x: np.sum(x), 1, 1)
        
        if plot:
            fig, ax = plt.subplots()

        # TODO: add noise test:
        # from Tiwary PIB paper, gaussian noise was added at 0.05 variance (sigma**2)= 0.05 stdev (sigma)
        # this was close to the initial weights
        # self.ml_input = np.multiply(self.ml_input, np.random.normal(np.average(self.ml_input), 
        #                                                             feat_w[0]/10,
        #                                                             #np.sqrt(0.05), # 0.22 
        #                                                             self.ml_input.shape[1]))
        # TODO: maybe I can also use the full 100ps tau for columns and not use ∆values
            # a more natural way to add "noise"

        # repeat iter then feat weight minimization n times
        for cycle in range(recycle):
            # eps = step size used for estimation of jacobian in minimization
            # eps must be large enough to get out of local mimima for SLSQP
            # gradually decrease step size per cycle
            options = {"eps": 1**-cycle}
            #options = {"eps": 1**(-8+cycle)}
            #options = {"eps": 1}
            # default (sqrt of machine epsilon for float64)
            #options = {"eps": 1.4901161193847656e-08}

            # only for first cycle
            if cycle == 0:
                print("--------------------------------------------")
                print(f"CYCLE: {cycle} | PRE LOSS: {-self.loss_f(feat_w)}")
                print("--------------------------------------------")
                if plot:
                    fpr, tpr, thresholds = sklearn.metrics.roc_curve(self.seg_labels,
                                                                    self.feat_weighted)
                    self.plot_roc_curve(fpr, tpr, -self.loss_f(feat_w), f"PRE | ", ax)

            # then optimize feature weighs using optimized iteration weights
            # SLSQP local minimization method for each scalar in output array 
            feat_min = scipy.optimize.minimize(self.loss_f, feat_w, 
                                               constraints=constraints,
                                               bounds=feat_bounds, options=options)

            # the function may not be smooth so trying a stochastic minimization (non-gradient based)
            # feat_min = scipy.optimize.differential_evolution(self.loss_f, x0=feat_w, 
            #                                                  #constraints=constraints,
            #                                                  bounds=feat_bounds, maxiter=10)

            # set new weights var to be output array of minimization
            feat_w = feat_min.x
            # var to compare each scoring function
            feat_loss = feat_min.fun

            #print(feat_min)
            print("--------------------------------------------")
            print(f"CYCLE: {cycle} | POST LOSS: {-feat_loss}")
            print("--------------------------------------------")
            if plot:
                fpr, tpr, thresholds = sklearn.metrics.roc_curve(self.seg_labels,
                                                                 self.feat_weighted)
                self.plot_roc_curve(fpr, tpr, -feat_loss, f"POST | ", ax)

        if plot:
            fig.tight_layout()
            plt.show()
            #plt.savefig("roc.png", dpi=300, transparent=True)
        self.feat_w = feat_w
        #return feat_w
        # return the scipy min object
        return feat_min

    def split_score(self, model=None, score="auc"):
        """
        Split ml_input into test/train datasets, opt weights using training data, 
        and use those weights or the input model to predict on test data.

        Parameters
        ----------
        model : sklearn model object
        score : str
            'auc', 'f1', 'acc', 'bacc'

        Returns
        -------
        metric : float
            Score for either the weight optimization or input model.
        """
        # split dataset (stratify to include equal True in splits)
        X_train, X_test, y_train, y_test = \
            sklearn.model_selection.train_test_split(self.ml_input, self.seg_labels, test_size=0.8, stratify=self.seg_labels)

        # run weight gradient opt on training 
        if model is None:
            # set ml_input to be training set
            self.ml_input = X_train
            self.seg_labels = y_train

            # optimize weights on training data
            self.optimize_pcoord()
            
            # use optimized weights to score the test data
            y_pred = np.average(X_test, weights=self.feat_w, axis=1)

            # using arbitray logistic mapping since non auc metrics can't use probability
            if score != "auc":
                # fit to binary for scoring
                log_fit = linear_model.LogisticRegression().fit(self.feat_weighted.reshape(-1, 1), self.seg_labels)
                y_pred = log_fit.predict(y_pred.reshape(-1, 1))

        # otherwise, try other models besides my opt model (RF, logistic, etc)
        else:
            self.model = model.fit(X_train, y_train)
            # estimate test labels
            y_pred = self.model.predict(X_test)

        if score == "auc":
            metric = sklearn.metrics.roc_auc_score(y_test, y_pred)
            fpr, tpr, thresholds = sklearn.metrics.roc_curve(y_test, y_pred)
            self.plot_roc_curve(fpr, tpr, metric)
            plt.show()
        elif score == "f1":
            metric = sklearn.metrics.f1_score(y_test, y_pred)
        elif score == "acc":
            metric = sklearn.metrics.accuracy_score(y_test, y_pred)
        elif score == "bacc":
            metric = sklearn.metrics.balanced_accuracy_score(y_test, y_pred)

        #print(np.testing.assert_array_equal(y_test, y_pred))

        return metric

    def plot_roc_curve(self, x, y, score, label="", ax=None):
        """
        Function for plotting the reciever operator characteristic curve 
        with the X axis as the false positive rate and the y axis as the 
        true positive rate.
        """
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = plt.gcf()
        ax.plot(x, y, label=f"{label}AUC: {score:0.3f}")
        ax.plot([0, 1], [0, 1], color="k", linestyle="--")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("Receiver Operating Characteristic (ROC) Curve", fontsize=16)
        ax.legend()
    
    def plot_weights(self, top_n=10, ax=None, weights=None):
        """
        Function for plotting the optimized weights of each feature.

        Parameters
        ----------
        top : int
            The amount of top features to return.
        ax : mpl axes object
        weights : array
            Optionally input weights or feature importances.
            By default will use self.feat_w.
        """
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = plt.gcf()

        # for substituting with e.g. rg feature importances
        if weights is not None:
            self.feat_w = weights

        ax.scatter([i for i in range(self.feat_w.shape[0])], self.feat_w)
        ax.set_xlabel("Feature")
        ax.set_ylabel("Weight")
        ax.set_title("Weight per Feature")
        fig.tight_layout()
        plt.savefig("weights.png", dpi=300, transparent=True)
        #plt.show()

        top = np.argpartition(self.feat_w, -top_n)[-top_n:]
        
        # sort by weight and return top n
        return sorted(zip(np.array(self.feat_names)[top], self.feat_w[top]), key=lambda t: t[1], reverse=True)


if __name__ == "__main__":
    ### making a few datasets ###
    # this dataset didn't have good results, too easy of a classification problem
    # ml = ML_Pcoord(h5="data/ctd_ub_1d_v00.h5", first_iter=10, last_iter=160)
    # ml.create_ml_input(savefile="ml_input.tsv")

    # this was a better dataset, more variety of trajectories
    # ml = ML_Pcoord(h5="data/ctd_ub_1d_v04.h5")
    # ml.create_ml_input(savefile="ml_input.tsv")
    
    # without the pcoord_1 and min_dist datasets (which define the recycle boundary)
    # ml = ML_Pcoord(h5="data/ctd_ub_1d_v04.h5", skip_feats=["pcoord_1", "min_dist"])
    # ml.create_ml_input(savefile="ml_input_cut.tsv")

    # random test dataset
    # ml = ML_Pcoord(h5="data/ctd_ub_1d_v04.h5")
    # ml.create_ml_input(savefile="ml_input_rand.tsv", random=True)

    ### eda ###
    # find and output all successfull trajectories
    # ml = ML_Pcoord(h5="data/ctd_ub_1d_v04.h5")
    # succ = ml.w_succ()
    # print(succ)

    # load dataset
    # X = np.loadtxt("X_ml_input.tsv")
    # y = np.loadtxt("y_ml_input.tsv")
    
    # count how many True
    #print(np.count_nonzero(y))

    # count how many False
    # print(np.count_nonzero(y==0))

    # random dataset plot
    # plt.plot(X[120])
    # plt.show()

    ### rocauc plot and opt all feats ###
    # ml = ML_Pcoord(h5="data/ctd_ub_1d_v04.h5", ml_input="X_ml_input.tsv", seg_labels="y_ml_input.tsv")
    # names = np.array(ml.feat_names)
    # fw = ml.optimize_pcoord(plot=True, recycle=1)
    # top = ml.plot_weights()
    # print(top)
    # plt.show()

    ### rocauc plot and opt with skip_feats | also testing with and without std/norm ###
    ### from tests, going to go with no standardization and max vector based norm ###
    # ml = ML_Pcoord(h5="data/ctd_ub_1d_v04.h5", ml_input="X_ml_input_cut.tsv", seg_labels="y_ml_input_cut.tsv", 
    #                skip_feats=["pcoord_1", "min_dist"])
    # names = np.array(ml.feat_names)
    # fw = ml.optimize_pcoord(plot=True, recycle=1)
    # # seems like there are 6 non-near-zero features from the plot
    # top = ml.plot_weights(top_n=10)
    # print(top)
    # plt.show()

    # TODO: test/train split, run cv, and calc confusion matrix
    # also try random forest to compare feature importance and opt weights
    # mention in nb that using weights, can get probability estimates for binary classification
    # of new segments not in training data

    ### random dataset ###
    # ml = ML_Pcoord(h5="data/ctd_ub_1d_v04.h5", ml_input="X_ml_input_rand.tsv", seg_labels="y_ml_input_rand.tsv")
    # names = np.array(ml.feat_names)
    # fw = ml.optimize_pcoord(plot=True, recycle=1)
    # top = ml.plot_weights(top_n=10)
    # print(top)
    # plt.show()

    ### test/train split and validate weight opt ###
    # for both the easy and difficult dataset, RF model is not as good as gradient descent method
    # ml = ML_Pcoord(h5="data/ctd_ub_1d_v04.h5", ml_input="X_ml_input_cut.tsv", seg_labels="y_ml_input_cut.tsv", 
    #                skip_feats=["pcoord_1", "min_dist"])
    
    # trying with v00 instead since it had proper recycling
    #ml = ML_Pcoord(h5="data/ctd_ub_1d_v00.h5", savefile="ml_input_v01.tsv")
    ml = ML_Pcoord(h5="data/ctd_ub_1d_v00.h5", ml_input="X_ml_input_v01.tsv", seg_labels="y_ml_input_v01.tsv")
    
    # score = ml.split_score(score="auc")
    # print(ml.plot_weights())

    score = ml.split_score(ensemble.RandomForestClassifier(oob_score=True), score="auc")
    print(f"OOB: {ml.model.oob_score_}")
    print(ml.plot_weights(weights=ml.model.feature_importances_))

    plt.show()
    print(score)

    # TODO: run some RF random and grid search for hyperparameter opt

    ### trying with more positive labels from history ### (TODO)
    # ml = ML_Pcoord(h5="data/ctd_ub_1d_v04.h5")
    # ml.create_ml_input(savefile="ml_input_cut.tsv")
    #ml = ML_Pcoord(h5="data/ctd_ub_1d_v04.h5", ml_input="X_ml_input_cut.tsv", seg_labels="y_ml_input_cut.tsv")
    #score = ml.split_score(score="auc")
    #print(score)