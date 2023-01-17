"""
ML_Pcoord class to create the initial dataset and run weight optimization.
TODO: Eventually, integrate into wedap.
"""

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import scipy.optimize
import sklearn.metrics
import sklearn.preprocessing
import sklearn.model_selection

from sklearn import linear_model
from sklearn import ensemble

plt.style.use("/Users/darian/github/wedap/wedap/styles/default.mplstyle")

# Suppress divide-by-zero in log
np.seterr(divide="ignore", invalid="ignore")

class ML_Pcoord:
    def __init__(self, ml_input):
        """
        Methods to generate weights for a machine learning based pcoord from a west.h5 file.

        # TODO: option to change standardization/normalization type
        #       e.g. max, l1, l2, std, None ; 'delta', 'None'
        #       question is, do this here or in ml_input_gen?

        Parameters
        ----------
        ml_input : str
            Path to input data file.
        """
        # import ml_input data
        df = pd.read_csv(ml_input, sep="\t")
        
        # array of every column except for last (which are labels)
        self.ml_input = df.iloc[:,:-1].to_numpy()

        # last column is the seg labels
        self.seg_labels = df.iloc[:,-1].to_numpy(int)

        # save column names separately
        self.feat_names = df.iloc[:,:-1].columns.to_numpy()
        self.n_features = self.feat_names.shape[0]

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

        # implement equality constraint (equals 0): sum of output array = 1
        constraints = ({"type": "eq", "fun": lambda x: np.sum(x) -1})
        #constraints = scipy.optimize.NonlinearConstraint(lambda x: np.sum(x), 1, 1)
        
        if plot:
            fig, ax = plt.subplots()

        # TODO: add noise testing:
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
                    fpr, tpr, _ = sklearn.metrics.roc_curve(self.seg_labels,
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
                fpr, tpr, _ = sklearn.metrics.roc_curve(self.seg_labels,
                                                                 self.feat_weighted)
                self.plot_roc_curve(fpr, tpr, -feat_loss, f"POST | ", ax)

        if plot:
            fig.tight_layout()
            #plt.show()
            #plt.savefig("roc.png", dpi=300, transparent=True)

        # save weights
        self.feat_w = feat_w

        # return the scipy min object
        return feat_min

    def split_score(self, model=None, score="auc", confusion=False, opt_weights=False):
        """
        Split ml_input into test/train datasets, opt weights using training data, 
        and use those weights or the input model to predict on test data.

        Parameters
        ----------
        model : sklearn model object
        score : str
            'auc', 'f1', 'acc', 'bacc'
        confusion : bool
            True to compute and print the confusion matrix.
            Note, only possible with binary classification model, not with weight probabilites.
        opt_weights : bool
            Optionally opt weights and then use input model.

        Returns
        -------
        metric : float
            Score for either the weight optimization or input model.
        """
        # split dataset (stratify to include equal True in splits)
        X_train, X_test, y_train, y_test = \
            sklearn.model_selection.train_test_split(self.ml_input, self.seg_labels, 
                                                     test_size=0.8, stratify=self.seg_labels)

        # run weight gradient opt on training 
        if model is None or opt_weights is True:
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
                log_fit = linear_model.LogisticRegression().fit(self.feat_weighted.reshape(-1, 1), 
                                                                self.seg_labels)
                y_pred = log_fit.predict(y_pred.reshape(-1, 1))

        # otherwise, try other models besides my opt model (RF, logistic, etc)
        if model is not None:
            if opt_weights:
                X_train = np.multiply(X_train, self.feat_w)
                X_test = np.multiply(X_test, self.feat_w)
            self.model = model.fit(X_train, y_train)
            # estimate test labels
            y_pred = self.model.predict(X_test)

        if score == "auc":
            metric = sklearn.metrics.roc_auc_score(y_test, y_pred)
            fpr, tpr, _ = sklearn.metrics.roc_curve(y_test, y_pred)
            self.plot_roc_curve(fpr, tpr, metric)
        elif score == "f1":
            metric = sklearn.metrics.f1_score(y_test, y_pred)
        elif score == "acc":
            metric = sklearn.metrics.accuracy_score(y_test, y_pred)
        elif score == "bacc":
            metric = sklearn.metrics.balanced_accuracy_score(y_test, y_pred)

        # print confusion matrix
        if confusion:
            print("Confusion Matrix (tn, fp, fn, tp):")
            print(sklearn.metrics.confusion_matrix(y_test, y_pred).ravel())

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

        Returns
        -------
        sorted_top_n_features : array
            top_n features with weights, sorted by decreasing weight.
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
        #plt.savefig("weights.png", dpi=300, transparent=True)

        top = np.argpartition(self.feat_w, -top_n)[-top_n:]
        
        # sort by weight and return top n
        return sorted(zip(np.array(self.feat_names)[top], self.feat_w[top]), 
                                   key=lambda t: t[1], reverse=True)

    def count_tf(self):
        """
        Print amount of T/F labels in dataset.
        """
        # count how many True
        t = np.count_nonzero(self.seg_labels)

        # count how many False
        f = np.count_nonzero(self.seg_labels==0)

        print(f"TRUE: {t} | FALSE: {f}")

if __name__ == "__main__":
    ml = ML_Pcoord("ml_input/ml_input_v06_dmatfull2.tsv")
    ml.optimize_pcoord(plot=True)
    top = ml.plot_weights(15)
    print(top)
    ml.count_tf()
    plt.show()

    # save weights to file
    #np.savetxt("M2W184_M1_DMAT_weights.txt", ml.feat_w, delimiter="\t")
    np.savetxt("M1W184_M2_DMAT_weights.txt", ml.feat_w, delimiter="\t")