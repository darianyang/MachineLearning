"""
Initially written 2022-10-02 by DTY.

Inspired by: 
https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
https://towardsdatascience.com/create-your-own-k-means-clustering-algorithm-in-python-d7d4c9077670
https://analyticsarora.com/k-means-for-beginners-how-to-build-from-scratch-in-python/
"""

import numpy as np
import random

class KMeans:
    """
    Custom k-means clustering function with options for alternative distance metrics.
    """
    def __init__(self, n_clusters=8, max_iter=300, n_init=5, distance_metric="euclidean"):
        """
        Parameters
        ----------
        n_clusters : int
            The number of clusters to form as well as the number of centroids to generate.
        max_iter : int
            Maximum number of iterations of the k-means algorithm for a single run.
        n_init : int
            Number of times the k-means algorithm will be run with different centroid seeds. 
            The final results will be the best output of n_init consecutive runs in terms of inertia.
        distance_metric : str (TODO)
            Distance calculations can be 'euclidean', 'jaccard', 'cosine', or 'manhattan'.
        """
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.n_init = n_init
        self.distance_metric = distance_metric

    def calc_distance(self, point, data):
        """
        Calculate the distance between point & data.
        Point has dimensions (m,), data has dimensions (n,m), and output will be of size (n,).
        """
        # TODO: if self.name == euclidian and etc
        return np.sqrt(np.sum((point - data)**2, axis=1))

    # TODO: call this euclidian and add others
    def pairwise_dist(self, x, y):
        """
        Euclidian pairwise distance matrix.

        Parameters
        ----------
        x : N x D numpy array
        y : M x D numpy array

        Returns
        -------
        dist: N x M array
            Where dist2[i, j] is the euclidean distance between x[i, :] and y[j, :].
        """
        xSumSquare = np.sum(np.square(x),axis=1)
        ySumSquare = np.sum(np.square(y),axis=1)
        mul = np.dot(x, y.T)
        dists = np.sqrt(abs(xSumSquare[:, np.newaxis] + ySumSquare-2*mul))
        return dists

    def _get_loss(self, centers, cluster_labels, X):
        """
        Here, the loss function being optimized is the inertia of the resultant clusters.

        Parameters
        ----------
        centers : KxD numpy array
            Where K is the number of clusters, and D is the dimension.
        cluster_labels : numpy array of length N 
            The cluster assignment for each point.
        X : NxD numpy array 
            The observations

        Returns
        -------
        loss : float 
            Inertia value of the current clustering iteration. 
        """
        dists = self.pairwise_dist(X, centers)
        loss = 0.0
        N, D = X.shape

        for i in range(N):
            loss = loss + np.square(dists[i][cluster_labels[i]])
        
        return loss

    def _init_centers(self, X):
        """
        Initialize the centroids, using the "k-means++" method, where a random datapoint is selected as the first,
        then the rest are initialized w/ probabilities proportional to their distances to the first.

        Parameters
        ----------
        X : NxD numpy array 
            The observations.
        
        Returns
        -------
        centroids : array
            Initial centroid positions.
        """
        # Pick a random point from train data for first centroid
        # TODO: self.centroids?
        centroids = [random.choice(X)]

        # k-means++
        for _ in range(self.n_clusters-1):
            # Calculate distances from points to the centroids
            dists = np.sum([self.calc_distance(centroid, X) for centroid in centroids], axis=0)
            
            # Normalize the distances
            dists /= np.sum(dists)
            
            # Choose remaining points based on their distances
            new_centroid_labels, = np.random.choice(range(len(X)), size=1, p=dists)
            centroids += [X[new_centroid_labels]]

        # This initial method of randomly selecting centroid starts is less effective
        # min_, max_ = np.min(X_train, axis=0), np.max(X_train, axis=0)
        # self.centroids = [uniform(min_, max_) for _ in range(self.n_clusters)]

        return np.array(centroids)

    def _update_assignment(self, centers, X):
        """
        For choosing which cluster each point should belong to. 

        Parameters
        ----------
        centers: KxD numpy array
            Where K is the number of clusters, and D is the dimension.
        X: NxD numpy array
            The observations.

        Returns
        -------
        cluster_labels: numpy array of length N 
            The cluster assignments for each point.
        """
        row, col = X.shape
        cluster_labels = np.empty([row])
        distances = self.pairwise_dist(X, centers)
        cluster_labels = np.argmin(distances, axis=1)

        return cluster_labels

    def _update_centers(self, old_centers, cluster_labels, X):
        """
        Averages all the points that belong to a given cluster. 
        This average is the new centroid of the respective cluster. 
        This function returns the array of new centers.

        Parameters
        ----------
        old_centers: KxD numpy array
            Where K is the number of clusters, and D is the dimension.
        cluster_labels : numpy array of length N 
            The cluster assignment for each point.
        X : NxD numpy array 
            The observations.
        
        Returns
        -------
        centers : K x D numpy array
            New centers where K is the number of clusters, and D is the dimension.
        """
        K, D = old_centers.shape
        new_centers = np.empty(old_centers.shape)
        for i in range(K):
            new_centers[i] = np.mean(X[cluster_labels == i], axis = 0)
        return new_centers

    def _opt_clusters(self, X, init, abs_tol=1e-16, rel_tol=1e-16, verbose=False):
        """
        Optimize clusters for one initialization of k-means.
        """
        centers = self._init_centers(X)

        for it in range(self.max_iter):
            cluster_labels = self._update_assignment(centers, X)
            centers = self._update_centers(centers, cluster_labels, X)
            loss = self._get_loss(centers, cluster_labels, X)
            
            if it:
                diff = np.abs(prev_loss - loss)
                # loss function based convergence within tolerance
                if diff < abs_tol and diff / prev_loss < rel_tol:
                    break

            prev_loss = loss
            if verbose:
                print("init %d, iter %d, loss: %.4f" % (init, it, loss))

        return cluster_labels, centers, loss

    def fit(self, X, abs_tol=1e-16, rel_tol=1e-16, verbose=False):
        """
        Main public method for fitting k-means model.

        Parameters
        ----------
        X : NxD array
            Where N is # points and D is the dimensionality.
        abs_tol : float
            Convergence criteria w.r.t absolute change of loss.
        rel_tol : float 
            Convergence criteria w.r.t relative change of loss.
        verbose : bool
            Boolean to set whether method should print loss function (inertia).
            
        Returns
        -------
        cluster_labels : Nx1 int numpy array
            Labels for each data point.
        centers : K x D numpy array
            K centroid positions.
        loss : float
            Final inertia value of the objective function of KMeans.
        """
        # run k-means fitting n_init times and select the best result from different initializations
        final_loss = 99999999
        for init in range(self.n_init):

            cluster_labels, centers, loss = self._opt_clusters(X, init, abs_tol, rel_tol, verbose)
            
            # replace the final values if the inertia is better
            if loss < final_loss:
                final_cluster_labels = cluster_labels
                final_centers = centers
                final_loss = loss

        # update sklearn like class attributes with final fit values
        self.labels_ = final_cluster_labels
        self.cluster_centers_ = final_centers
        self.inertia_ = final_loss

        #return cluster_labels, centers, loss


# test this clustering method on some example data
if __name__ == "__main__":

    from sklearn.preprocessing import StandardScaler
    from sklearn.datasets import make_blobs
    import matplotlib.pyplot as plt

    # need a custom cmap 
    import matplotlib.cm as cm
    from matplotlib.colors import Normalize
    cmap = cm.tab10
    norm = Normalize(vmin=0, vmax=5)   

    # Create a dataset of 2D distributions
    centers = 5
    X_train, true_labels = make_blobs(n_samples=100, centers=centers, random_state=42)
    X_train = StandardScaler().fit_transform(X_train)

    # comparing to sklearn
    # from sklearn.cluster import KMeans
    # kmeans = KMeans(centers).fit(X_train)
    # colors = [cmap(norm(label)) for label in kmeans.labels_]
    # plt.scatter(X_train[:,0], X_train[:,1], c=colors)
    # plt.plot(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], 
    #          "k+", markersize=10)

    # my kmeans implementation
    # Fit centroids to dataset
    km = KMeans(n_clusters=centers)
    km.fit(X_train)
    print(km.inertia_)

    #make colors array for each datapoint cluster label
    colors = [cmap(norm(label)) for label in km.labels_]

    # View results
    plt.scatter(X_train[:,0], X_train[:,1], c=colors)
    plt.plot(km.cluster_centers_[:,0], km.cluster_centers_[:,1], 
             "k+", markersize=10)

    plt.show()