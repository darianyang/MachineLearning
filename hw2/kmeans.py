"""
Initially written 2022-10-02 by DTY.

Inspired by: 
https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
https://towardsdatascience.com/create-your-own-k-means-clustering-algorithm-in-python-d7d4c9077670
https://analyticsarora.com/k-means-for-beginners-how-to-build-from-scratch-in-python/
"""

import numpy as np
import random
from numpy.random import uniform

class KMeans:
    """
    Custom k-means clustering function with options for alternative distance metrics.
    """
    def __init__(self, n_clusters=8, max_iter=300, distance_metric="euclidean"):
        """
        Parameters
        ----------
        n_clusters : int
            The number of clusters to form as well as the number of centroids to generate.
        max_iter : int
            Maximum number of iterations of the k-means algorithm for a single run.
        n_init : int, default=10 (TODO)
            Number of times the k-means algorithm will be run with different centroid seeds. 
            The final results will be the best output of n_init consecutive runs in terms of inertia.
        distance_metric : str (TODO)
            Distance calculations can be 'euclidean', 'jaccard', 'cosine', or 'manhattan'.
        """
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.distance_metric = distance_metric

    def euclidean(self, point, data):
        """
        Euclidean distance between point & data.
        Point has dimensions (m,), data has dimensions (n,m), and output will be of size (n,).
        """
        return np.sqrt(np.sum((point - data)**2, axis=1))

    def _calc_inertia(self):
        """
        Calculate the inertia of a clustering run.
        """
        pass

    # TODO
    def _get_loss(self, centers, cluster_idx, points):
        """
        The loss function is the metric by which we evaluate the performance of our clustering algorithm. 
        Our loss is simply the sum of the square distances between each point and its cluster centroid.

        Args:
            centers: KxD numpy array, where K is the number of clusters, and D is the dimension
            cluster_idx: numpy array of length N, the cluster assignment for each point
            points: NxD numpy array, the observations
        Return:
            loss: a single float number, which is the objective function of KMeans. 
        """
        dists = self.pairwise_dist(points, centers)
        loss = 0.0
        N, D = points.shape
        for i in range(N):
            loss = loss + np.square(dists[i][cluster_idx[i]])
        
        return loss

    def _init_centers(self):
        """
        Initialize the centroids, using the "k-means++" method, where a random datapoint is selected as the first,
        then the rest are initialized w/ probabilities proportional to their distances to the first.
        """
        # TODO: put this whole process in a loop for the n_init
        # Pick a random point from train data for first centroid
        self.centroids = [random.choice(X_train)]

        # k-means++
        for _ in range(self.n_clusters-1):
            # Calculate distances from points to the centroids
            dists = np.sum([self.euclidean(centroid, X_train) for centroid in self.centroids], axis=0)
            
            # Normalize the distances
            dists /= np.sum(dists)
            
            # Choose remaining points based on their distances
            new_centroid_idx, = np.random.choice(range(len(X_train)), size=1, p=dists)
            self.centroids += [X_train[new_centroid_idx]]

        # This initial method of randomly selecting centroid starts is less effective
        # min_, max_ = np.min(X_train, axis=0), np.max(X_train, axis=0)
        # self.centroids = [uniform(min_, max_) for _ in range(self.n_clusters)]

    def fit(self, X_train):
        # first initialize centers using k-means++
        self._init_centers()

        # Iterate, adjusting centroids until converged or until passed max_iter
        iteration = 0
        prev_centroids = None
        while np.not_equal(self.centroids, prev_centroids).any() and iteration < self.max_iter:
            
            # Sort each datapoint, assigning to nearest centroid
            sorted_points = [[] for _ in range(self.n_clusters)]
            
            for x in X_train:
                dists = self.euclidean(x, self.centroids)
                centroid_idx = np.argmin(dists)
                sorted_points[centroid_idx].append(x)
            
            # Push current centroids to previous, reassign centroids as mean of the points belonging to them
            prev_centroids = self.centroids
            self.centroids = [np.mean(cluster, axis=0) for cluster in sorted_points]
            
            for i, centroid in enumerate(self.centroids):
                # Catch any np.nans, resulting from a centroid having no points
                if np.isnan(centroid).any():
                    self.centroids[i] = prev_centroids[i]
            
            iteration += 1

    def evaluate(self, X):
        """
        Eval func
        """
        centroids = []
        centroid_idxs = []

        for x in X:
            dists = self.euclidean(x, self.centroids)
            centroid_idx = np.argmin(dists)
            centroids.append(self.centroids[centroid_idx])
            centroid_idxs.append(centroid_idx)

        return centroids, centroid_idxs



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

    # Fit centroids to dataset
    kmeans = KMeans(n_clusters=centers, max_iter=1000)
    kmeans.fit(X_train)

    #make colors array for each datapoint cluster label
    colors = [cmap(norm(label)) for label in true_labels]

    # View results
    class_centers, classification = kmeans.evaluate(X_train)
    plt.scatter(X_train[:,0], X_train[:,1], c=colors)
    plt.plot([x for x, _ in kmeans.centroids],
             [y for _, y in kmeans.centroids],
             "k+", markersize=10)

    # comparing to sklearn
    # from sklearn.cluster import KMeans
    # kmeans = KMeans(centers).fit(X_train)
    # colors = [cmap(norm(label)) for label in kmeans.labels_]
    # plt.scatter(X_train[:,0], X_train[:,1], c=colors)
    # plt.plot(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], 
    #          "k+", markersize=10)

    plt.show()