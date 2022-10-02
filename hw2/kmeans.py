"""
Initially written 2022-10-02 by DTY.

Inspired by: 
https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
https://towardsdatascience.com/create-your-own-k-means-clustering-algorithm-in-python-d7d4c9077670
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
        distance_metric : str (TODO)
            Distance calculations can be 'euclidean', 'jaccard', 'cosine', or 'manhattan'.
        """
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.distance_metric = distance_metric

    def euclidean(self, point, data):
        """
        Euclidean distance (sum of squares) between point & data.
        Point has dimensions (m,), data has dimensions (n,m), and output will be of size (n,).
        """
        return np.sqrt(np.sum((point - data)**2, axis=1))

    # TODO
    def fit(self, X_train):
        """
        Initialize the centroids, using the "k-means++" method, where a random datapoint is selected as the first,
        then the rest are initialized w/ probabilities proportional to their distances to the first
        Pick a random point from train data for first centroid
        """
        self.centroids = [random.choice(X_train)]

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
                if np.isnan(centroid).any():  # Catch any np.nans, resulting from a centroid having no points
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

    # Create a dataset of 2D distributions
    centers = 5
    X_train, true_labels = make_blobs(n_samples=100, centers=centers, random_state=42)
    X_train = StandardScaler().fit_transform(X_train)

    # Fit centroids to dataset
    kmeans = KMeans(n_clusters=centers)
    kmeans.fit(X_train)

    # need a custom cmap
    import matplotlib.cm as cm
    from matplotlib.colors import Normalize
    cmap = cm.tab10
    norm = Normalize(vmin=0, vmax=5)     

    #make colors array for each datapoint cluster label
    colors = [cmap(norm(label)) for label in true_labels]

    # View results
    class_centers, classification = kmeans.evaluate(X_train)
    plt.scatter(X_train[:,0], X_train[:,1], c=colors)
    plt.plot([x for x, _ in kmeans.centroids],
             [y for _, y in kmeans.centroids],
             "k+", markersize=10)
    plt.show()