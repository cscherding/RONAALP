import numpy as np
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import euclidean_distances


class K_Means:
    """K-means clustering custom class based on skicit-learn version augmented with a sequential (online) update procedure.

    Parameters
    ----------
    k   : int, default = 2
            The number of clusters to form as well as the number of
            centroids to generate.

    max_iter : int, default=300
        Maximum number of iterations of the k-means algorithm for a
        single run.

    tol : float, default=1e-3
        Relative tolerance with regards to Frobenius norm of the difference
        in the cluster centers of two consecutive iterations to declare
        convergence.

    n_knn : int, default=5
        Number of centroid neighbors to consider when computing the mean inter cluster distance.

    Attributes
    ----------
    centroids : ndarray of shape (k, n_features)
        Coordinates of cluster centers. If the algorithm stops before fully
        converging (see ``tol`` and ``max_iter``), these will not be
        consistent with ``labels_``.

    labels_ : ndarray of shape (n_samples)
        Labels of each point.

    delta : float
        Mean of inter cluster distance.

    counts : ndarray of shape (k,)
        Number of data point belonging to each cluster.

    nearest_C : Sklearn nearest neighbor object
        Nearest neighbor graph fitted on k-means centroid.
    """

    def __init__(self, k=2, tol=1e-3, max_iter=300, n_knn=5):
        self.k = k
        self.tol = tol
        self.max_iter = max_iter
        self.n_knn = n_knn

    def fit(self, data):
        """Fit kmeans centroids to data using sklearns implementation.

        Parameters
        ----------
        data : ndarray, shape (n_samples, n_features,)
            Array of points to divide in k clusters.

        """

        km = KMeans(
            n_clusters=self.k, random_state=42, tol=self.tol, max_iter=self.max_iter
        ).fit(data)

        self.centroids = km.cluster_centers_

        # fit a NearestNeighbors graph
        self.nearest_C = NearestNeighbors(n_neighbors=1, radius=0.4)
        self.nearest_C.fit(self.centroids)

        # find delta threshold

        A = np.empty((self.k, self.n_knn))
        nn_c = self.nearest_C.kneighbors(
            self.centroids, self.n_knn + 1, return_distance=False
        )

        for i in range(self.k):
            A[i, :] = euclidean_distances(
                self.centroids[i, :].reshape(1, -1), self.centroids[nn_c[i, 1:], :]
            )

        self.delta = np.mean(A)

        # get Classification

        self.labels = km.labels_
        self.counts = np.zeros((self.k))
        for i in range(self.k):
            self.counts[i] = len(np.where(km.labels_ == i)[0])

    def set_delta(self, new_delta):
        """Update delta parameter."""
        self.delta = new_delta

    def predict(self, data):
        """Predict within which cluster lie new data.

        Parameters
        ----------
        data : ndarray, shape (n_samples, n_features,)
            Array of points to classify.
        """
        distances = [
            np.linalg.norm(data - self.centroids[centroid])
            for centroid in self.centroids
        ]
        classification = distances.index(min(distances))
        return classification

    def update(self, new_data):
        """Sequentially update the clustering using online k-means version.

        Parameters
        ----------
        new_data : ndarray, shape (n_samples2, n_features,)
            Array of points to sequentially clusterize.

        References
        ----------
        .. [1] Hart, P. E., Stork, D. G., & Duda, R. O. (2000). Pattern classification. Hoboken: Wiley.
        """

        new_count = 0

        newnn = self.nearest_C.kneighbors(new_data, 1, return_distance=False)

        for i, x_new in enumerate(new_data):
            index = newnn[i]
            ni = self.counts[newnn[i]]
            mi = self.centroids[newnn[i]]

            if np.linalg.norm(x_new - mi) < self.delta:
                ni = ni + 1
                mi = mi + (1 / ni) * (x_new - mi)

                self.counts[newnn[i]] = ni
                self.centroids[newnn[i]] = mi

            else:
                A = np.append(self.centroids, x_new.reshape(1, -1), axis=0)
                self.centroids = A
                A = np.append(self.counts, np.ones((1)), axis=0)
                self.counts = A
                self.k = self.k + 1
                new_count += 1

                self.nearest_C = NearestNeighbors(n_neighbors=1, radius=0.4)
                self.nearest_C.fit(self.centroids)
                newnn = self.nearest_C.kneighbors(
                    new_data[:, :], 1, return_distance=False
                )

        return self.centroids[-new_count:]
