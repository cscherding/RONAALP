# IMPORTING MODULES
import math
import numpy as np

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import NearestNeighbors

import keras
from keras.models import Sequential
from keras.layers import InputLayer

from .utilities.kernels import RBF_Exponential, RBF_Spline
from .utilities import clustering_newman
from .utilities import K_Means
from .utilities import create_autoencoder

METHODS = ["Exponential", "Spline"]


class Ronaalp:

    """Reduced Order Nonlinear Approximation with Active Learning Procedure.


    Parameters
    ----------
    d : int, default = 2
            Dimension of the latent space.
    feat_min : float, default = 0
            Minimum value for MinMaxScaling.
    feat_max : float, default = 1
            Maximum value for MinMaxScaling.
    n_tree : int, default = 20
            Number of trees in RandomForest classifier.
    n_ghost : int, default = 1000
            Number of neighboring point to add to build the ghost layer of each cluster
    n_rbf : int, default = 200
            Number of kernels in each clusters.

    method : str, optional
            Kernel function to be used for RBF. Should be:
            - 'Exponential'   : exponential kernel exp(-r^2/l)
            - 'Spline'        : thin-plate spline kernel r^2*log(r)
    rbf_degree : int, default = 3
            Highest degree of added polynomial terms for RBF. Only used when method='Spline'
    rbf_smoothing : float, default = 1e-3
            Smoothing parameter for RBF.

    architecture : list, default = [12,6]
        A list specifying the number of neurons in each
        hidden layer for the encoding and decoding parts.
    clustering : boolean, default = True
        Specify if the user wants to perform Newman clustering in the latent space.

    Attributes
    ----------

    scaler_x : MinMaxScaler object
        Scaler for input data.
    scaler_y : MinMaxScaler object
        Scaler for latent space.
    scaler_z : MinMaxScaler object
        Scaler for output data.

    encoder: Keras neural network.
        Deep encoder to project high-dimensional input data in low-dimensional latent space.

    n_clusters: int
        Number of clusters found by Newman algorithm in the latent space.
    classifier: Scikit-learn random forest classifier
        Classifier trained on Newman clusters.

    kmeans_list: list of length n_clusters
        List containing the kmeans objects of each cluster.

    surrogate_list: list of length n_clusters
        List containing the RBF surrogate models of each cluster.

    References
    ----------
    .. [1] Scherding, C., Rigas, G., Sipp, D., Schmid, P. J., & Sayadi, T. (2023). Data-driven framework for input/output lookup tables reduction: Application to hypersonic flows in chemical nonequilibrium. Physical Review Fluids, 8(2), 023201.
    .. [2] Scherding, C. (2023). Predictive modeling of hypersonic flows in thermochemical nonequilibrium: physics-based and data-driven approaches. PhD Thesis, Sorbone University.
    """

    def __init__(
        self,
        d=2,
        feat_min=0,
        feat_max=1,
        method="Spline",
        n_rbf=200,
        n_ghost=1000,
        n_tree=20,
        rbf_degree=3,
        rbf_smoothing=1e-3,
        n_epochs=300,
        architecture=[12, 6],
        clustering=True,
    ):
        if method not in METHODS:
            raise ValueError(
                "invalid method %s,\n  valid options are %s" % (method, METHODS)
            )

        self.d = d

        self.feat_range = (feat_min, feat_max)

        self.n_tree = n_tree
        self.n_rbf = n_rbf
        self.n_ghost = n_ghost

        self.n_epochs = n_epochs
        self.architecture = architecture

        self.method = method
        self.rbf_degree = rbf_degree
        self.rbf_smoothing = rbf_smoothing

        self.surrogate_list = []
        self.kmeans_list = []

        self.clustering = clustering

        return

    def train(self, x_train, z_train, x_test, z_test):
        """Train the model to find a reduced-order representation of the high-dimensional mapping between ``x_train`` and ``z_train`` in a latent space of dimension ``d`` with ``n_clusters`` distinct clusters .

        Parameters
        ----------
        x_train : ndarray, shape (m,D,)
            Array of points where function values are known. m is the
            number of sampling points and D is the number of input dimensions.
        z_train : ndarray, shape (m,P,)
            Array of function values at ``x_train``. P is the number of output dimensions.

        x_test : ndarray, shape (_,D,)
             Similar as ``x_train`` but for testing.
        z_test : ndarray, shape (_,P,)
             Similar as ``z_train`` but for testing.
        """

        # 0. set # of input/output features

        m = x_train.shape[0]

        self.D = x_train.shape[1]
        self.P = z_train.shape[1]

        # 1. scale data

        print("1. Setting up scalers ...")
        self.scaler_x = MinMaxScaler(feature_range=self.feat_range)

        x_train = self.scaler_x.fit_transform(x_train)
        x_test = self.scaler_x.transform(x_test)

        self.scaler_z = MinMaxScaler(feature_range=self.feat_range)

        z_train = self.scaler_z.fit_transform(z_train)
        z_test = self.scaler_z.transform(z_test)
        print("... Done !")

        # 2. Train deep encoder

        print("2. Training encoder, latent space dimenion  d = " + str(self.d) + " ...")

        if self.d != self.D:
            # Create autoencoder based on architecture parameter
            autoencoder, self.encoder = create_autoencoder(
                self.architecture, self.D, self.d, self.P
            )
            autoencoder.compile(optimizer="adam", loss="mse", metrics=["accuracy"])

            # Train autoencoder
            autoencoder.fit(
                x_train,
                z_train,
                epochs=self.n_epochs,
                batch_size=256,
                shuffle=True,
                validation_data=(x_test, z_test),
            )
        else:
            # If latent dimension is equal to input dimension, an identity model is created
            self.encoder = Sequential()
            self.encoder.add(InputLayer(input_shape=(self.D,)))

        # Scaler for latent space
        self.scaler_y = MinMaxScaler(feature_range=self.feat_range)

        y_train = self.encoder.predict(x_train)
        y_train = self.scaler_y.fit_transform(y_train)

        print("... Done !")

        # 3. Find spectral clusters in the latent space

        print("3. Find spectral clusters in the latent space ...")

        # Sample points in the latent space,
        # capped at 10000 due to the time-complexity
        # of Newman clustering algorithm

        n_samples = np.minimum(10000, m)
        rand_index = np.random.choice(m, n_samples, replace=False)
        y_newman = y_train[rand_index, :]

        if self.clustering:
            # Build distance matrix
            A_in = euclidean_distances(y_newman, y_newman)

            # Threshold to build graph connectivity
            thres = 1.0 * np.mean(A_in)

            # Apply clustering algorithm
            _, _, _, Ci, nci = clustering_newman(A_in, thres)

            Ci = Ci - 1
            Ci = Ci.flatten()
            Ci = Ci.astype(int)
        else:
            Ci = np.zeros((n_samples))
            nci = 1

        self.n_clusters = nci

        print("... Done !")

        # 4. Train classifier
        print("4. Training classifier ...")

        self.classifier = RandomForestClassifier(
            n_estimators=self.n_tree, criterion="entropy", random_state=42
        )
        self.classifier.fit(y_newman, Ci)

        Ci_train = self.classifier.predict(y_train)

        print("... Done !")

        # 5. Find ghost layer of each cluster

        print("5. Creating ghost layers ...")

        if self.n_clusters > 1:
            ghost_y = [0.0] * nci

            for i in range(self.n_clusters):
                centroid = np.mean(y_train[Ci_train == i, :], axis=0)

                for j in range(nci):
                    if j != i:  # Loop over other clusters
                        y_work = y_train[Ci_train == j, :]

                        # Find nearest neighbors in that cluster
                        knn = NearestNeighbors(n_neighbors=self.n_ghost, n_jobs=1).fit(
                            y_work
                        )
                        indices = knn.kneighbors(
                            centroid.reshape(1, -1), return_distance=False
                        )[:, :]

                        y_nn = y_work[indices[0], :]

                        # Check if they are adjacent clusters

                        A_dist = euclidean_distances(y_nn, y_train[Ci_train == i, :])

                        if np.min(A_dist) < 1e-2:
                            if type(ghost_y[i]) == float:
                                ghost_y[i] = y_nn
                            else:
                                ghost_y[i] = np.concatenate(
                                    (ghost_y[i], y_nn), axis=0
                                )

        print("... Done !")

        # 6. Build surrogate model for each spectral cluster

        print("6. Building RBFs ...")
        nc_rbf = self.n_rbf

        for ic in range(self.n_clusters):
            # If more than one cluster, append ghost layer
            if self.n_clusters > 1:
                n_ghost = ghost_y[ic].shape[0]

                XY = np.concatenate((y_train[Ci_train == ic, :], ghost_y[ic]), axis=0)
            else:
                XY = y_train
                n_ghost = 0

            # Construct cluster tessalation for RBF using custom K_Means class based on scikit-learn kmeans

            kmeans = K_Means(k=nc_rbf)
            kmeans.fit(XY)

            Centroids = kmeans.centroids

            # Replace centroid by closest representation in training data for a cheap 'k-medoid' alternative.

            nn = NearestNeighbors(n_neighbors=1, radius=0.4)
            nn.fit(y_train)

            ind = nn.kneighbors(Centroids, 1, return_distance=False)

            x_centroids = np.squeeze(x_train[ind, :])
            y_centroids = np.squeeze(y_train[ind, :])
            z_centroids = np.squeeze(z_train[ind, :])

            # Initialize and train RBF

            if self.method == "Exponential":
                # internal parameter is set to mean inter-center distance
                l = np.mean(euclidean_distances(y_centroids, y_centroids))

                surrogate = RBF_Exponential(l=self.l, epsilon=self.rbf_smoothing)
            elif self.method == "Spline":
                surrogate = RBF_Spline(
                    epsilon=self.rbf_smoothing, degree=self.rbf_degree
                )

            surrogate.fit(y_centroids, z_centroids)

            # Append objects to the cluster lists

            self.kmeans_list.append(kmeans)
            self.surrogate_list.append(surrogate)

        ####
        print("... Done !")

        return None

    def evaluate(self, x):
        """Evaluate the reduced-order model at given points.

        Parameters
        ----------
        x : ndarray, shape (n_state,D,)
            Array of points where we want to evaluate the surrogate model.

        Returns
        -------
        z : ndarray, shape(n_state,P,)
            Array of interpolated values at ``x``.
        extrp_flag : ndarray, shape(n_state,)
            Array of extrapolation flag at ``x``:
                0=interpolation,
                1=extrapolation.
        """

        n_state = x.shape[0]

        z = np.empty((n_state, self.P))
        extrp_flag = np.empty((n_state))

        # 1. Scale X vector

        x = self.scaler_x.transform(x)

        # 2. transform new point in embedded coordinates

        y = self.encoder.predict(x, batch_size=n_state, verbose=0)
        y = self.scaler_y.transform(y)

        # 3. Call classifier
        c_index = self.classifier.predict(y)
        c_prob = self.classifier.predict_proba(y)

        # 4. Call surrogate surface on each spectral cluster

        for i in range(self.n_clusters):
            index = np.where(c_index == i)[0]
            if len(index) != 0:
                z[index, :], extrp_flag[index] = self.surrogate_list[i].evaluate(y[index, :])
            ####
        ####

        # 5. Scale back thermochemical properties to physical units

        z = self.scaler_z.inverse_transform(z)

        return z, extrp_flag

    def update(self, x_new, z_new):
        """Retrain the model by first tesselating the new points and then retraining the RBFs using the Schur complement technique.

        Parameters
        ----------
        x_new : ndarray, shape (l,d,)
            New array of points where function values are known. l is the number of new sampling points.
        z_new : ndarray, shape (l,p)
            Array of new function values at ``x_new``.
        """

        # 1. scaling

        x_new = self.scaler_x.transform(x_new)
        z_new = self.scaler_z.transform(z_new)

        # 2. embedding

        y_new = self.encoder.predict(x_new, verbose=0)
        y_new = self.scaler_y.transform(y_new)

        # 3. classification

        c_edge_index = self.classifier.predict(y_new)

        # 4. Loop over Newman's clusters for retrain

        for i in range(self.n_clusters):
            # 4.0 Get corresponding points

            x_edge = x_new[c_edge_index == i, :]
            y_edge = y_new[c_edge_index == i, :]
            z_edge = z_new[c_edge_index == i, :]

            if x_edge.shape[0] != 0:
                rbf_retrain = self.surrogate_list[i]
                kmeans_retrain = self.kmeans_list[i]

                # 4.1 Update kmeans centroids sequentially

                new_centroids = kmeans_retrain.update(y_edge)

                if new_centroids.shape[0] != 0:
                    # 4.2 Find closest point in training set

                    nn = NearestNeighbors(n_neighbors=1, radius=0.4)
                    nn.fit(y_edge)

                    ind = nn.kneighbors(new_centroids, 1, return_distance=False)[:, 0]

                    # 4.3 Get corresponding inputs/outputs

                    y_centers_retrain = y_edge[ind, :]
                    z_centers_retrain = z_edge[ind, :]

                    # 4.4 Retrain RBF with Schur complement

                    rbf_retrain.retrain_schur(y_centers_retrain, z_centers_retrain)

                    # 5. update surrogate and kmeans objects

                    self.surrogate_list[i] = rbf_retrain

                    self.kmeans_list[i] = kmeans_retrain

            else:
                print("No edge state found in cluster #" + str(i))

        return
