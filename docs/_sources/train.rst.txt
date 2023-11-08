Training
================

The training is done in three sequential steps:

1. Training of the :func:`auto-encoder <.utilities.create_autoencoder>` to find a low-dimensional subspace (latent space of dimension ``d``) of the inputs that accounts for the variation of the outputs.

2. Clustering in the latent space using :func:`Newman's clustering <.utilities.clustering_newman>` algorithm to separate regions with different dynamics.

3. Training :class:`RBF <.utilities.kernels.exponential.RBF_Exponential>` networks on each Newman cluster in a two-stage framework:

    a. ``n_rbf`` centers :math:`\mathbf{X}^c` are computed using the :class:`K_Means <.utilities.online_kmeans.K_Means>` algorithm.
    b. The optimal weights are obtained by solving the corresponding RBF linear system.

.. toctree::
    encoder
    newman
    kmeans
    kernels
    
.. autofunction:: RONAALP.Ronaalp.train
    
