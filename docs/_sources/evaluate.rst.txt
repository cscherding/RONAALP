Evaluate
========

During the evaluation of the model, the extrapolation function is defined as

.. math::
    f_e(\mathbf{x}^t) = \min_{ \mathbf{x}^c ~\in~\mathbf{X}^c }  \lVert \mathbf{x}^t - \mathbf{X}^c \rVert,

where  :math:`\mathbf{X}^c` is the set of centers of the RBF. The extrapolation threshold is defined as

.. math::
    d_e = \frac{1}{N_\textrm{R}}\sum_{i=1}^{N_\textrm{R}}  \left( \frac{1}{k}\sum_{j=1}^{k} \rVert \mathbf{x}^c_i - \mathbf{x}^c_{i,j} \lVert \right)

where :math:`\mathbf{X}^c_{i,k} = [\mathbf{x}^c_{i,1} \dots \mathbf{x}^c_{i,k}]` represents the matrix containing the k-nearest neighbors of centroid :math:`\mathbf{x}^c_i`

.. autofunction:: RONAALP.Ronaalp.evaluate
