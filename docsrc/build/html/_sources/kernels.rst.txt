Kernel functions
----------------

Here we provide a description of the different classes used to build
and evaluate the radial basis function networks. Two different
kernel functions :math:`\phi(r)` are defined in the code, the Exponential
kernel (:class:`RBF_Exponential <.utilities.kernels.exponential.RBF_Exponential>`) and the
the Spline kernel (:class:`RBF_Spline <.utilities.kernels.spline.RBF_Spline>`).

The ``fit`` method of these classes solves the system :math:`A s = F`.
In the case of RBF interpolants, the vector :math:`F` contains the values
of the function where the points have been evaluated and the matrix :math:`A`
is defined as:

.. math::
    A = \begin{bmatrix}
            \Phi & P \\
            P^T & 0.
        \end{bmatrix}

The matrix :math:`\Phi` is defined as:

.. math::
    \Phi_{i,j} = \phi(r_{i,j}),

where :math:`r_{i,j} = \left \| x^i-x^j \right\|` is the Euclidean distance
between the centers :math:`x^i` and :math:`x^j`. :math:`P` represents the added polynomial terms and the vector :math:`s` contains the weights of the RBF network.

Once the ``fit`` method has been used to build the surrogate model (find the vector :math:`s`), it is possible
to evaluate points using the method ``evaluate``.

If new centers are added, the methods ``retrain`` and ``retrain_schur`` can be used to inverse the augmented RBF matrix

.. math::
    A = \begin{bmatrix}
        \Phi & \Phi_{1,2} \\
        \Phi_{2,1} & \Phi_{2,2}.
    \end{bmatrix}

where :math:`\mathbf{\Phi}_{2,2}` represents the kernel matrix of the new centers added by the online k-means algorithm, and :math:`\mathbf{\Phi}_{1,2} = \mathbf{\Phi}_{2,1}^T` the cross kernel matrix between initial and new centers, respectively.

.. autofunction:: RONAALP.utilities.schur_complement.schur_inverse

Exponential Kernel
~~~~~~~~~~~~~~~~~~
The Exponential Kernel is defined as follows:

.. math::
    \Phi(r) = \exp \left( -\dfrac{r^2}{2 l^2} \right),
    
where :math:`l>0` and :math:`r=\|y-x\|`, where :math:`\|\cdot\|` is the
Euclidean norm. In this case, the polynomial terms can be omitted.

.. automodule:: RONAALP.utilities.kernels.exponential
    :members:

Spline Kernel
~~~~~~~~~~~~~
The thin-plate spline kernel is defined as follows:

.. math::
    \Phi(r) = r^2 \log \left(r\right),

where :math:`r=\|y-x\|`, where :math:`\|\cdot\|` is the Euclidean norm. 


.. automodule:: RONAALP.utilities.kernels.spline
    :members:
