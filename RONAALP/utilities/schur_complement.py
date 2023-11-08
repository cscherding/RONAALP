##############################################################################
# %% IMPORTING MODULES

import math
import numpy as np
import scipy.linalg as la
import scipy.linalg

##############################################################################


def schur_inverse(Ainv, B, C, D):
    """Compute inverse of matrix M = [[A B], [C D]], defined by the block matrix A, B, C and D using the Schur complement.

    Parameters
    ----------
    Ainv : array-like, shape (l1,l1), inverse of block matrix A
    B    : array-like, shape (l1,l2), block matrix B
    C    : array-like, shape (l2,l1), block matrix C
    D    : array-like, shape (l2,l2), block matrix D

    Returns
    -------
    Minv : array-like, shape (l1+l2,l1+l2), inverse of matrix M

    References
    ----------
    .. [1] https://chrisyeh96.github.io/2021/05/19/schur-complement.html
    """

    l1, _ = Ainv.shape
    l2, _ = D.shape

    schurA = D - C @ Ainv @ B
    schurA_inv = la.inv(schurA)

    Minv = np.empty((l1 + l2, l1 + l2))

    Minv[:l1, :l1] = Ainv + Ainv @ B @ schurA_inv @ C @ Ainv
    Minv[:l1, l1:] = -Ainv @ B @ schurA_inv
    Minv[l1:, :l1] = -schurA_inv @ C @ Ainv
    Minv[l1:, l1:] = schurA_inv

    return Minv
