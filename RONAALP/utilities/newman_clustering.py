##############################################################################
# %% IMPORTING MODULES

import math
import numpy as np
import scipy.linalg as la
import scipy.linalg

##############################################################################


def clustering_newman(A, epsilon=1e-1):
    """Perform Newmann clustering on adjacency matrix A given epsilon threshold.

    Parameters
    ----------
    A       : array-like, shape (N,N), distance matrix
    epsilon : float, default=1e-1
     thresholding to build adjacency matrix

    Returns
    -------
    AA      : array-like, shape (N,N)
    AA_     : array-like, shape (N,N), graph matrix obtained with thresholding
    AAA     : array-like, shape (N,N), ordered graph matrix
    Ci      : array-like, shape (N,1) cluster label of each pts.
    nc      : int, number of communities found by the algorithm

    References
    ----------
    .. [1] Leicht, E. A., & Newman, M. E. (2008). Community structure in directed networks. Physical review letters, 100(11), 118703.
    """

    N = len(A)  # cluster sizes
    nc = 10  # initial guess for number of clusters

    AA_ = np.zeros((N, N))

    AA_[A < epsilon] = 1
    AA_[A >= epsilon] = 0

    AA_ = AA_ - np.diag(np.diag(AA_))
    AA_ = (AA_ + AA_.T) / 2

    AA = AA_ + np.eye(len(AA_))

    Ki = np.sum(AA, axis=0, dtype=np.int32)
    Ki = Ki[np.newaxis, :]
    Ko = np.sum(AA, axis=1, dtype=np.int32)
    Ko = Ko[:, np.newaxis]

    m = np.sum(Ki)

    b = AA - (Ko @ Ki) / m

    B = b + b.T

    Ci = np.ones((N, 1))
    cn = 1
    U = [1, 0]
    ind = np.arange(N)
    Bg = B.copy()
    Ng = N

    it = 0
    while U[0] != 0:
        u1, v = la.eigh(Bg)
        v1 = v[:, np.argmax(u1)]
        S = np.ones((Ng, 1))
        S[v1 < 0] = -1
        q = S.T @ Bg @ S
        if q > 1e-10:
            qmax = q
            Bg = Bg - np.diag(np.diag(Bg))
            indg = np.ones((Ng, 1), dtype=bool)
            Sit = S
            while indg.any() is True:
                Qit = qmax - 4 * Sit * (Bg @ Sit)
                qmax = max(Qit * indg)
                imax = np.where(Qit == qmax)[0]
                Sit[imax] = -Sit[imax]
                indg[imax] = False
                if qmax > q:
                    q = qmax
                    S = Sit

            if abs(sum(S)) == Ng:
                del U[0]
            else:
                cn = cn + 1
                ci = Ci[Ci == U[0]][:, np.newaxis]
                ci[S == 1] = U[0]
                ci[S == -1] = cn
                Ci[Ci == U[0]] = ci[:, 0]
                U.insert(0, cn)

        else:
            del U[0]
        it += 1
        if U[0] == 0:
            break

        ind = np.where(Ci == U[0])[0]
        bg = B[ind, :][:, ind]
        Bg = bg - np.diag(sum(bg))
        Ng = len(ind)

    s = np.tile(Ci, (1, N))
    Q = np.sum(B[s - s.T == 0] / (2 * m))

    nc = int(np.max(Ci))
    print(nc, " communities found")

    for i in range(1, nc + 1):
        print("points in Cluster ", i, " =  ", len(Ci[Ci == i]))

    # Reordering matrix

    jj = []

    for i in range(1, nc + 1):
        ind = np.where(Ci == i)[0]
        jj.extend(ind)

    jj_test = np.sort(jj)

    PP = np.eye(AA.shape[0])
    PP = PP[jj, :]
    AAA = PP @ AA_ @ PP.T

    return AA, AA_, AAA, Ci, nc
