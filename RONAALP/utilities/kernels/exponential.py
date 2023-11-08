import numpy as np
import numpy.linalg as nla
import scipy.linalg as la
from scipy.optimize import differential_evolution, NonlinearConstraint
from sklearn.metrics.pairwise import euclidean_distances
from RONAALP.utilities import schur_inverse

EPS = np.finfo(np.float64).eps


class RBF_Exponential:
    """RBF Exponential kernel class.

    Parameters
    ----------
    l : float or ndarray, shape (m,), optional
        Internal parameter. Width of the kernel.
    epsilon : float, optional
              Smoothing parameter.

    Attributes
    ----------
    l : float or ndarray, shape (m,)
        Internal parameter. Width of the kernel.
    s : ndarray, shape(m,)
        RBF coefficients.
    x : ndarray, shape (m,d,)
        Array of points where function values are known. m is the
        number of sampling points and d is the number of dimensions.
    f : ndarray, shape (m,p,)
        Array of function values at ``x``.
    """

    def __init__(self, l=1.0, epsilon=1e-3):
        self.l = l
        self.s = None
        self.x = None
        self.f = None
        self.epsilon = epsilon
        self.phi_min = np.exp(-0.08)

    def fit(self, x, f):
        """Build surrogate model.

        Parameters
        ----------
        x : ndarray, shape (m,d,)
            Array of points where function values are known. m is the
            number of sampling points and d is the number of dimensions.
        f : ndarray, shape (m,p)
            Array of function values at ``x``.

        Returns
        -------
        Phi : ndarray, shape(m,m,)
            RBF matrix.
        A : ndarray, shape(m*(d+1),m*(d+1),)
            RBF matrix with linear polynomial terms.
        """
        self.x = x
        self.f = f

        m, d = self.x.shape
        mm, p = self.f.shape

        assert mm == m

        self.m = m
        self.d = d
        self.p = p

        if np.isscalar(self.l):
            self.l = np.ones(m) * self.l

        # Calculate distance matrix
        R = euclidean_distances(x, x) ** 2

        # Scale each kernel by individual (or not) scale factor
        for i in range(m):
            R[i, :] = R[i, :] / self.l**2

        # Apply kernel
        Phi = np.exp(-R / 2)

        # Apply smoothing
        self.A = Phi + self.epsilon * np.eye(m)

        # Solve

        self.Ainv = la.inv(self.A)

        self.s = np.dot(self.Ainv, self.f)

        return Phi

    def evaluate(self, y):
        """Evaluate surrogate model at given points.

        Parameters
        ----------
        y : ndarray, shape (n,d,)
            Array of points where we want to evaluate the surrogate model.

        Returns
        -------
        f : ndarray, shape(n,)
            Array of interpolated values at ``y``.
        """
        if self.s is None:
            return None

        y = np.array(y)
        n, d = y.shape

        # Compute distance matrix
        R = euclidean_distances(y, self.x) ** 2

        # Scale each kernel by individual (or not) scale factor
        for i in range(n):
            R[i, :] = R[i, :] / self.l**2

        # Apply kernel
        Phi = np.exp(-R / 2)

        # Calculate extrapolation flag
        SigMax = np.max(Phi, axis=1)
        extrp = 1 - np.floor(
            (1 / self.phi_min) * np.minimum(self.phi_min * np.ones((n)), SigMax)
        )

        # Evaluation of interpolant
        f = np.dot(Phi, self.s)

        return f, extrp

    def retrain(self, x_new, f_new, l_new=1.0):
        """Retrain the surrogate model in a brute force approach.

        Parameters
        ----------
        x_new : ndarray, shape (m2,d,)
            New array of points where function values are known. m2 is the number of new sampling points.
        f_new : ndarray, shape (m2,p)
            Array of new function values at ``x_new``.
        l_new : float or ndarray, shape (m2,)
            Internal parameter. Width of the kernel on this new points.
        """
        # concatenate new x and f vectors and update sizes
        self.x = np.concatenate((self.x, x_new), axis=0)
        self.f = np.concatenate((self.f, f_new), axis=0)

        m, _ = self.x.shape
        self.m = m

        if np.isscalar(l_new):
            self.l = self.l[0] * np.ones(m)
        else:
            self.l = np.concatenate((self.l, l_new), axis=0)

        # Calculate new distance matrix
        R = euclidean_distances(self.x, self.x) ** 2

        # Scale each kernel by individual (or not) scale factor
        for i in range(m):
            R[i, :] = R[i, :] / self.l**2

        # Apply kernel
        Phi = np.exp(-R / 2)

        # Apply smoothing
        self.A = Phi + self.epsilon * np.eye(m)

        # Solve
        self.Ainv = la.inv(self.A)

        self.s = np.dot(self.Ainv, self.f)

        return None

    def retrain_schur(self, x_new, f_new, l_new=1.0):
        """Efficiently retrain the surrogate model using the Schur complement.

        Parameters
        ----------
        x_new : ndarray, shape (m2,d,)
            New array of points where function values are known. m2 is the number of new sampling points.
        f_new : ndarray, shape (m2,p)
            Array of new function values at ``x_new``.
        l_new : float or ndarray, shape (m2,)
            Internal parameter. Width of the kernel on this new points.
        """

        m_new, d_new = x_new.shape

        if np.isscalar(l_new):
            self.l_new = self.l[0] * np.ones(m_new)

        # Calculation of D block

        R = euclidean_distances(x_new, x_new) ** 2

        for i in range(m_new):
            R[i, :] = R[i, :] / self.l_new**2

        D = np.exp(-R / 2)
        D = D + self.epsilon * np.eye(m_new)

        # Calculation of B block

        R = euclidean_distances(self.x, x_new) ** 2
        G = np.copy(R.T)

        for i in range(self.m):
            R[i, :] = R[i, :] / self.l_new**2

        B = np.exp(-R / 2)

        # Calculation of C block

        for i in range(m_new):
            G[i, :] = G[i, :] / self.l**2

        C = np.exp(-G / 2)

        # Calculation of Schur inverse

        self.Ainv = schur_inverse(self.Ainv, B, C, D)

        # Update of x and f vector
        self.x = np.concatenate((self.x, x_new), axis=0)
        self.f = np.concatenate((self.f, f_new), axis=0)
        self.l = np.concatenate((self.l, self.l_new), axis=0)

        m, d = self.x.shape
        self.m = m

        self.s = np.dot(self.Ainv, self.f)

        return None

    def update(self, l=1.0):
        """Update internal parameters of the kernel.

        Parameters
        ----------
        l : float or ndarray, shape (d,), optional
            Internal parameter. Width of the kernel.
        """
        self.l = l
