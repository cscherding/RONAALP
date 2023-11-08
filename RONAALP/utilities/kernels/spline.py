import numpy as np
import numpy.linalg as nla
import scipy.linalg as la
from scipy.optimize import differential_evolution, NonlinearConstraint
from sklearn.metrics.pairwise import euclidean_distances
from scipy.special import comb
from itertools import combinations_with_replacement
from RONAALP.utilities import schur_inverse

EPS = np.finfo(np.float64).eps


def monomial_powers(ndim, degree):
    """Return the powers for each monomial in a polynomial.

    Parameters
    ----------
    ndim : int
        Number of variables in the polynomial.
    degree : int
        Degree of the polynomial.

    Returns
    -------
    (nmonos, ndim) int ndarray
        Array where each row contains the powers for each variable in a
        monomial.
    """

    nmonos = comb(degree + ndim, ndim, exact=True)
    out = np.zeros((nmonos, ndim), dtype=int)
    count = 0
    for deg in range(degree + 1):
        for mono in combinations_with_replacement(range(ndim), deg):
            # `mono` is a tuple of variables in the current monomial with
            # multiplicity indicating power (e.g., (0, 1, 1) represents x*y**2)
            for var in mono:
                out[count, var] += 1

            count += 1

    return out


def polynomial_matrix(x, powers):
    """Evaluate monomials, with exponents from `powers`, at `x`."""
    num_rows = x.shape[0]
    num_cols = powers.shape[0]
    out = np.empty((num_rows, num_cols))
    for j in range(num_cols):
        out[:, j] = np.prod(np.power(x, powers[j]), axis=1)
    return out


class RBF_Spline:
    """RBF thinplate spline kernel class.

    Parameters
    ----------
    degree  : int, default = 1
            Highest degree of added polynomial terms.
    epsilon : float, default = 1e-3
            Smoothing parameter.
    phi_min : float, default = 5e-3
            Distance threshold for extrapolation detection.

    Attributes
    ----------
    s : ndarray, shape(m+monomes(degree),)
        RBF coefficients: first m coefficients correspond to each kernel, followed by monomes(degree) coeffiecients for polynomial terms.
    x : ndarray, shape (m,d,)
        Array of points where function values are known. m is the number of sampling points and d is the number of dimensions.

    f  : ndarray, shape (m,p,)
        Array of function values at ``x``.
    f0 : ndarray, shape (m+monomes(degree),p,)
        Array of function values at ``x`` supplemented with zeros for polyharmonic terms.

    Kinv : ndarray, shape (m,m,)
        Inverse of the RBF kernel matrix Phi.
    """

    def __init__(self, epsilon=1e-3, degree=1, phi_min=5.0e-3):
        self.s = None
        self.x = None
        self.f = None
        self.epsilon = epsilon
        self.phi_min = phi_min
        self.degree = degree

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
        """

        self.x = x
        self.f = f

        m, d = x.shape
        mm, p = f.shape

        assert mm == m

        self.m = m
        self.d = d
        self.p = p

        # 1. Kernel matrix
        # Calculate distance matrix
        R = euclidean_distances(x, x)

        # Apply kernel
        Phi = R * np.log(np.power(R, R))

        # Apply smoothing
        Phi = Phi + self.epsilon * np.eye(m)

        # Find inverse of kernel matrix
        self.Kinv = la.inv(Phi)

        # 2. Polynomial matrix
        self.powers = monomial_powers(d, self.degree)
        r = self.powers.shape[0]
        self.r = r

        # Shift and scale the polynomial domain to be between -1 and 1
        mins = np.min(x, axis=0)
        maxs = np.max(x, axis=0)
        shift = (maxs + mins) / 2
        scale = (maxs - mins) / 2
        # The scale may be zero if there is a single point or all the points have
        # the same value for some dimension. Avoid division by zero by replacing
        # zeros with ones.
        scale[scale == 0.0] = 1.0

        xhat = (x - shift) / scale

        self.shift = shift
        self.scale = scale

        self.P = polynomial_matrix(xhat, self.powers)

        print("P size", self.P.shape)
        # 3. Build and solve full system with Schur complement

        Ainv = schur_inverse(self.Kinv, self.P, self.P.T, np.zeros((r, r)))

        self.f0 = np.zeros((m + r, p))
        self.f0[:m] = self.f

        self.s = np.dot(Ainv, self.f0)

        return Phi

    def evaluate(self, y):
        """Evaluate surrogate model at given points.

        Parameters
        ----------
        y : ndarray, shape (n,d,)
            Array of points where we want to evaluate the surrogate model.

        Returns
        -------
        f : ndarray, shape(n,p,)
            Array of interpolated values.
        extrp: ndarray, shape(n,)
            Array of extrapolation flag:
                0 -> interpolation,
                1 -> extrapolation.
        """
        if self.s is None:
            return None

        y = np.array(y)
        n, d = y.shape

        # Compute distance matrix
        R = euclidean_distances(y, self.x)

        # Apply kernel
        Phi = R * np.log(np.power(R, R))

        # Calculate extrapolation flag
        SigMax = np.min(R, axis=1)
        SigMax = np.abs(SigMax * np.log(np.power(SigMax, SigMax)))
        extrp = np.floor(
            (1 / self.phi_min) * np.minimum(self.phi_min * np.ones((n)), SigMax)
        )

        # Polynomial matrix
        yhat = (y - self.shift) / self.scale
        P = polynomial_matrix(yhat, self.powers)

        B = np.zeros((n, self.m + self.r))

        # Build full system
        B[:n, : self.m] = Phi
        B[:n, self.m :] = P

        # Evaluation of interpolant
        f = np.dot(B, self.s)

        return f, extrp

    def retrain(self, x_new, f_new):
        """Retrain the surrogate model in a brute force approach.

        Parameters
        ----------
        x_new : ndarray, shape (m2,d,)
            New array of points where function values are known. m2 is the number of new sampling points.
        f_new : ndarray, shape (m2,p)
            Array of new function values at ``x_new``.
        """
        # concatenate new x and f vectors and update sizes
        self.x = np.concatenate((self.x, x_new), axis=0)
        self.f = np.concatenate((self.f, f_new), axis=0)

        m, _ = self.x.shape
        self.m = m

        # 1. Kernel matrix
        # Calculate distance matrix
        R = euclidean_distances(self.x, self.x)

        # Apply kernel
        Phi = R * np.log(np.power(R, R))

        # Apply smoothing
        Phi = Phi + self.epsilon * np.eye(m)

        # Find inverse of kernel matrix
        self.Kinv = la.inv(Phi)

        # 2. Polynomial matrix

        xhat = (self.x - self.shift) / self.scale

        self.P = polynomial_matrix(xhat, self.powers)

        # 3. Build and solve full system with Schur complement

        Ainv = schur_inverse(self.Kinv, self.P, self.P.T, np.zeros((self.r, self.r)))

        self.f0 = np.zeros((self.m + self.r, self.p))
        self.f0[:m] = self.f

        self.s = np.dot(Ainv, self.f0)

        return None

    def retrain_schur(self, x_new, f_new):
        """Efficiently retrain the surrogate model using the Schur complement.

        Parameters
        ----------
        x_new : ndarray, shape (m2,d,)
            New array of points where function values are known. m2 is the number of new sampling points.
        f_new : ndarray, shape (m2,p)
            Array of new function values at ``x_new``.
        """

        m_new, d_new = x_new.shape

        assert m_new == f_new.shape[0]

        # Calculation of D block

        R = euclidean_distances(x_new, x_new)
        D = R * np.log(np.power(R, R)) + self.epsilon * np.eye(m_new)

        # Calculation of B block
        R = euclidean_distances(self.x, x_new)

        B = R * np.log(np.power(R, R))

        # Calculation of C block

        G = np.copy(R.T)
        C = G * np.log(np.power(G, G))

        Kinv = schur_inverse(self.Kinv, B, C, D)

        self.Kinv = Kinv

        # Polynomial term

        xhat = (x_new - self.shift) / self.scale
        P2 = polynomial_matrix(xhat, self.powers)

        P = np.concatenate((self.P, P2), axis=0)
        self.P = P

        Ainv = schur_inverse(self.Kinv, self.P, self.P.T, np.zeros((self.r, self.r)))

        # Update of x and f vector
        self.x = np.concatenate((self.x, x_new), axis=0)
        self.f = np.concatenate((self.f, f_new), axis=0)

        m, d = self.x.shape

        self.f0 = np.zeros((m + self.r, self.p))
        self.f0[:m] = self.f
        self.m = m

        # Update RBF coefficients
        self.s = np.dot(Ainv, self.f0)

        return None
