"""
Multivariate distributions

@authors :  Daniel Csaba <daniel.csaba@nyu.edu>
            Spencer Lyon <spencer.lyon@stern.nyu.edu>
@date : 2016-07-26
"""

import numpy as np
import numba
from numba import vectorize, jit
from numba import vectorize, jit, jitclass
from numba import int32, int64, float32, float64

from numba import cffi_support
from . import _rmath_ffi
cffi_support.register_module(_rmath_ffi)


# -------------------
# Multivariate Normal
# -------------------

# ========================================================================
# no sufficient linalg support in numba, linalg.det in next release though
# ========================================================================
#@vectorize(nopython=True)
#def mvnormal_pdf(mu, sigma, dim, x):
#    return np.sqrt((2*np.pi)**dim*np.linalg.det(sigma))*np.exp(-.5*(x - mu)@np.linalg.inv(sigma)@(x - mu))


rnorm = _rmath_ffi.lib.rnorm

# random number generation from independent standard normals

@jit(nopython=True)
def mvnormal_rand(dim):
    out = np.empty(dim)
    for i, _ in np.ndenumerate(out):
        out[i] = rnorm(0, 1)
    return out

# =============================================================================
# for now from http://drsfenner.org/blog/2016/02/basic-cholesky-implementation/
# let's write our own to avoid licensing issues
# =============================================================================
@jit(numba.double[:,:](numba.double[:,:]), nopython=True)
def cholesky(A):
    """
       Performs a Cholesky decomposition of on symmetric, pos-def A.
       Returns lower-triangular L (full sized, zeroed above diag)
    """
    n = A.shape[0]
    L = np.empty_like(A)

    # Perform the Cholesky decomposition
    for row in range(n):
        for col in range(row+1):
            tmp_sum = 0.0
            for j in range(col):
                tmp_sum += L[row,j] * L[col,j]
            if (row == col): 
                # diag elts.
                L[row,col] = np.sqrt(A[row,row] - tmp_sum)
            else:
                # off diag elts.
                L[row,col] = (1.0 / L[col,col] * (A[row,col] - tmp_sum))
        L[row, row+1:] = 0.0
    return L


spec = [
        ('mu', float64[:]),     # array field for mean 
        ('sigma', float64[:,:]),    # array field for covariance matrix
        ('dim', int64)          # scalar for dimension  
    ]

@jitclass(spec)
class MvNormal_c(object):

    def __init__(self, mu, sigma):
        self.mu = mu #np.asarray(mu, dtype=np.float32)
        self.sigma = sigma #np.asarray(sigma, dtype=np.float32)
        self.dim = mu.size

        # =====================================
        # raise ValueError not working in numba
        # =====================================

        #Check if 'mu' and 'sigma' are compatible in size
        # ***FAIL at typeinference: "constant inference not possible for $const57.2 % $57.7"
        #if sigma.shape != (self.dim, self.dim):
        #   raise ValueError("Array 'sigma' must be a square matrix of dimension (%d, %d)" % (self.dim, self.dim))

        # Check if 'sigma' is symmetric
        # ***FAIL at typeinference: "constant inference not possible for $const57.2 % $57.7"
        #if not (self.sigma.T == self.sigma).all():
        #   raise ValueError("Array 'sigma' must be symmetric.")



    def __str__(self):
        return "MvNormal(mu=%s, sigma=%s)" % self.params

    def __repr__(self):
        return self.__str__()


    # ===================
    # Parameter retrieval
    # ===================

    @property
    def params(self):
        """Return the parameters."""
        return (self.mu, self.sigma)

    @property
    def length(self):
        """Return the length of random vector."""
        return self.dim

    # ==========
    # Statistics
    # ==========

    @property
    def mean(self):
        """Return the mean vector."""
        return self.mu

    @property
    def cov(self):
        """Return the covariance matrix."""
        return self.sigma

    @property
    def var(self):
        """Return the vector of component-wise variances."""
        return np.diag(self.sigma)

    @property
    def corr(self):
        """Return the correlation matrix."""  
        return np.diag(self.var**(-.5)) @ self.cov @ np.diag(self.var**(-.5))

# ======================================
# np.linalg.det in next release of numba
# ======================================
#   @property
#   def entropy(self):
#       """Return the entropy."""
#       return .5*(self.length*(1 + np.log(2*np.pi)) + np.log(np.linalg.det(self.sigma)))


    # ==========
    # Evaluation
    # ==========

    def insupport(self, x):
        """When 'x' is a vector, return whether 'x' is 
        within the support of the distribution. When 'x' 
        is a matrix, return whether every column of 'x' 
        is within the support of the distribution."""
        x = np.array(x, dtype=float64)
        # can't raise valueerror in numba
        #if x.shape[0] != self.dim:
        #   raise ValueError("Array 'x' must be of dimension (%d)" % self.dim)
        return (-np.inf < x).all() and (x < np.inf).all() and x.shape[0] <= self.dim

# =====================================
# no sufficient linalg support in numba
# =====================================
#   def pdf(self, x):
#       """Return the probabilty density evaluated at 'x'. If 'x' is a 
#       vector then return the result as a scalar. If 'x' is a matrix
#       then return the result as an array."""
#
#       # pdf only exists if 'sigma' is positive definite
#       if not np.all(np.linalg.eigvals(self.sigma) > 0):
#           raise ValueError("The pdf only exists if 'sigma' is positive definite.")
#       return mvnormal_pdf(self.mu, self.sigma, self.dim, x)

#   def logpdf(self, x):
#       """Return the logarithm of the probabilty density evaluated
#       at 'x'. If 'x' is a vector then return the result as a scalar.
#       If 'x' is a matrix then return the result as an array."""
#       return np.log(self.pdf(x))

    # ========
    # Sampling
    # ========

    def rand(self):
        """Sample n vectors from the distribution. This returns 
        a matrix of size (dim, n), where each column is a sample."""
        L = cholesky(self.sigma)
        # ======================================================
        # not working for n dimensional yet
        #out = np.empty(n)
        #for i, _ in np.ndenumerate(out):
            #out[:, i] = L @ mvnormal_rand(self.dim) + self.mean
        # ======================================================
        return L @ mvnormal_rand(self.dim) + self.mu

# define wrapper to overcome failure of type inference
def MvNormal(mu, sigma):
    return MvNormal_c(np.array(mu, dtype=np.float64), np.array(sigma, dtype=np.float64))
