"""
Univariate distributions
"""

from numba import vectorize, jit
from math import sqrt, log, pi, exp, inf, lgamma
import numpy as np
from scipy.special import digamma
from rmath_wrap import DistFromRmath

__all__ = ["Normal", "Chisq"]


#  ------
#  Normal
#  ------

class Normal(DistFromRmath):
    """
    The Normal distribution with mean mu and standard deviation sigma.

    Parameters
    ----------
    mu : scalar(float)
        Mean of the normal distribution
    sigma : scalar(float)
        Standard deviaton of the normal distribution
    """

    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma


    # set dist_name for rmath before calling super's __init__
        self.dist_name = "rmath_norm.norm_"
    # set parameters callable for super
        self._rmath_params = (mu, sigma)
    # call super
        super(Normal, self).__init__()

    @property
    def params(self):
        """Returns parameters."""
        return (self.mu, self.sigma)


#  ----------- #
#  Chi Squared #
#  ----------- #

class Chisq(DistFromRmath):
    """
    The Chi-squared distribution with nu, "v", degrees of freedom.

    Parameters
    ----------
    v : scalar(float)
        Degrees of freedom
    """

    def __init__(self, v):
        self.v = v

    # set dist_name for rmath before calling super's __init__
        self.dist_name = "rmath_chisq.chisq_"
    # set parameters callable for super
        self._rmath_params = (v,)
    # call super
        super(Chisq, self).__init__()

    @property
    def params(self):
        """Returns parameters."""
        return (self.v,)