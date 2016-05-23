"""
Test ChiSq distribution with the Numba-fied versions of the Rmath functions.

This is from the corresponding Julia package.

Chisq(ν)
The *Chi squared distribution* (typically written χ²) with `ν` degrees of freedom has the
probability density function
$f(x; k) = \frac{x^{k/2 - 1} e^{-x/2}}{2^{k/2} \Gamma(k/2)}, \quad x > 0.$
If `ν` is an integer, then it is the distribution of the sum of squares of `ν` independent standard [`Normal`](:func:`Normal`) variates.
```julia
Chisq(k)     # Chi-squared distribution with k degrees of freedom
params(d)    # Get the parameters, i.e. (k,)
dof(d)       # Get the degrees of freedom, i.e. k
```
External links
* [Chi-squared distribution on Wikipedia](http://en.wikipedia.org/wiki/Chi-squared_distribution)
"""

import rmath_chisq
import numpy as np
from numba import vectorize, jit
from math import inf, lgamma
from scipy.special import digamma

#  -----------
#  Chi-squared
#  -----------

class Chisq():
    """
    The Chi-squared distribution with nu, "v", degrees of freedom.

    Parameters
    ----------
    v : scalar(float)
        Degrees of freedom
    """

    def __init__(self, v):
        self.v = v

    # ===================
    # Parameter retrieval
    # ===================

    @property
    def params(self):
        """Returns parameters."""
        return (self.v,)

    @property
    def shape(self):
        """Returns shape as degrees of freedom."""
        return self.v

    # ==========
    # Statistics
    # ==========

    @property
    def mean(self):
        """Returns mean."""
        return self.v

    # note: either @property is on, or approx feature is working
    def median(self, approx=False):
        """Returns median. If approx==True, returns 
        approximation of median."""
        if approx:
            return self.v * (1.0 - 2.0 / (9.0 * self.v))**3
        else:
            return self.quantile(.5)

    @property
    def mode(self):
        """Returns mode."""
        return max(self.v - 2, 0)

    @property
    def var(self):
        """Returns variance."""
        return self.v * 2.0

    @property
    def std(self):
        """Returns standard deviation."""
        return np.sqrt(self.v * 2.0)

    @property
    def skewness(self):
        """Returns skewness."""
        return np.sqrt(8.0 / self.v)

    @property
    def kurtosis(self):
        """Returns kurtosis."""
        return 12.0 / self.v

    @property
    def isplatykurtic(self):
        """Kurtosis being greater than zero."""
        return self.kurtosis > 0

    @property
    def isleptokurtic(self):
        """Kurtosis being smaller than zero."""
        return self.kurtosis < 0

    @property
    def ismesokurtic(self):
        """Kurtosis being equal to zero."""
        return self.kurtosis == 0.0

    @property # does it make sense to jit this?
    def entropy(self):
        """Returns entropy."""
        hv = .5 * self.v
        return hv +  np.log(2.0) + lgamma(hv) + (1.0 - hv) * digamma(hv)

    @vectorize(nopython=True)
    def mgf(self, x):
        """Evaluate moment generating function at x."""
        return (1.0 - 2.0 * x)**(-self.v * 0.5)

    @vectorize(nopython=True)
    def cf(self, x):
        """Evaluate characteristic function at x."""
        return (1.0 - 2.0 * 1j * t)**(-self.v * 0.5)

    # ==========
    # Evaluation
    # ==========

    def insupport(self, x):
        """When x is a scalar, it returns whether x is within the support of
        the distribution. When x is an array, it returns whether every element"""
        return 0 <= x < inf

    def pdf(self, x):
        """The pdf value(s) evaluated at x."""
        return rmath_chisq.chisq_pdf(self.v, x)

    def logpdf(self, x):
        """The logarithm of the pdf value(s) evaluated at x."""
        return rmath_chisq.chisq_logpdf(self.v, x)

    def loglikelihood(self, x):
        """The log-likelihood of the Normal distribution w.r.t. all
        samples contained in array x."""
        return sum(rmath_chisq.chisq_logpdf(self.v, x))

    def cdf(self, x):
        """The cdf value(s) evaluated at x."""
        return rmath_chisq.chisq_cdf(self.v, x)

    def logcdf(self, x):
        """The logarithm of the cdf value(s) evaluated at x."""
        return rmath_chisq.chisq_logcdf(self.v, x)

    def logcdf(self, x):
        """The logarithm of the cdf value(s) evaluated at x."""
        return rmath_chisq.chisq_logcdf(self.v, x)

    def ccdf(self, x):
        """The complementary cdf evaluated at x, i.e. 1 - cdf(x)."""
        return rmath_chisq.chisq_ccdf(self.v, x)

    def logccdf(self, x):
        """The logarithm of the complementary cdf evaluated at x."""
        return rmath_chisq.chisq_logccdf(self.v, x)

    def quantile(self, q):
        """The quantile value evaluated at q."""
        return rmath_chisq.chisq_invcdf(self.v, q)

    def cquantile(self, q):
        """The complementary quantile value evaluated at q."""
        return rmath_chisq.chisq_invccdf(self.v, q)

    def invlogcdf(self, lq):
        """The inverse function of logcdf."""
        return rmath_chisq.chisq_invlogcdf(self.v, lq)

    def invlogccdf(self, lq):
        """The inverse function of logccdf."""
        return rmath_chisq.chisq_invlogccdf(self.v, lq)

    # ========
    # Sampling
    # ========

    def rand(self, *n):
        """Generates a random draw from the distribution."""
        if len(n) == 0:
            n = (1,)

        out = np.empty(n)
        for i, _ in np.ndenumerate(out):
            out[i] = rmath_chisq.chisq_rand(self.v)

        return out
