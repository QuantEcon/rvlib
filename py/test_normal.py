"""
Test Normal distribution with the Numba-fied versions of the Rmath functions.

This is from the corresponding Julia package.
Normal(μ,σ)
The *Normal distribution* with mean `μ` and standard deviation `σ` has probability density function
$f(x; \mu, \sigma) = \frac{1}{\sqrt{2 \pi \sigma^2}}
\exp \left( - \frac{(x - \mu)^2}{2 \sigma^2} \right)$


Normal()          # standard Normal distribution with zero mean and unit variance
Normal(mu)        # Normal distribution with mean mu and unit variance
Normal(mu, sig)   # Normal distribution with mean mu and variance sig^2
params(d)         # Get the parameters, i.e. (mu, sig)
mean(d)           # Get the mean, i.e. mu
std(d)            # Get the standard deviation, i.e. sig

External links

[Normal distribution on Wikipedia](http://en.wikipedia.org/wiki/Normal_distribution)
"""

import rmath_norm
import numpy as np
from numba import vectorize, jit
from math import inf

#  ------
#  Normal
#  ------

class Normal():
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

    # ===================
    # Parameter retrieval
    # ===================

    @property
    def params(self):
        """Returns parameters."""
        return (self.mu, self.sigma)

    @property
    def location(self):
        """Returns parameters."""
        return self.mu

    @property
    def scale(self):
        """Returns parameters."""
        return self.sigma

    # ==========
    # Statistics
    # ==========

    @property
    def mean(self):
        """Returns mean."""
        return self.mu

    @property
    def median(self):
        """Returns median."""
        return self.mu

    @property
    def mode(self):
        """Returns mode."""
        return self.mu

    @property
    def var(self):
        """Returns variance."""
        return self.sigma ** 2

    @property
    def std(self):
        """Returns standard deviation."""
        return self.sigma

    @property
    def skewness(self):
        """Returns skewness."""
        return 0.0

    @property
    def kurtosis(self):
        """Returns kurtosis."""
        return 0.0

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

    @property
    def entropy(self):
        """Returns entropy."""
        return 0.5 * (np.log(2*np.pi) + 1.0) + np.log(self.sigma)

    @vectorize(nopython=True)
    def mgf(self, x):
        """Evaluate moment generating function at x."""
        return np.exp(x * self.mu + 0.5 * self.sigma**2 * x**2)

    @vectorize(nopython=True)
    def cf(self, x):
        """Evaluate characteristic function at x."""
        return np.exp(1j * x * self.mu - 0.5 * self.sigma**2 * x**2)

    # ==========
    # Evaluation
    # ==========

    def insupport(self, x):
        """When x is a scalar, it returns whether x is within the support of
        the distribution. When x is an array, it returns whether every element"""
        return -inf < x < inf

    def pdf(self, x):
        """The pdf value(s) evaluated at x."""
        return rmath_norm.norm_pdf(self.mu, self.sigma, x)

    def logpdf(self, x):
        """The logarithm of the pdf value(s) evaluated at x."""
        return rmath_norm.norm_logpdf(self.mu, self.sigma, x)

    def loglikelihood(self, x):
        """The log-likelihood of the Normal distribution w.r.t. all
        samples contained in array x."""
        return sum(rmath_norm.norm_logpdf(self.mu, self.sigma, x))

    def cdf(self, x):
        """The cdf value(s) evaluated at x."""
        return rmath_norm.norm_cdf(self.mu, self.sigma, x)

    def logcdf(self, x):
        """The logarithm of the cdf value(s) evaluated at x."""
        return rmath_norm.norm_logcdf(self.mu, self.sigma, x)

    def logcdf(self, x):
        """The logarithm of the cdf value(s) evaluated at x."""
        return rmath_norm.norm_logcdf(self.mu, self.sigma, x)

    def ccdf(self, x):
        """The complementary cdf evaluated at x, i.e. 1- cdf(x)."""
        return rmath_norm.norm_ccdf(self.mu, self.sigma, x)

    def logccdf(self, x):
        """The logarithm of the complementary cdf evaluated at x."""
        return rmath_norm.norm_logccdf(self.mu, self.sigma, x)

    def quantile(self, q):
        """The quantile value evaluated at q."""
        return rmath_norm.norm_invcdf(self.mu, self.sigma, q)

    def cquantile(self, q):
        """The complementary quantile value evaluated at q."""
        return rmath_norm.norm_invccdf(self.mu, self.sigma, q)

    def invlogcdf(self, lq):
        """The inverse function of logcdf."""
        return rmath_norm.norm_invlogcdf(self.mu, self.sigma, lq)

    def invlogccdf(self, lq):
        """The inverse function of logccdf."""
        return rmath_norm.norm_invlogccdf(self.mu, self.sigma, lq)

    # ========
    # Sampling
    # ========

    def rand(self, *n):
        """Generates a random draw from the distribution."""
        if len(n) == 0:
            n = (1,)

        out = np.empty(n)
        for i, _ in np.ndenumerate(out):
            out[i] = rmath_norm.norm_rand(self.mu, self.sigma)

        return out
