from numba import vectorize, jit, jitclass
from numba import int32, float32

import numpy as np
from math import inf, lgamma
# from scipy.special import digamma
from digamma import digamma

import _rmath_ffi
from numba import cffi_support

cffi_support.register_module(_rmath_ffi)

# shut down divide by zero warnings for now
import warnings
warnings.filterwarnings("ignore")

# ============================= NEW DISTRIBUTION =================================
dnorm = _rmath_ffi.lib.dnorm
pnorm = _rmath_ffi.lib.pnorm
qnorm = _rmath_ffi.lib.qnorm

@vectorize(nopython=True)
def norm_pdf(mu, sigma, x):
    return dnorm(x, mu, sigma, 0)


@vectorize(nopython=True)
def norm_logpdf(mu, sigma, x):
    return dnorm(x, mu, sigma, 1)


@vectorize(nopython=True)
def norm_cdf(mu, sigma, x):
    return pnorm(x, mu, sigma, 1, 0)


@vectorize(nopython=True)
def norm_ccdf(mu, sigma, x):
    return pnorm(x, mu, sigma, 0, 0)


@vectorize(nopython=True)
def norm_logcdf(mu, sigma, x):
    return pnorm(x, mu, sigma, 1, 1)


@vectorize(nopython=True)
def norm_logccdf(mu, sigma, x):
    return pnorm(x, mu, sigma, 0, 1)


@vectorize(nopython=True)
def norm_invcdf(mu, sigma, q):
    return qnorm(q, mu, sigma, 1, 0)


@vectorize(nopython=True)
def norm_invccdf(mu, sigma, q):
    return qnorm(q, mu, sigma, 0, 0)


@vectorize(nopython=True)
def norm_invlogcdf(mu, sigma, lq):
    return qnorm(lq, mu, sigma, 1, 1)


@vectorize(nopython=True)
def norm_invlogccdf(mu, sigma, lq):
    return qnorm(lq, mu, sigma, 0, 1)

rnorm = _rmath_ffi.lib.rnorm

@jit(nopython=True)
def norm_rand(mu, sigma):
    return rnorm(mu, sigma)


@vectorize(nopython=True)
def norm_mgf(mu, sigma, x):
    return np.exp(x * mu + 0.5 * sigma**2 * x**2)

@vectorize(nopython=True)
def norm_cf(mu, sigma, x):
    return np.exp(1j * x * mu - 0.5 * sigma**2 * x**2)

#  ------
#  Normal
#  ------

spec = [
    ('mu', float32), ('sigma', float32)
]

@jitclass(spec)
class Normal():
    """
    add doc later
    """

    def __init__(self, mu, sigma):
        self.mu, self.sigma = mu, sigma

    # ===================
    # Parameter retrieval
    # ===================

    @property
    def params(self):
        """Returns parameters."""
        return (self.mu, self.sigma)

    @property
    def location(self):
        """Returns lcoation parameter if exists."""
        return self.mu

    @property
    def scale(self):
        """Returns scale parameter if exists."""
        return self.sigma

    @property
    def shape(self):
        """Returns shape parameter if exists."""
        return None

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
        return self.quantile(.5)

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

    def mgf(self, x):
        """Evaluate moment generating function at x."""
        return norm_mgf(self.mu, self.sigma, x)

    def cf(self, x):
        """Evaluate characteristic function at x."""
        return norm_cf(self.mu, self.sigma, x)

    # ==========
    # Evaluation
    # ==========

    def insupport(self, x):
        """When x is a scalar, it returns whether x is within
        the support of the distribution. When x is an array,
        it returns whether every element."""
        return -inf < x < inf

    def pdf(self, x):
        """The pdf value(s) evaluated at x."""
        return norm_pdf(self.mu, self.sigma, x)
    
    def logpdf(self, x):
        """The logarithm of the pdf value(s) evaluated at x."""
        return norm_logpdf(self.mu, self.sigma, x)

    def loglikelihood(self, x):
        """The log-likelihood of the Normal distribution w.r.t. all
        samples contained in array x."""
        return sum(norm_logpdf(self.mu, self.sigma, x))
    
    def cdf(self, x):
        """The cdf value(s) evaluated at x."""
        return norm_cdf(self.mu, self.sigma, x)
    
    def ccdf(self, x):
        """The complementary cdf evaluated at x, i.e. 1 - cdf(x)."""
        return norm_ccdf(self.mu, self.sigma, x)
    
    def logcdf(self, x):
        """The logarithm of the cdf value(s) evaluated at x."""
        return norm_logcdf(self.mu, self.sigma, x)
    
    def logccdf(self, x):
        """The logarithm of the complementary cdf evaluated at x."""
        return norm_logccdf(self.mu, self.sigma, x)
    
    def quantile(self, q):
        """The quantile value evaluated at q."""
        return norm_invcdf(self.mu, self.sigma, q)
    
    def cquantile(self, q):
        """The complementary quantile value evaluated at q."""
        return norm_invccdf(self.mu, self.sigma, q)
    
    def invlogcdf(self, lq):
        """The inverse function of logcdf."""
        return norm_invlogcdf(self.mu, self.sigma, lq)
    
    def invlogccdf(self, lq):
        """The inverse function of logccdf."""
        return norm_invlogccdf(self.mu, self.sigma, lq)
    
    # ========
    # Sampling
    # ========
    
    def rand(self, n):
        """Generates a random draw from the distribution."""
        out = np.empty(n)
        for i, _ in np.ndenumerate(out):
            out[i] = norm_rand(self.mu, self.sigma)
        return out

    
# ============================= NEW DISTRIBUTION =================================
dchisq = _rmath_ffi.lib.dchisq
pchisq = _rmath_ffi.lib.pchisq
qchisq = _rmath_ffi.lib.qchisq

@vectorize(nopython=True)
def chisq_pdf(v, x):
    return dchisq(x, v, 0)


@vectorize(nopython=True)
def chisq_logpdf(v, x):
    return dchisq(x, v, 1)


@vectorize(nopython=True)
def chisq_cdf(v, x):
    return pchisq(x, v, 1, 0)


@vectorize(nopython=True)
def chisq_ccdf(v, x):
    return pchisq(x, v, 0, 0)


@vectorize(nopython=True)
def chisq_logcdf(v, x):
    return pchisq(x, v, 1, 1)


@vectorize(nopython=True)
def chisq_logccdf(v, x):
    return pchisq(x, v, 0, 1)


@vectorize(nopython=True)
def chisq_invcdf(v, q):
    return qchisq(q, v, 1, 0)


@vectorize(nopython=True)
def chisq_invccdf(v, q):
    return qchisq(q, v, 0, 0)


@vectorize(nopython=True)
def chisq_invlogcdf(v, lq):
    return qchisq(lq, v, 1, 1)


@vectorize(nopython=True)
def chisq_invlogccdf(v, lq):
    return qchisq(lq, v, 0, 1)

rchisq = _rmath_ffi.lib.rchisq

@jit(nopython=True)
def chisq_rand(v):
    return rchisq(v)


@vectorize(nopython=True)
def chisq_mgf(v, x):
    return (1.0 - 2.0 * x)**(-v * 0.5)

@vectorize(nopython=True)
def chisq_cf(v, x):
    return (1.0 - 2.0 * 1j * x)**(-v * 0.5)

#  ------
#  Chisq
#  ------

spec = [
    ('v', int32)
]

@jitclass(spec)
class Chisq():
    """
    add doc later
    """

    def __init__(self, v):
        self.v = v

    # ===================
    # Parameter retrieval
    # ===================

    @property
    def params(self):
        """Returns parameters."""
        return (self.v)

    @property
    def location(self):
        """Returns lcoation parameter if exists."""
        return None

    @property
    def scale(self):
        """Returns scale parameter if exists."""
        return None

    @property
    def shape(self):
        """Returns shape parameter if exists."""
        return self.v

    # ==========
    # Statistics
    # ==========

    @property
    def mean(self):
        """Returns mean."""
        return self.v

    @property
    def median(self):
        """Returns median."""
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

    @property
    def entropy(self):
        """Returns entropy."""
        return .5 * self.v +  np.log(2.0) + lgamma(.5 * self.v) + (1.0 - .5 * self.v) * digamma(.5 * self.v)

    def mgf(self, x):
        """Evaluate moment generating function at x."""
        return chisq_mgf(self.v, x)

    def cf(self, x):
        """Evaluate characteristic function at x."""
        return chisq_cf(self.v, x)

    # ==========
    # Evaluation
    # ==========

    def insupport(self, x):
        """When x is a scalar, it returns whether x is within
        the support of the distribution. When x is an array,
        it returns whether every element."""
        return 0 <= x < inf

    def pdf(self, x):
        """The pdf value(s) evaluated at x."""
        return chisq_pdf(self.v, x)
    
    def logpdf(self, x):
        """The logarithm of the pdf value(s) evaluated at x."""
        return chisq_logpdf(self.v, x)

    def loglikelihood(self, x):
        """The log-likelihood of the Normal distribution w.r.t. all
        samples contained in array x."""
        return sum(chisq_logpdf(self.v, x))
    
    def cdf(self, x):
        """The cdf value(s) evaluated at x."""
        return chisq_cdf(self.v, x)
    
    def ccdf(self, x):
        """The complementary cdf evaluated at x, i.e. 1 - cdf(x)."""
        return chisq_ccdf(self.v, x)
    
    def logcdf(self, x):
        """The logarithm of the cdf value(s) evaluated at x."""
        return chisq_logcdf(self.v, x)
    
    def logccdf(self, x):
        """The logarithm of the complementary cdf evaluated at x."""
        return chisq_logccdf(self.v, x)
    
    def quantile(self, q):
        """The quantile value evaluated at q."""
        return chisq_invcdf(self.v, q)
    
    def cquantile(self, q):
        """The complementary quantile value evaluated at q."""
        return chisq_invccdf(self.v, q)
    
    def invlogcdf(self, lq):
        """The inverse function of logcdf."""
        return chisq_invlogcdf(self.v, lq)
    
    def invlogccdf(self, lq):
        """The inverse function of logccdf."""
        return chisq_invlogccdf(self.v, lq)
    
    # ========
    # Sampling
    # ========
    
    def rand(self, n):
        """Generates a random draw from the distribution."""
        out = np.empty(n)
        for i, _ in np.ndenumerate(out):
            out[i] = chisq_rand(self.v)
        return out

    