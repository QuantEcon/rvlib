from numba import vectorize, jit, jitclass
from numba import int32, float32

import numpy as np
from math import inf, lgamma

from specials import digamma

import _rmath_ffi
from numba import cffi_support

cffi_support.register_module(_rmath_ffi)

# shut down divide by zero warnings for now
import warnings
warnings.filterwarnings("ignore")

import yaml
with open("metadata.yml", 'r') as ymlfile:
    mtdt = yaml.load(ymlfile)

# --------------------------------------------------
# docstring following Spencer Lyon's distcan package
# https://github.com/spencerlyon2/distcan.git
# --------------------------------------------------

univariate_class_docstr = r"""
Construct a distribution representing {name_doc} random variables. The pdf
of the distribution is given by

.. math::

    {pdf_tex}

Parameters
----------
{param_list}

Attributes
----------
{param_attributes}
location: scalar(float)
    lcoation of the distribution
scale: scalar(float)
    scale of the distribution 
shape: scalar(float)
    shape of the distribution
mean :  scalar(float)
    mean of the distribution
median: scalar(float)
    median of the distribution
mode :  scalar(float)
    mode of the distribution
var :  scalar(float)
    var of the distribution
std :  scalar(float)
    std of the distribution
skewness :  scalar(float)
    skewness of the distribution
kurtosis :  scalar(float)
    kurtosis of the distribution
isplatykurtic :  Boolean
    boolean indicating if d.kurtosis > 0
isleptokurtic :  bool
    boolean indicating if d.kurtosis < 0
ismesokurtic :  bool
    boolean indicating if d.kurtosis == 0
entropy :  scalar(float)
    entropy value of the distribution
"""

param_str = "{name_doc} : {kind}\n    {descr}"


def _create_param_list_str(names, descrs, kinds="scalar(float)"):

    names = (names, ) if isinstance(names, str) else names
    names = (names, ) if isinstance(names, str) else names

    if isinstance(kinds, (list, tuple)):
        if len(names) != len(kinds):
            raise ValueError("Must have same number of names and kinds")

    if isinstance(kinds, str):
        kinds = [kinds for i in range(len(names))]

    if len(descrs) != len(names):
        raise ValueError("Must have same number of names and descrs")


    params = []
    for i in range(len(names)):
        n, k, d = names[i], kinds[i], descrs[i]
        params.append(param_str.format(name_doc=n, kind=k, descr=d))

    return str.join("\n", params)


def _create_class_docstr(name_doc, param_names, param_descrs,
                         param_kinds="scalar(float)",
                         pdf_tex=r"\text{not given}", **kwargs):
    param_list = _create_param_list_str(param_names, param_descrs,
                                        param_kinds)

    param_attributes = str.join(", ", param_names) + " : See Parameters"

    return univariate_class_docstr.format(**locals())

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

    # set docstring
    __doc__ = _create_class_docstr(**mtdt['Normal'])

    def __init__(self, mu, sigma):
        self.mu, self.sigma = mu, sigma

    def __str__(self):
        return "Normal(mu=%.5f, sigma=%.5f)" %(self.params)

    def __repr__(self):
        return self.__str__()

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

    # set docstring
    __doc__ = _create_class_docstr(**mtdt['Chisq'])

    def __init__(self, v):
        self.v = v

    def __str__(self):
        return "ChiSquared(k=%.5f)" %(self.params)

    def __repr__(self):
        return self.__str__()

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

    