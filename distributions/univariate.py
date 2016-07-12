from numba import vectorize, jit, jitclass
from numba import int32, float32

import numpy as np
from math import inf, gamma, lgamma, ceil, floor
from numpy.random import beta
from .specials import digamma

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
        """Returns location parameter if exists."""
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
        """The log-likelihood of the distribution w.r.t. all
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
        """Returns location parameter if exists."""
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
        """The log-likelihood of the distribution w.r.t. all
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

    
# ============================= NEW DISTRIBUTION =================================
dunif = _rmath_ffi.lib.dunif
punif = _rmath_ffi.lib.punif
qunif = _rmath_ffi.lib.qunif

@vectorize(nopython=True)
def unif_pdf(a, b, x):
    return dunif(x, a, b, 0)


@vectorize(nopython=True)
def unif_logpdf(a, b, x):
    return dunif(x, a, b, 1)


@vectorize(nopython=True)
def unif_cdf(a, b, x):
    return punif(x, a, b, 1, 0)


@vectorize(nopython=True)
def unif_ccdf(a, b, x):
    return punif(x, a, b, 0, 0)


@vectorize(nopython=True)
def unif_logcdf(a, b, x):
    return punif(x, a, b, 1, 1)


@vectorize(nopython=True)
def unif_logccdf(a, b, x):
    return punif(x, a, b, 0, 1)


@vectorize(nopython=True)
def unif_invcdf(a, b, q):
    return qunif(q, a, b, 1, 0)


@vectorize(nopython=True)
def unif_invccdf(a, b, q):
    return qunif(q, a, b, 0, 0)


@vectorize(nopython=True)
def unif_invlogcdf(a, b, lq):
    return qunif(lq, a, b, 1, 1)


@vectorize(nopython=True)
def unif_invlogccdf(a, b, lq):
    return qunif(lq, a, b, 0, 1)

runif = _rmath_ffi.lib.runif

@jit(nopython=True)
def unif_rand(a, b):
    return runif(a, b)


@vectorize(nopython=True)
def unif_mgf(a, b, x):
    return (np.exp(x * b) - np.exp(x * a))/(x * (b - a)) if x != 0 else 1

@vectorize(nopython=True)
def unif_cf(a, b, x):
    return (np.exp(1j * x * b) - np.exp(1j * x * a))/(1j * x * (b - a))

#  ------
#  Uniform
#  ------

spec = [
    ('a', float32), ('b', float32)
]

@jitclass(spec)
class Uniform():

    # set docstring
    __doc__ = _create_class_docstr(**mtdt['Uniform'])

    def __init__(self, a, b):
        self.a, self.b = a, b

    def __str__(self):
        return "Uniform(a=%.5f, b=%.5f)" %(self.params)

    def __repr__(self):
        return self.__str__()

    # ===================
    # Parameter retrieval
    # ===================

    @property
    def params(self):
        """Returns parameters."""
        return (self.a, self.b)

    @property
    def location(self):
        """Returns location parameter if exists."""
        return self.a

    @property
    def scale(self):
        """Returns scale parameter if exists."""
        return self.b - self.a

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
        return .5 * (self.a + self.b)

    @property
    def median(self):
        """Returns median."""
        return .5 * (self.a + self.b)

    @property
    def mode(self):
        """Returns mode."""
        return None

    @property
    def var(self):
        """Returns variance."""
        return (self.b - self.a)**2/12

    @property
    def std(self):
        """Returns standard deviation."""
        return (self.b - self.a)/np.sqrt(12)

    @property
    def skewness(self):
        """Returns skewness."""
        return 0

    @property
    def kurtosis(self):
        """Returns kurtosis."""
        return -1.2

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
        return np.log(self.b - self.a)

    def mgf(self, x):
        """Evaluate moment generating function at x."""
        return unif_mgf(self.a, self.b, x)

    def cf(self, x):
        """Evaluate characteristic function at x."""
        return unif_cf(self.a, self.b, x)

    # ==========
    # Evaluation
    # ==========

    def insupport(self, x):
        """When x is a scalar, it returns whether x is within
        the support of the distribution. When x is an array,
        it returns whether every element."""
        return a <= x < b

    def pdf(self, x):
        """The pdf value(s) evaluated at x."""
        return unif_pdf(self.a, self.b, x)
    
    def logpdf(self, x):
        """The logarithm of the pdf value(s) evaluated at x."""
        return unif_logpdf(self.a, self.b, x)

    def loglikelihood(self, x):
        """The log-likelihood of the distribution w.r.t. all
        samples contained in array x."""
        return sum(unif_logpdf(self.a, self.b, x))
    
    def cdf(self, x):
        """The cdf value(s) evaluated at x."""
        return unif_cdf(self.a, self.b, x)
    
    def ccdf(self, x):
        """The complementary cdf evaluated at x, i.e. 1 - cdf(x)."""
        return unif_ccdf(self.a, self.b, x)
    
    def logcdf(self, x):
        """The logarithm of the cdf value(s) evaluated at x."""
        return unif_logcdf(self.a, self.b, x)
    
    def logccdf(self, x):
        """The logarithm of the complementary cdf evaluated at x."""
        return unif_logccdf(self.a, self.b, x)
    
    def quantile(self, q):
        """The quantile value evaluated at q."""
        return unif_invcdf(self.a, self.b, q)
    
    def cquantile(self, q):
        """The complementary quantile value evaluated at q."""
        return unif_invccdf(self.a, self.b, q)
    
    def invlogcdf(self, lq):
        """The inverse function of logcdf."""
        return unif_invlogcdf(self.a, self.b, lq)
    
    def invlogccdf(self, lq):
        """The inverse function of logccdf."""
        return unif_invlogccdf(self.a, self.b, lq)
    
    # ========
    # Sampling
    # ========
    
    def rand(self, n):
        """Generates a random draw from the distribution."""
        out = np.empty(n)
        for i, _ in np.ndenumerate(out):
            out[i] = unif_rand(self.a, self.b)
        return out

    
# ============================= NEW DISTRIBUTION =================================
dt = _rmath_ffi.lib.dt
pt = _rmath_ffi.lib.pt
qt = _rmath_ffi.lib.qt

@vectorize(nopython=True)
def tdist_pdf(v, x):
    return dt(x, v, 0)


@vectorize(nopython=True)
def tdist_logpdf(v, x):
    return dt(x, v, 1)


@vectorize(nopython=True)
def tdist_cdf(v, x):
    return pt(x, v, 1, 0)


@vectorize(nopython=True)
def tdist_ccdf(v, x):
    return pt(x, v, 0, 0)


@vectorize(nopython=True)
def tdist_logcdf(v, x):
    return pt(x, v, 1, 1)


@vectorize(nopython=True)
def tdist_logccdf(v, x):
    return pt(x, v, 0, 1)


@vectorize(nopython=True)
def tdist_invcdf(v, q):
    return qt(q, v, 1, 0)


@vectorize(nopython=True)
def tdist_invccdf(v, q):
    return qt(q, v, 0, 0)


@vectorize(nopython=True)
def tdist_invlogcdf(v, lq):
    return qt(lq, v, 1, 1)


@vectorize(nopython=True)
def tdist_invlogccdf(v, lq):
    return qt(lq, v, 0, 1)

rt = _rmath_ffi.lib.rt

@jit(nopython=True)
def tdist_rand(v):
    return rt(v)


@vectorize(nopython=True)
def tdist_mgf(v, x):
    return None

@vectorize(nopython=True)
def tdist_cf(v, x):
    return None

#  ------
#  T
#  ------

spec = [
    ('v', int32)
]

@jitclass(spec)
class T():

    # set docstring
    __doc__ = _create_class_docstr(**mtdt['T'])

    def __init__(self, v):
        self.v = v

    def __str__(self):
        return "T(df=%.5f)" %(self.params)

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
        """Returns location parameter if exists."""
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
        return 0

    @property
    def median(self):
        """Returns median."""
        return 0

    @property
    def mode(self):
        """Returns mode."""
        return 0

    @property
    def var(self):
        """Returns variance."""
        return self.v/(self.v - 2) if self.v > 2 else inf

    @property
    def std(self):
        """Returns standard deviation."""
        return np.sqrt(self.v/(self.v - 2)) if self.v > 2 else inf

    @property
    def skewness(self):
        """Returns skewness."""
        return 0 if self.v > 3 else None

    @property
    def kurtosis(self):
        """Returns kurtosis."""
        return 6/(self.v - 4) if self.v > 4 else inf

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
        return .5*(self.v + 1)*(digamma(.5*(self.v + 1)) - digamma(.5*self.v)) + np.log(np.sqrt(self.v) * beta(.5*self.v, .5))

    def mgf(self, x):
        """Evaluate moment generating function at x."""
        return tdist_mgf(self.v, x)

    def cf(self, x):
        """Evaluate characteristic function at x."""
        return tdist_cf(self.v, x)

    # ==========
    # Evaluation
    # ==========

    def insupport(self, x):
        """When x is a scalar, it returns whether x is within
        the support of the distribution. When x is an array,
        it returns whether every element."""
        return -inf <= x < inf

    def pdf(self, x):
        """The pdf value(s) evaluated at x."""
        return tdist_pdf(self.v, x)
    
    def logpdf(self, x):
        """The logarithm of the pdf value(s) evaluated at x."""
        return tdist_logpdf(self.v, x)

    def loglikelihood(self, x):
        """The log-likelihood of the distribution w.r.t. all
        samples contained in array x."""
        return sum(tdist_logpdf(self.v, x))
    
    def cdf(self, x):
        """The cdf value(s) evaluated at x."""
        return tdist_cdf(self.v, x)
    
    def ccdf(self, x):
        """The complementary cdf evaluated at x, i.e. 1 - cdf(x)."""
        return tdist_ccdf(self.v, x)
    
    def logcdf(self, x):
        """The logarithm of the cdf value(s) evaluated at x."""
        return tdist_logcdf(self.v, x)
    
    def logccdf(self, x):
        """The logarithm of the complementary cdf evaluated at x."""
        return tdist_logccdf(self.v, x)
    
    def quantile(self, q):
        """The quantile value evaluated at q."""
        return tdist_invcdf(self.v, q)
    
    def cquantile(self, q):
        """The complementary quantile value evaluated at q."""
        return tdist_invccdf(self.v, q)
    
    def invlogcdf(self, lq):
        """The inverse function of logcdf."""
        return tdist_invlogcdf(self.v, lq)
    
    def invlogccdf(self, lq):
        """The inverse function of logccdf."""
        return tdist_invlogccdf(self.v, lq)
    
    # ========
    # Sampling
    # ========
    
    def rand(self, n):
        """Generates a random draw from the distribution."""
        out = np.empty(n)
        for i, _ in np.ndenumerate(out):
            out[i] = tdist_rand(self.v)
        return out

    
# ============================= NEW DISTRIBUTION =================================
dlnorm = _rmath_ffi.lib.dlnorm
plnorm = _rmath_ffi.lib.plnorm
qlnorm = _rmath_ffi.lib.qlnorm

@vectorize(nopython=True)
def lognormal_pdf(mu, sigma, x):
    return dlnorm(x, mu, sigma, 0)


@vectorize(nopython=True)
def lognormal_logpdf(mu, sigma, x):
    return dlnorm(x, mu, sigma, 1)


@vectorize(nopython=True)
def lognormal_cdf(mu, sigma, x):
    return plnorm(x, mu, sigma, 1, 0)


@vectorize(nopython=True)
def lognormal_ccdf(mu, sigma, x):
    return plnorm(x, mu, sigma, 0, 0)


@vectorize(nopython=True)
def lognormal_logcdf(mu, sigma, x):
    return plnorm(x, mu, sigma, 1, 1)


@vectorize(nopython=True)
def lognormal_logccdf(mu, sigma, x):
    return plnorm(x, mu, sigma, 0, 1)


@vectorize(nopython=True)
def lognormal_invcdf(mu, sigma, q):
    return qlnorm(q, mu, sigma, 1, 0)


@vectorize(nopython=True)
def lognormal_invccdf(mu, sigma, q):
    return qlnorm(q, mu, sigma, 0, 0)


@vectorize(nopython=True)
def lognormal_invlogcdf(mu, sigma, lq):
    return qlnorm(lq, mu, sigma, 1, 1)


@vectorize(nopython=True)
def lognormal_invlogccdf(mu, sigma, lq):
    return qlnorm(lq, mu, sigma, 0, 1)

rlnorm = _rmath_ffi.lib.rlnorm

@jit(nopython=True)
def lognormal_rand(mu, sigma):
    return rlnorm(mu, sigma)


@vectorize(nopython=True)
def lnorm_mgf(mu, sigma, x):
    return None

@vectorize(nopython=True)
def lnorm_cf(mu, sigma, x):
    return None

#  ------
#  LogNormal
#  ------

spec = [
    ('mu', float32), ('sigma', float32)
]

@jitclass(spec)
class LogNormal():

    # set docstring
    __doc__ = _create_class_docstr(**mtdt['LogNormal'])

    def __init__(self, mu, sigma):
        self.mu, self.sigma = mu, sigma

    def __str__(self):
        return "LogNormal(mu=%.5f, sigma=%.5f)" %(self.params)

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
        """Returns location parameter if exists."""
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
        return np.exp(self.mu + .5* self.sigma**2)

    @property
    def median(self):
        """Returns median."""
        return np.exp(self.mu)

    @property
    def mode(self):
        """Returns mode."""
        return np.exp(self.mu - self.sigma**2)

    @property
    def var(self):
        """Returns variance."""
        return (np.exp(self.sigma**2) - 1) * np.exp(2*self.mu + self.sigma**2)

    @property
    def std(self):
        """Returns standard deviation."""
        return np.sqrt(self.var)

    @property
    def skewness(self):
        """Returns skewness."""
        return (np.exp(self.sigma**2) + 2) * np.sqrt(np.exp(self.sigma**2) - 1)

    @property
    def kurtosis(self):
        """Returns kurtosis."""
        return np.exp(4*self.sigma**2) + 2*np.exp(3*self.sigma**2) + 3*np.exp(2*self.sigma**2) - 6

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
        return np.log(self.sigma*np.exp(self.mu + .5)*np.sqrt(2*np.pi))

    def mgf(self, x):
        """Evaluate moment generating function at x."""
        return lnorm_mgf(self.mu, self.sigma, x)

    def cf(self, x):
        """Evaluate characteristic function at x."""
        return lnorm_cf(self.mu, self.sigma, x)

    # ==========
    # Evaluation
    # ==========

    def insupport(self, x):
        """When x is a scalar, it returns whether x is within
        the support of the distribution. When x is an array,
        it returns whether every element."""
        return 0 < x < inf

    def pdf(self, x):
        """The pdf value(s) evaluated at x."""
        return lognormal_pdf(self.mu, self.sigma, x)
    
    def logpdf(self, x):
        """The logarithm of the pdf value(s) evaluated at x."""
        return lognormal_logpdf(self.mu, self.sigma, x)

    def loglikelihood(self, x):
        """The log-likelihood of the distribution w.r.t. all
        samples contained in array x."""
        return sum(lognormal_logpdf(self.mu, self.sigma, x))
    
    def cdf(self, x):
        """The cdf value(s) evaluated at x."""
        return lognormal_cdf(self.mu, self.sigma, x)
    
    def ccdf(self, x):
        """The complementary cdf evaluated at x, i.e. 1 - cdf(x)."""
        return lognormal_ccdf(self.mu, self.sigma, x)
    
    def logcdf(self, x):
        """The logarithm of the cdf value(s) evaluated at x."""
        return lognormal_logcdf(self.mu, self.sigma, x)
    
    def logccdf(self, x):
        """The logarithm of the complementary cdf evaluated at x."""
        return lognormal_logccdf(self.mu, self.sigma, x)
    
    def quantile(self, q):
        """The quantile value evaluated at q."""
        return lognormal_invcdf(self.mu, self.sigma, q)
    
    def cquantile(self, q):
        """The complementary quantile value evaluated at q."""
        return lognormal_invccdf(self.mu, self.sigma, q)
    
    def invlogcdf(self, lq):
        """The inverse function of logcdf."""
        return lognormal_invlogcdf(self.mu, self.sigma, lq)
    
    def invlogccdf(self, lq):
        """The inverse function of logccdf."""
        return lognormal_invlogccdf(self.mu, self.sigma, lq)
    
    # ========
    # Sampling
    # ========
    
    def rand(self, n):
        """Generates a random draw from the distribution."""
        out = np.empty(n)
        for i, _ in np.ndenumerate(out):
            out[i] = lognormal_rand(self.mu, self.sigma)
        return out

    
# ============================= NEW DISTRIBUTION =================================
df = _rmath_ffi.lib.df
pf = _rmath_ffi.lib.pf
qf = _rmath_ffi.lib.qf

@vectorize(nopython=True)
def fdist_pdf(v1, v2, x):
    return df(x, v1, v2, 0)


@vectorize(nopython=True)
def fdist_logpdf(v1, v2, x):
    return df(x, v1, v2, 1)


@vectorize(nopython=True)
def fdist_cdf(v1, v2, x):
    return pf(x, v1, v2, 1, 0)


@vectorize(nopython=True)
def fdist_ccdf(v1, v2, x):
    return pf(x, v1, v2, 0, 0)


@vectorize(nopython=True)
def fdist_logcdf(v1, v2, x):
    return pf(x, v1, v2, 1, 1)


@vectorize(nopython=True)
def fdist_logccdf(v1, v2, x):
    return pf(x, v1, v2, 0, 1)


@vectorize(nopython=True)
def fdist_invcdf(v1, v2, q):
    return qf(q, v1, v2, 1, 0)


@vectorize(nopython=True)
def fdist_invccdf(v1, v2, q):
    return qf(q, v1, v2, 0, 0)


@vectorize(nopython=True)
def fdist_invlogcdf(v1, v2, lq):
    return qf(lq, v1, v2, 1, 1)


@vectorize(nopython=True)
def fdist_invlogccdf(v1, v2, lq):
    return qf(lq, v1, v2, 0, 1)

rf = _rmath_ffi.lib.rf

@jit(nopython=True)
def fdist_rand(v1, v2):
    return rf(v1, v2)


@vectorize(nopython=True)
def fdist_mgf(v1, v2, x):
    return None

@vectorize(nopython=True)
def fdist_cf(v1, v2, x):
    return None

#  ------
#  F
#  ------

spec = [
    ('v1', float32), ('v2', float32)
]

@jitclass(spec)
class F():

    # set docstring
    __doc__ = _create_class_docstr(**mtdt['F'])

    def __init__(self, v1, v2):
        self.v1, self.v2 = v1, v2

    def __str__(self):
        return "F(d1=%.5f, d2=%.5f)" %(self.params)

    def __repr__(self):
        return self.__str__()

    # ===================
    # Parameter retrieval
    # ===================

    @property
    def params(self):
        """Returns parameters."""
        return (self.v1, self.v2)

    @property
    def location(self):
        """Returns location parameter if exists."""
        return None

    @property
    def scale(self):
        """Returns scale parameter if exists."""
        return None

    @property
    def shape(self):
        """Returns shape parameter if exists."""
        return (self.v1, self.v2)

    # ==========
    # Statistics
    # ==========

    @property
    def mean(self):
        """Returns mean."""
        return self.v2/(self.v2 - 2) if self.v2 > 2 else inf

    @property
    def median(self):
        """Returns median."""
        return None

    @property
    def mode(self):
        """Returns mode."""
        return (self.v1 - 2)/self.v1 * self.v2/(self.v2 + 2) if self.v1 > 2 else inf

    @property
    def var(self):
        """Returns variance."""
        return 2*self.v2**2*(self.v1 + self.v2 - 2)/ (self.v1*(self.v2 - 2)**2*(self.v2 - 4)) if self.v2 > 4 else inf

    @property
    def std(self):
        """Returns standard deviation."""
        return np.sqrt(self.var)

    @property
    def skewness(self):
        """Returns skewness."""
        return (2*self.v1 + self.v2 - 2)*np.sqrt(8*(self.v2 - 4))/ ((self.v2 - 6)*np.sqrt(self.v1*(self.v1+self.v2-2))) if self.v2 > 6 else inf

    @property
    def kurtosis(self):
        """Returns kurtosis."""
        return 3 + 12*(self.v1*(5*self.v2 - 22)*(self.v1+self.v2-2) + (self.v2 - 4)*(self.v2 - 2)**2)/ (self.v1*(self.v2-6)*(self.v2-8)*(self.v1+self.v2-2))

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
        return None

    def mgf(self, x):
        """Evaluate moment generating function at x."""
        return fdist_mgf(self.v1, self.v2, x)

    def cf(self, x):
        """Evaluate characteristic function at x."""
        return fdist_cf(self.v1, self.v2, x)

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
        return fdist_pdf(self.v1, self.v2, x)
    
    def logpdf(self, x):
        """The logarithm of the pdf value(s) evaluated at x."""
        return fdist_logpdf(self.v1, self.v2, x)

    def loglikelihood(self, x):
        """The log-likelihood of the distribution w.r.t. all
        samples contained in array x."""
        return sum(fdist_logpdf(self.v1, self.v2, x))
    
    def cdf(self, x):
        """The cdf value(s) evaluated at x."""
        return fdist_cdf(self.v1, self.v2, x)
    
    def ccdf(self, x):
        """The complementary cdf evaluated at x, i.e. 1 - cdf(x)."""
        return fdist_ccdf(self.v1, self.v2, x)
    
    def logcdf(self, x):
        """The logarithm of the cdf value(s) evaluated at x."""
        return fdist_logcdf(self.v1, self.v2, x)
    
    def logccdf(self, x):
        """The logarithm of the complementary cdf evaluated at x."""
        return fdist_logccdf(self.v1, self.v2, x)
    
    def quantile(self, q):
        """The quantile value evaluated at q."""
        return fdist_invcdf(self.v1, self.v2, q)
    
    def cquantile(self, q):
        """The complementary quantile value evaluated at q."""
        return fdist_invccdf(self.v1, self.v2, q)
    
    def invlogcdf(self, lq):
        """The inverse function of logcdf."""
        return fdist_invlogcdf(self.v1, self.v2, lq)
    
    def invlogccdf(self, lq):
        """The inverse function of logccdf."""
        return fdist_invlogccdf(self.v1, self.v2, lq)
    
    # ========
    # Sampling
    # ========
    
    def rand(self, n):
        """Generates a random draw from the distribution."""
        out = np.empty(n)
        for i, _ in np.ndenumerate(out):
            out[i] = fdist_rand(self.v1, self.v2)
        return out

    
# ============================= NEW DISTRIBUTION =================================
dgamma = _rmath_ffi.lib.dgamma
pgamma = _rmath_ffi.lib.pgamma
qgamma = _rmath_ffi.lib.qgamma

@vectorize(nopython=True)
def gamma_pdf(alpha, beta, x):
    return dgamma(x, alpha, beta, 0)


@vectorize(nopython=True)
def gamma_logpdf(alpha, beta, x):
    return dgamma(x, alpha, beta, 1)


@vectorize(nopython=True)
def gamma_cdf(alpha, beta, x):
    return pgamma(x, alpha, beta, 1, 0)


@vectorize(nopython=True)
def gamma_ccdf(alpha, beta, x):
    return pgamma(x, alpha, beta, 0, 0)


@vectorize(nopython=True)
def gamma_logcdf(alpha, beta, x):
    return pgamma(x, alpha, beta, 1, 1)


@vectorize(nopython=True)
def gamma_logccdf(alpha, beta, x):
    return pgamma(x, alpha, beta, 0, 1)


@vectorize(nopython=True)
def gamma_invcdf(alpha, beta, q):
    return qgamma(q, alpha, beta, 1, 0)


@vectorize(nopython=True)
def gamma_invccdf(alpha, beta, q):
    return qgamma(q, alpha, beta, 0, 0)


@vectorize(nopython=True)
def gamma_invlogcdf(alpha, beta, lq):
    return qgamma(lq, alpha, beta, 1, 1)


@vectorize(nopython=True)
def gamma_invlogccdf(alpha, beta, lq):
    return qgamma(lq, alpha, beta, 0, 1)

rgamma = _rmath_ffi.lib.rgamma

@jit(nopython=True)
def gamma_rand(alpha, beta):
    return rgamma(alpha, beta)


@vectorize(nopython=True)
def gamma_mgf(alpha, beta, x):
    return (1 - x/beta)**(-alpha) if x < beta else None

@vectorize(nopython=True)
def gamma_cf(alpha, beta, x):
    return (1 - (1j * x)/beta)**(-alpha)

#  ------
#  Gamma
#  ------

spec = [
    ('alpha', float32), ('beta', float32)
]

@jitclass(spec)
class Gamma():

    # set docstring
    __doc__ = _create_class_docstr(**mtdt['Gamma'])

    def __init__(self, alpha, beta):
        self.alpha, self.beta = alpha, beta

    def __str__(self):
        return "Gamma(alpha=%.5f, beta=%.5f)" %(self.params)

    def __repr__(self):
        return self.__str__()

    # ===================
    # Parameter retrieval
    # ===================

    @property
    def params(self):
        """Returns parameters."""
        return (self.alpha, self.beta)

    @property
    def location(self):
        """Returns location parameter if exists."""
        return None

    @property
    def scale(self):
        """Returns scale parameter if exists."""
        return 1/self.beta

    @property
    def shape(self):
        """Returns shape parameter if exists."""
        return self.alpha

    # ==========
    # Statistics
    # ==========

    @property
    def mean(self):
        """Returns mean."""
        return self.alpha/self.beta

    @property
    def median(self):
        """Returns median."""
        return None

    @property
    def mode(self):
        """Returns mode."""
        return (self.alpha - 1)/self.beta if self.alpha >= 1 else None

    @property
    def var(self):
        """Returns variance."""
        return self.alpha/(self.beta**2)

    @property
    def std(self):
        """Returns standard deviation."""
        return np.sqrt(self.var)

    @property
    def skewness(self):
        """Returns skewness."""
        return 2/(np.sqrt(self.apha))

    @property
    def kurtosis(self):
        """Returns kurtosis."""
        return 3 + 6/self.alpha

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
        return self.alpha - np.log(self.beta) + np.log(gamma(self.alpha)) + (1 - self.alpha)*digamma(self.alpha)

    def mgf(self, x):
        """Evaluate moment generating function at x."""
        return gamma_mgf(self.alpha, self.beta, x)

    def cf(self, x):
        """Evaluate characteristic function at x."""
        return gamma_cf(self.alpha, self.beta, x)

    # ==========
    # Evaluation
    # ==========

    def insupport(self, x):
        """When x is a scalar, it returns whether x is within
        the support of the distribution. When x is an array,
        it returns whether every element."""
        return 0 < x < inf

    def pdf(self, x):
        """The pdf value(s) evaluated at x."""
        return gamma_pdf(self.alpha, self.beta, x)
    
    def logpdf(self, x):
        """The logarithm of the pdf value(s) evaluated at x."""
        return gamma_logpdf(self.alpha, self.beta, x)

    def loglikelihood(self, x):
        """The log-likelihood of the distribution w.r.t. all
        samples contained in array x."""
        return sum(gamma_logpdf(self.alpha, self.beta, x))
    
    def cdf(self, x):
        """The cdf value(s) evaluated at x."""
        return gamma_cdf(self.alpha, self.beta, x)
    
    def ccdf(self, x):
        """The complementary cdf evaluated at x, i.e. 1 - cdf(x)."""
        return gamma_ccdf(self.alpha, self.beta, x)
    
    def logcdf(self, x):
        """The logarithm of the cdf value(s) evaluated at x."""
        return gamma_logcdf(self.alpha, self.beta, x)
    
    def logccdf(self, x):
        """The logarithm of the complementary cdf evaluated at x."""
        return gamma_logccdf(self.alpha, self.beta, x)
    
    def quantile(self, q):
        """The quantile value evaluated at q."""
        return gamma_invcdf(self.alpha, self.beta, q)
    
    def cquantile(self, q):
        """The complementary quantile value evaluated at q."""
        return gamma_invccdf(self.alpha, self.beta, q)
    
    def invlogcdf(self, lq):
        """The inverse function of logcdf."""
        return gamma_invlogcdf(self.alpha, self.beta, lq)
    
    def invlogccdf(self, lq):
        """The inverse function of logccdf."""
        return gamma_invlogccdf(self.alpha, self.beta, lq)
    
    # ========
    # Sampling
    # ========
    
    def rand(self, n):
        """Generates a random draw from the distribution."""
        out = np.empty(n)
        for i, _ in np.ndenumerate(out):
            out[i] = gamma_rand(self.alpha, self.beta)
        return out

    
# ============================= NEW DISTRIBUTION =================================
dbeta = _rmath_ffi.lib.dbeta
pbeta = _rmath_ffi.lib.pbeta
qbeta = _rmath_ffi.lib.qbeta

@vectorize(nopython=True)
def beta_pdf(alpha, beta, x):
    return dbeta(x, alpha, beta, 0)


@vectorize(nopython=True)
def beta_logpdf(alpha, beta, x):
    return dbeta(x, alpha, beta, 1)


@vectorize(nopython=True)
def beta_cdf(alpha, beta, x):
    return pbeta(x, alpha, beta, 1, 0)


@vectorize(nopython=True)
def beta_ccdf(alpha, beta, x):
    return pbeta(x, alpha, beta, 0, 0)


@vectorize(nopython=True)
def beta_logcdf(alpha, beta, x):
    return pbeta(x, alpha, beta, 1, 1)


@vectorize(nopython=True)
def beta_logccdf(alpha, beta, x):
    return pbeta(x, alpha, beta, 0, 1)


@vectorize(nopython=True)
def beta_invcdf(alpha, beta, q):
    return qbeta(q, alpha, beta, 1, 0)


@vectorize(nopython=True)
def beta_invccdf(alpha, beta, q):
    return qbeta(q, alpha, beta, 0, 0)


@vectorize(nopython=True)
def beta_invlogcdf(alpha, beta, lq):
    return qbeta(lq, alpha, beta, 1, 1)


@vectorize(nopython=True)
def beta_invlogccdf(alpha, beta, lq):
    return qbeta(lq, alpha, beta, 0, 1)

rbeta = _rmath_ffi.lib.rbeta

@jit(nopython=True)
def beta_rand(alpha, beta):
    return rbeta(alpha, beta)


@vectorize(nopython=True)
def beta_mgf(alpha, beta, x):
    return None

@vectorize(nopython=True)
def beta_cf(alpha, beta, x):
    return None

#  ------
#  Beta
#  ------

spec = [
    ('alpha', float32), ('beta', float32)
]

@jitclass(spec)
class Beta():

    # set docstring
    __doc__ = _create_class_docstr(**mtdt['Beta'])

    def __init__(self, alpha, beta):
        self.alpha, self.beta = alpha, beta

    def __str__(self):
        return "Beta(alpha=%.5f, beta=%.5f)" %(self.params)

    def __repr__(self):
        return self.__str__()

    # ===================
    # Parameter retrieval
    # ===================

    @property
    def params(self):
        """Returns parameters."""
        return (self.alpha, self.beta)

    @property
    def location(self):
        """Returns location parameter if exists."""
        return None

    @property
    def scale(self):
        """Returns scale parameter if exists."""
        return None

    @property
    def shape(self):
        """Returns shape parameter if exists."""
        return (self.alpha, self.beta)

    # ==========
    # Statistics
    # ==========

    @property
    def mean(self):
        """Returns mean."""
        return self.alpha/(self.alpha + self.beta)

    @property
    def median(self):
        """Returns median."""
        return (self.alpha - 1/3)/(self.alpha + self.beta - 2/3) if self.alpha >=1 and self.beta >= 1 else None

    @property
    def mode(self):
        """Returns mode."""
        return (self.alpha - 1)/(self.alpha + self.beta - 2) if self.alpha > 1 and self.beta > 1 else None

    @property
    def var(self):
        """Returns variance."""
        return (self.alpha * self.beta)/ ((self.alpha + self.beta)**2 * (self.alpha + self.beta + 1))

    @property
    def std(self):
        """Returns standard deviation."""
        return np.sqrt(self.var)

    @property
    def skewness(self):
        """Returns skewness."""
        return 2 * (self.beta - self.alpha) * np.sqrt(self.alpha + self.beta + 1)/ ((self.alpha + self.beta + 2) * np.sqrt(self.apha * self.beta))

    @property
    def kurtosis(self):
        """Returns kurtosis."""
        return 3 + 6 * ((self.alpha - self.beta)**2*(self.alpha + self.beta + 1) - self.alpha * self.beta * (self.alpha + self.beta + 2) )/ (self.alpha * self.beta * (self.alpha + self.beta + 2) * (self.alpha + self.beta + 3))

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
        return np.log(beta(self.alpha, self.beta)) - (self.alpha - 1)* digamma(self.alpha) - (self.beta - 1)*digamma(self.beta) + (self.alpha + self.beta - 2)*digamma(self.alpha + self.beta)

    def mgf(self, x):
        """Evaluate moment generating function at x."""
        return beta_mgf(self.alpha, self.beta, x)

    def cf(self, x):
        """Evaluate characteristic function at x."""
        return beta_cf(self.alpha, self.beta, x)

    # ==========
    # Evaluation
    # ==========

    def insupport(self, x):
        """When x is a scalar, it returns whether x is within
        the support of the distribution. When x is an array,
        it returns whether every element."""
        return 0 < x < 1

    def pdf(self, x):
        """The pdf value(s) evaluated at x."""
        return beta_pdf(self.alpha, self.beta, x)
    
    def logpdf(self, x):
        """The logarithm of the pdf value(s) evaluated at x."""
        return beta_logpdf(self.alpha, self.beta, x)

    def loglikelihood(self, x):
        """The log-likelihood of the distribution w.r.t. all
        samples contained in array x."""
        return sum(beta_logpdf(self.alpha, self.beta, x))
    
    def cdf(self, x):
        """The cdf value(s) evaluated at x."""
        return beta_cdf(self.alpha, self.beta, x)
    
    def ccdf(self, x):
        """The complementary cdf evaluated at x, i.e. 1 - cdf(x)."""
        return beta_ccdf(self.alpha, self.beta, x)
    
    def logcdf(self, x):
        """The logarithm of the cdf value(s) evaluated at x."""
        return beta_logcdf(self.alpha, self.beta, x)
    
    def logccdf(self, x):
        """The logarithm of the complementary cdf evaluated at x."""
        return beta_logccdf(self.alpha, self.beta, x)
    
    def quantile(self, q):
        """The quantile value evaluated at q."""
        return beta_invcdf(self.alpha, self.beta, q)
    
    def cquantile(self, q):
        """The complementary quantile value evaluated at q."""
        return beta_invccdf(self.alpha, self.beta, q)
    
    def invlogcdf(self, lq):
        """The inverse function of logcdf."""
        return beta_invlogcdf(self.alpha, self.beta, lq)
    
    def invlogccdf(self, lq):
        """The inverse function of logccdf."""
        return beta_invlogccdf(self.alpha, self.beta, lq)
    
    # ========
    # Sampling
    # ========
    
    def rand(self, n):
        """Generates a random draw from the distribution."""
        out = np.empty(n)
        for i, _ in np.ndenumerate(out):
            out[i] = beta_rand(self.alpha, self.beta)
        return out

    
# ============================= NEW DISTRIBUTION =================================
dexp = _rmath_ffi.lib.dexp
pexp = _rmath_ffi.lib.pexp
qexp = _rmath_ffi.lib.qexp

@vectorize(nopython=True)
def exp_pdf(theta, x):
    return dexp(x, theta, 0)


@vectorize(nopython=True)
def exp_logpdf(theta, x):
    return dexp(x, theta, 1)


@vectorize(nopython=True)
def exp_cdf(theta, x):
    return pexp(x, theta, 1, 0)


@vectorize(nopython=True)
def exp_ccdf(theta, x):
    return pexp(x, theta, 0, 0)


@vectorize(nopython=True)
def exp_logcdf(theta, x):
    return pexp(x, theta, 1, 1)


@vectorize(nopython=True)
def exp_logccdf(theta, x):
    return pexp(x, theta, 0, 1)


@vectorize(nopython=True)
def exp_invcdf(theta, q):
    return qexp(q, theta, 1, 0)


@vectorize(nopython=True)
def exp_invccdf(theta, q):
    return qexp(q, theta, 0, 0)


@vectorize(nopython=True)
def exp_invlogcdf(theta, lq):
    return qexp(lq, theta, 1, 1)


@vectorize(nopython=True)
def exp_invlogccdf(theta, lq):
    return qexp(lq, theta, 0, 1)

rexp = _rmath_ffi.lib.rexp

@jit(nopython=True)
def exp_rand(theta):
    return rexp(theta)


@vectorize(nopython=True)
def exp_mgf(theta, x):
    return theta/(theta - x) if x < theta else None

@vectorize(nopython=True)
def exp_cf(theta, x):
    return theta/(theta - 1j*x)

#  ------
#  Exponential
#  ------

spec = [
    ('theta', float32)
]

@jitclass(spec)
class Exponential():

    # set docstring
    __doc__ = _create_class_docstr(**mtdt['Exponential'])

    def __init__(self, theta):
        self.theta = theta

    def __str__(self):
        return "Exponential(theta=%.5f)" %(self.params)

    def __repr__(self):
        return self.__str__()

    # ===================
    # Parameter retrieval
    # ===================

    @property
    def params(self):
        """Returns parameters."""
        return (self.theta)

    @property
    def location(self):
        """Returns location parameter if exists."""
        return None

    @property
    def scale(self):
        """Returns scale parameter if exists."""
        return 1/self.theta

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
        return 1/self.theta

    @property
    def median(self):
        """Returns median."""
        return 1/self.theta * np.log(2)

    @property
    def mode(self):
        """Returns mode."""
        return 0

    @property
    def var(self):
        """Returns variance."""
        return self.theta**(-2)

    @property
    def std(self):
        """Returns standard deviation."""
        return np.sqrt(self.var)

    @property
    def skewness(self):
        """Returns skewness."""
        return 2

    @property
    def kurtosis(self):
        """Returns kurtosis."""
        return 9

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
        return 1 - np.log(self.theta)

    def mgf(self, x):
        """Evaluate moment generating function at x."""
        return exp_mgf(self.theta, x)

    def cf(self, x):
        """Evaluate characteristic function at x."""
        return exp_cf(self.theta, x)

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
        return exp_pdf(self.theta, x)
    
    def logpdf(self, x):
        """The logarithm of the pdf value(s) evaluated at x."""
        return exp_logpdf(self.theta, x)

    def loglikelihood(self, x):
        """The log-likelihood of the distribution w.r.t. all
        samples contained in array x."""
        return sum(exp_logpdf(self.theta, x))
    
    def cdf(self, x):
        """The cdf value(s) evaluated at x."""
        return exp_cdf(self.theta, x)
    
    def ccdf(self, x):
        """The complementary cdf evaluated at x, i.e. 1 - cdf(x)."""
        return exp_ccdf(self.theta, x)
    
    def logcdf(self, x):
        """The logarithm of the cdf value(s) evaluated at x."""
        return exp_logcdf(self.theta, x)
    
    def logccdf(self, x):
        """The logarithm of the complementary cdf evaluated at x."""
        return exp_logccdf(self.theta, x)
    
    def quantile(self, q):
        """The quantile value evaluated at q."""
        return exp_invcdf(self.theta, q)
    
    def cquantile(self, q):
        """The complementary quantile value evaluated at q."""
        return exp_invccdf(self.theta, q)
    
    def invlogcdf(self, lq):
        """The inverse function of logcdf."""
        return exp_invlogcdf(self.theta, lq)
    
    def invlogccdf(self, lq):
        """The inverse function of logccdf."""
        return exp_invlogccdf(self.theta, lq)
    
    # ========
    # Sampling
    # ========
    
    def rand(self, n):
        """Generates a random draw from the distribution."""
        out = np.empty(n)
        for i, _ in np.ndenumerate(out):
            out[i] = exp_rand(self.theta)
        return out

    
# ============================= NEW DISTRIBUTION =================================
dcauchy = _rmath_ffi.lib.dcauchy
pcauchy = _rmath_ffi.lib.pcauchy
qcauchy = _rmath_ffi.lib.qcauchy

@vectorize(nopython=True)
def cauchy_pdf(mu, sigma, x):
    return dcauchy(x, mu, sigma, 0)


@vectorize(nopython=True)
def cauchy_logpdf(mu, sigma, x):
    return dcauchy(x, mu, sigma, 1)


@vectorize(nopython=True)
def cauchy_cdf(mu, sigma, x):
    return pcauchy(x, mu, sigma, 1, 0)


@vectorize(nopython=True)
def cauchy_ccdf(mu, sigma, x):
    return pcauchy(x, mu, sigma, 0, 0)


@vectorize(nopython=True)
def cauchy_logcdf(mu, sigma, x):
    return pcauchy(x, mu, sigma, 1, 1)


@vectorize(nopython=True)
def cauchy_logccdf(mu, sigma, x):
    return pcauchy(x, mu, sigma, 0, 1)


@vectorize(nopython=True)
def cauchy_invcdf(mu, sigma, q):
    return qcauchy(q, mu, sigma, 1, 0)


@vectorize(nopython=True)
def cauchy_invccdf(mu, sigma, q):
    return qcauchy(q, mu, sigma, 0, 0)


@vectorize(nopython=True)
def cauchy_invlogcdf(mu, sigma, lq):
    return qcauchy(lq, mu, sigma, 1, 1)


@vectorize(nopython=True)
def cauchy_invlogccdf(mu, sigma, lq):
    return qcauchy(lq, mu, sigma, 0, 1)

rcauchy = _rmath_ffi.lib.rcauchy

@jit(nopython=True)
def cauchy_rand(mu, sigma):
    return rcauchy(mu, sigma)


@vectorize(nopython=True)
def cauchy_mgf(mu, sigma, x):
    return None

@vectorize(nopython=True)
def cauchy_cf(mu, sigma, x):
    return np.exp(mu*1j*x - sigma*np.abs(x))

#  ------
#  Cauchy
#  ------

spec = [
    ('mu', float32), ('sigma', float32)
]

@jitclass(spec)
class Cauchy():

    # set docstring
    __doc__ = _create_class_docstr(**mtdt['Cauchy'])

    def __init__(self, mu, sigma):
        self.mu, self.sigma = mu, sigma

    def __str__(self):
        return "Cauchy(mu=%.5f, sigma=%.5f)" %(self.params)

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
        """Returns location parameter if exists."""
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
        return None

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
        return None

    @property
    def std(self):
        """Returns standard deviation."""
        return None

    @property
    def skewness(self):
        """Returns skewness."""
        return None

    @property
    def kurtosis(self):
        """Returns kurtosis."""
        return None

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
        return np.log(self.sigma) + np.log(4*np.pi)

    def mgf(self, x):
        """Evaluate moment generating function at x."""
        return cauchy_mgf(self.mu, self.sigma, x)

    def cf(self, x):
        """Evaluate characteristic function at x."""
        return cauchy_cf(self.mu, self.sigma, x)

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
        return cauchy_pdf(self.mu, self.sigma, x)
    
    def logpdf(self, x):
        """The logarithm of the pdf value(s) evaluated at x."""
        return cauchy_logpdf(self.mu, self.sigma, x)

    def loglikelihood(self, x):
        """The log-likelihood of the distribution w.r.t. all
        samples contained in array x."""
        return sum(cauchy_logpdf(self.mu, self.sigma, x))
    
    def cdf(self, x):
        """The cdf value(s) evaluated at x."""
        return cauchy_cdf(self.mu, self.sigma, x)
    
    def ccdf(self, x):
        """The complementary cdf evaluated at x, i.e. 1 - cdf(x)."""
        return cauchy_ccdf(self.mu, self.sigma, x)
    
    def logcdf(self, x):
        """The logarithm of the cdf value(s) evaluated at x."""
        return cauchy_logcdf(self.mu, self.sigma, x)
    
    def logccdf(self, x):
        """The logarithm of the complementary cdf evaluated at x."""
        return cauchy_logccdf(self.mu, self.sigma, x)
    
    def quantile(self, q):
        """The quantile value evaluated at q."""
        return cauchy_invcdf(self.mu, self.sigma, q)
    
    def cquantile(self, q):
        """The complementary quantile value evaluated at q."""
        return cauchy_invccdf(self.mu, self.sigma, q)
    
    def invlogcdf(self, lq):
        """The inverse function of logcdf."""
        return cauchy_invlogcdf(self.mu, self.sigma, lq)
    
    def invlogccdf(self, lq):
        """The inverse function of logccdf."""
        return cauchy_invlogccdf(self.mu, self.sigma, lq)
    
    # ========
    # Sampling
    # ========
    
    def rand(self, n):
        """Generates a random draw from the distribution."""
        out = np.empty(n)
        for i, _ in np.ndenumerate(out):
            out[i] = cauchy_rand(self.mu, self.sigma)
        return out

    
# ============================= NEW DISTRIBUTION =================================
dpois = _rmath_ffi.lib.dpois
ppois = _rmath_ffi.lib.ppois
qpois = _rmath_ffi.lib.qpois

@vectorize(nopython=True)
def pois_pdf(mu, x):
    return dpois(x, mu, 0)


@vectorize(nopython=True)
def pois_logpdf(mu, x):
    return dpois(x, mu, 1)


@vectorize(nopython=True)
def pois_cdf(mu, x):
    return ppois(x, mu, 1, 0)


@vectorize(nopython=True)
def pois_ccdf(mu, x):
    return ppois(x, mu, 0, 0)


@vectorize(nopython=True)
def pois_logcdf(mu, x):
    return ppois(x, mu, 1, 1)


@vectorize(nopython=True)
def pois_logccdf(mu, x):
    return ppois(x, mu, 0, 1)


@vectorize(nopython=True)
def pois_invcdf(mu, q):
    return qpois(q, mu, 1, 0)


@vectorize(nopython=True)
def pois_invccdf(mu, q):
    return qpois(q, mu, 0, 0)


@vectorize(nopython=True)
def pois_invlogcdf(mu, lq):
    return qpois(lq, mu, 1, 1)


@vectorize(nopython=True)
def pois_invlogccdf(mu, lq):
    return qpois(lq, mu, 0, 1)

rpois = _rmath_ffi.lib.rpois

@jit(nopython=True)
def pois_rand(mu):
    return rpois(mu)


@vectorize(nopython=True)
def pois_mgf(mu, x):
    return np.exp(mua(np.exp(x) - 1))

@vectorize(nopython=True)
def pois_cf(mu, x):
    return np.exp(mua(np.exp(1j*x) - 1))

#  ------
#  Poisson
#  ------

spec = [
    ('mu', float32)
]

@jitclass(spec)
class Poisson():

    # set docstring
    __doc__ = _create_class_docstr(**mtdt['Poisson'])

    def __init__(self, mu):
        self.mu = mu

    def __str__(self):
        return "Poisson(mu=%.5f)" %(self.params)

    def __repr__(self):
        return self.__str__()

    # ===================
    # Parameter retrieval
    # ===================

    @property
    def params(self):
        """Returns parameters."""
        return (self.mu)

    @property
    def location(self):
        """Returns location parameter if exists."""
        return None

    @property
    def scale(self):
        """Returns scale parameter if exists."""
        return None

    @property
    def shape(self):
        """Returns shape parameter if exists."""
        return self.mu

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
        return floor(self.mu + 1/3 - 0.02/self.mu)

    @property
    def mode(self):
        """Returns mode."""
        return (ceil(self.mu) - 1, floor(self.mu))

    @property
    def var(self):
        """Returns variance."""
        return self.mu

    @property
    def std(self):
        """Returns standard deviation."""
        return np.sqrt(self.var)

    @property
    def skewness(self):
        """Returns skewness."""
        return self.mu**(.5)

    @property
    def kurtosis(self):
        """Returns kurtosis."""
        return 1/self.mu

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
        return .5*np.log(2*np.pi*np.e*self.mu) - 1/(12*self.mu) - 1/(24*self.mu**2) - 19/(360*self.mu**3)

    def mgf(self, x):
        """Evaluate moment generating function at x."""
        return pois_mgf(self.mu, x)

    def cf(self, x):
        """Evaluate characteristic function at x."""
        return pois_cf(self.mu, x)

    # ==========
    # Evaluation
    # ==========

    def insupport(self, x):
        """When x is a scalar, it returns whether x is within
        the support of the distribution. When x is an array,
        it returns whether every element."""
        return isinstance(x, int)

    def pdf(self, x):
        """The pdf value(s) evaluated at x."""
        return pois_pdf(self.mu, x)
    
    def logpdf(self, x):
        """The logarithm of the pdf value(s) evaluated at x."""
        return pois_logpdf(self.mu, x)

    def loglikelihood(self, x):
        """The log-likelihood of the distribution w.r.t. all
        samples contained in array x."""
        return sum(pois_logpdf(self.mu, x))
    
    def cdf(self, x):
        """The cdf value(s) evaluated at x."""
        return pois_cdf(self.mu, x)
    
    def ccdf(self, x):
        """The complementary cdf evaluated at x, i.e. 1 - cdf(x)."""
        return pois_ccdf(self.mu, x)
    
    def logcdf(self, x):
        """The logarithm of the cdf value(s) evaluated at x."""
        return pois_logcdf(self.mu, x)
    
    def logccdf(self, x):
        """The logarithm of the complementary cdf evaluated at x."""
        return pois_logccdf(self.mu, x)
    
    def quantile(self, q):
        """The quantile value evaluated at q."""
        return pois_invcdf(self.mu, q)
    
    def cquantile(self, q):
        """The complementary quantile value evaluated at q."""
        return pois_invccdf(self.mu, q)
    
    def invlogcdf(self, lq):
        """The inverse function of logcdf."""
        return pois_invlogcdf(self.mu, lq)
    
    def invlogccdf(self, lq):
        """The inverse function of logccdf."""
        return pois_invlogccdf(self.mu, lq)
    
    # ========
    # Sampling
    # ========
    
    def rand(self, n):
        """Generates a random draw from the distribution."""
        out = np.empty(n)
        for i, _ in np.ndenumerate(out):
            out[i] = pois_rand(self.mu)
        return out

    