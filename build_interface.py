import glob
import os
import textwrap

import yaml

# =========================================
# write out file for special functions
# =========================================


def _initiate_specials():
    '''
    Initiate python file for special functions which are present in
    the Rmath.h file  -- used mainly for characteristic functions
    '''

    pre_code = """\
    \"""
    Special functions for distributions.

    @authors :  Daniel Csaba <daniel.csaba@nyu.edu>
                Spencer Lyon <spencer.lyon@stern.nyu.edu>
    @date : 2016-07-26
    \"""

    from . import _rmath_ffi
    from numba import vectorize, jit
    from numba import cffi_support

    cffi_support.register_module(_rmath_ffi)

    # -------
    # gamma
    # -------

    gammafn = _rmath_ffi.lib.gammafn

    @vectorize(nopython=True)
    def gamma(x):
        return gammafn(x)

    # -------
    # lgamma
    # -------

    lgammafn = _rmath_ffi.lib.lgammafn

    @vectorize(nopython=True)
    def lgamma(x):
        return lgammafn(x)

    # -------
    # digamma
    # -------

    digammafn = _rmath_ffi.lib.digamma

    @vectorize(nopython=True)
    def digamma(x):
        return digammafn(x)

    # -------
    # beta
    # -------

    betafn = _rmath_ffi.lib.beta

    @vectorize(nopython=True)
    def beta(x, y):
        return betafn(x, y)

    # -------
    # bessel_k
    # Modified Bessel function of the second kind
    # -------

    bessel_k_fn = _rmath_ffi.lib.bessel_k

    @vectorize(nopython=True)
    def bessel_k(nu, x):
        return bessel_k_fn(x, nu, 1)

    # ----------
    # set seed
    # ----------

    set_seed_rmath = _rmath_ffi.lib.set_seed

    def set_seed(x, y):
        return set_seed_rmath(x, y)
    """
    with open(os.path.join("rvlib", "specials.py"), "w") as f:
        f.write(textwrap.dedent(pre_code))


# =========================================
# write out preamble for univariate classes
# =========================================


def _initiate_univariate():
    '''
    Initiate python file which collects all the  
    classes of different univariate distributions.
    '''

    # Define code which appears irrespective of  the
    # specific class of distribution
    pre_code = """\
    \"""
    Univariate distributions.

    @authors :  Daniel Csaba <daniel.csaba@nyu.edu>
                Spencer Lyon <spencer.lyon@stern.nyu.edu>
    @date : 2016-07-26
    \"""

    from os.path import join, dirname, abspath
    from numba import vectorize, jit, jitclass
    from numba import int32, float32

    import numpy as np
    from .specials import gamma, lgamma, digamma, beta, bessel_k, set_seed

    from . import _rmath_ffi
    from numba import cffi_support

    cffi_support.register_module(_rmath_ffi)

    # shut down divide by zero warnings for now
    import warnings
    warnings.filterwarnings("ignore")

    import yaml
    fn = join(dirname(abspath(__file__)), "metadata.yaml")
    with open(fn, 'r') as ymlfile:
        mtdt = yaml.load(ymlfile)

    # --------------------------------------------------
    # docstring following Spencer Lyon's distcan package
    # https://github.com/spencerlyon2/distcan.git
    # --------------------------------------------------

    univariate_class_docstr = r\"""
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
    \"""

    param_str = "{name_doc} : {kind}\\n    {descr}"


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

        return str.join("\\n", params)


    def _create_class_docstr(name_doc, param_names, param_descrs,
                             param_kinds="scalar(float)",
                             pdf_tex=r"\\text{not given}", **kwargs):
        param_list = _create_param_list_str(param_names, param_descrs,
                                            param_kinds)

        param_attributes = str.join(", ", param_names) + " : See Parameters"

        return univariate_class_docstr.format(**locals())
    """
    with open(os.path.join("rvlib", "univariate.py"), "w") as f:
        f.write(textwrap.dedent(pre_code))


# globals called from the textwrapper 
# with distribution specific content

def _import_rmath(rname, pyname, *pargs):
    """
    # now we map from the _rmath_ffi.lib.Xrname functions
    # to the friendly names from the julia file here:
    # https://github.com/JuliaStats/StatsFuns.jl/blob/master/src/rmath.jl
    """
    # extract Rmath.h function names
    dfun = "d{}".format(rname)
    pfun = "p{}".format(rname)
    qfun = "q{}".format(rname)
    rfun = "r{}".format(rname)

    # construct python function names
    pdf = "{}_pdf".format(pyname)
    cdf = "{}_cdf".format(pyname)
    ccdf = "{}_ccdf".format(pyname)

    logpdf = "{}_logpdf".format(pyname)
    logcdf = "{}_logcdf".format(pyname)
    logccdf = "{}_logccdf".format(pyname)

    invcdf = "{}_invcdf".format(pyname)
    invccdf = "{}_invccdf".format(pyname)
    invlogcdf = "{}_invlogcdf".format(pyname)
    invlogccdf = "{}_invlogccdf".format(pyname)

    rand = "{}_rand".format(pyname)

    # make sure all names are available
    has_rand = True
    if rname == "nbeta" or rname == "nf" or rname == "nt":
        has_rand = False

    p_args = ", ".join(pargs)

    code = """\

    # ============================= NEW DISTRIBUTION =================================
    {dfun} = _rmath_ffi.lib.{dfun}
    {pfun} = _rmath_ffi.lib.{pfun}
    {qfun} = _rmath_ffi.lib.{qfun}

    @vectorize(nopython=True)
    def {pdf}({p_args}, x):
        return {dfun}(x, {p_args}, 0)


    @vectorize(nopython=True)
    def {logpdf}({p_args}, x):
        return {dfun}(x, {p_args}, 1)


    @vectorize(nopython=True)
    def {cdf}({p_args}, x):
        return {pfun}(x, {p_args}, 1, 0)


    @vectorize(nopython=True)
    def {ccdf}({p_args}, x):
        return {pfun}(x, {p_args}, 0, 0)


    @vectorize(nopython=True)
    def {logcdf}({p_args}, x):
        return {pfun}(x, {p_args}, 1, 1)


    @vectorize(nopython=True)
    def {logccdf}({p_args}, x):
        return {pfun}(x, {p_args}, 0, 1)


    @vectorize(nopython=True)
    def {invcdf}({p_args}, q):
        return {qfun}(q, {p_args}, 1, 0)


    @vectorize(nopython=True)
    def {invccdf}({p_args}, q):
        return {qfun}(q, {p_args}, 0, 0)


    @vectorize(nopython=True)
    def {invlogcdf}({p_args}, lq):
        return {qfun}(lq, {p_args}, 1, 1)


    @vectorize(nopython=True)
    def {invlogccdf}({p_args}, lq):
        return {qfun}(lq, {p_args}, 0, 1)

    """.format(**locals())

    # append code for class to main file
    with open(os.path.join("rvlib", "univariate.py"), "a") as f:
        f.write(textwrap.dedent(code))

    if not has_rand:
        # end here if we don't have rand. I put it in a `not has_rand` block
        # to the rand_code can be at the right indentation level below
        return

    rand_code = """\
    {rfun} = _rmath_ffi.lib.{rfun}

    @jit(nopython=True)
    def {rand}({p_args}):
        return {rfun}({p_args})

        """.format(**locals())
    with open(os.path.join("rvlib", "univariate.py"), "a") as f:
        f.write(textwrap.dedent(rand_code))


# function to write out the dist specific part
def _write_class_specific(metadata, *pargs):

    # write out the distribution specific part
    # of the class which is not related to the
    # imported rmath functions

    # use _metadata_DIST and some derived locals

    p_args = ", ".join(pargs)

    p_args_self = ", ".join(["".join(("self.", par)) for par in pargs])

    data = locals()
    data.update(metadata)

    class_specific = """\

    @vectorize(nopython=True)
    def {pyname}_mgf({p_args}, x):
        return {mgf}

    @vectorize(nopython=True)
    def {pyname}_cf({p_args}, x):
        return {cf}

    #  ------
    #  {name}
    #  ------

    spec = [
        {spec}
    ]

    @jitclass(spec)
    class {name}():

        # set docstring
        __doc__ = _create_class_docstr(**mtdt['{name}'])

        def __init__(self, {p_args}):
            {p_args_self} = {p_args}

        def __str__(self):
            return "{string}" %(self.params)

        def __repr__(self):
            return self.__str__()

        # ===================
        # Parameter retrieval
        # ===================

        @property
        def params(self):
            \"""Return a tuple of parameters.\"""
            return ({p_args_self})

        @property
        def location(self):
            \"""Return location parameter if exists.\"""
            return {loc}

        @property
        def scale(self):
            \"""Return scale parameter if exists.\"""
            return {scale}

        @property
        def shape(self):
            \"""Return shape parameter if exists.\"""
            return {shape}

        # ==========
        # Statistics
        # ==========

        @property
        def mean(self):
            \"""Return the mean.\"""
            return {mean}

        @property
        def median(self):
            \"""Return the median.\"""
            return {median}

        @property
        def mode(self):
            \"""Return the mode.\"""
            return {mode}

        @property
        def var(self):
            \"""Return the variance.\"""
            return {var}

        @property
        def std(self):
            \"""Return the standard deviation.\"""
            return {std}

        @property
        def skewness(self):
            \"""Return the skewness.\"""
            return {skewness}

        @property
        def kurtosis(self):
            \"""Return the kurtosis.\"""
            return {kurtosis}

        @property
        def isplatykurtic(self):
            \"""Kurtosis being greater than zero.\"""
            return self.kurtosis > 0

        @property
        def isleptokurtic(self):
            \"""Kurtosis being smaller than zero.\"""
            return self.kurtosis < 0

        @property
        def ismesokurtic(self):
            \"""Kurtosis being equal to zero.\"""
            return self.kurtosis == 0.0

        @property
        def entropy(self):
            \"""Return the entropy.\"""
            return {entropy}

        def mgf(self, x):
            \"""Evaluate moment generating function at x.\"""
            return {pyname}_mgf({p_args_self}, x)

        def cf(self, x):
            \"""Evaluate characteristic function at x.\"""
            return {pyname}_cf({p_args_self}, x)

        # ==========
        # Evaluation
        # ==========

        def insupport(self, x):
            \"""When x is a scalar, it returns whether x is within
            the support of the distribution. When x is an array,
            it returns whether every element of x is within
            the support of the distribution.\"""
            return {insupport}
        """.format(**data)

    # append code for class to main file
    with open(os.path.join("rvlib", "univariate.py"), "a") as f:
        f.write(textwrap.dedent(class_specific))


def _write_class_rmath(rname, pyname, *pargs):
    """
    call top level @vectorized evaluation methods
    """

    # construct distribution specific function names
    pdf = "{}_pdf".format(pyname)
    cdf = "{}_cdf".format(pyname)
    ccdf = "{}_ccdf".format(pyname)

    logpdf = "{}_logpdf".format(pyname)
    logcdf = "{}_logcdf".format(pyname)
    logccdf = "{}_logccdf".format(pyname)

    invcdf = "{}_invcdf".format(pyname)
    invccdf = "{}_invccdf".format(pyname)
    invlogcdf = "{}_invlogcdf".format(pyname)
    invlogccdf = "{}_invlogccdf".format(pyname)

    rand = "{}_rand".format(pyname)

    # make sure all names are available
    has_rand = True
    if rname == "nbeta" or rname == "nf" or rname == "nt":
        has_rand = False

    # append 'self.' at the beginning of each parameter
    p_args = ", ".join(["".join(("self.", par)) for par in pargs])

    loc_code = """\

    def pdf(self, x):
        \"""The pdf value(s) evaluated at x.\"""
        return {pdf}({p_args}, x)

    def logpdf(self, x):
        \"""The logarithm of the pdf value(s) evaluated at x.\"""
        return {logpdf}({p_args}, x)

    def loglikelihood(self, x):
        \"""The log-likelihood of the distribution w.r.t. all
        samples contained in array x.\"""
        return sum({logpdf}({p_args}, x))

    def cdf(self, x):
        \"""The cdf value(s) evaluated at x.\"""
        return {cdf}({p_args}, x)

    def ccdf(self, x):
        \"""The complementary cdf evaluated at x, i.e. 1 - cdf(x).\"""
        return {ccdf}({p_args}, x)

    def logcdf(self, x):
        \"""The logarithm of the cdf value(s) evaluated at x.\"""
        return {logcdf}({p_args}, x)

    def logccdf(self, x):
        \"""The logarithm of the complementary cdf evaluated at x.\"""
        return {logccdf}({p_args}, x)

    def quantile(self, q):
        \"""The quantile value evaluated at q.\"""
        return {invcdf}({p_args}, q)

    def cquantile(self, q):
        \"""The complementary quantile value evaluated at q.\"""
        return {invccdf}({p_args}, q)

    def invlogcdf(self, lq):
        \"""The inverse function of logcdf.\"""
        return {invlogcdf}({p_args}, lq)

    def invlogccdf(self, lq):
        \"""The inverse function of logccdf.\"""
        return {invlogccdf}({p_args}, lq)
    """.format(**locals())

    # append code for class to main file
    with open(os.path.join("rvlib", "univariate.py"), "a") as f:
        f.write(loc_code)

    if not has_rand:
        # end here if we don't have rand. I put it in a `not has_rand` block
        # to the rand_code can be at the right indentation level below
        return

    rand_code = """\

    # ========
    # Sampling
    # ========

    def rand(self, n):
        \"""Generates a vector of n independent samples from the distribution.\"""
        out = np.empty(n)
        for i, _ in np.ndenumerate(out):
            out[i] = {rand}({p_args})
        return out

    """.format(**locals())
    with open(os.path.join("rvlib", "univariate.py"), "a") as f:
        f.write(rand_code)



# ==================================
# write out multivariate normal dist
# ==================================

def _write_multivariate():
    '''
    Initiate python file for special functions which are present in
    the Rmath.h file
    '''

    # Define code for all special functions -- used mainly 
    # for characteristic functions
    pre_code = """\
\"""
Multivariate distributions

@authors :  Daniel Csaba <daniel.csaba@nyu.edu>
            Spencer Lyon <spencer.lyon@stern.nyu.edu>
@date : 2016-07-26
\"""

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


@jit(nopython=True)
def mvnormal_pdf(mu, sigma, dim, x):
    return 1/(np.sqrt((2*np.pi)**dim*np.linalg.det(sigma)))*np.exp(-.5*(x - mu)@np.linalg.inv(sigma)@(x - mu))


rnorm = _rmath_ffi.lib.rnorm

# random number generation from independent standard normals

@jit(nopython=True)
def mvnormal_rand(dim):
    out = np.empty(dim)
    for i, _ in np.ndenumerate(out):
        out[i] = rnorm(0, 1)
    return out

spec = [
        ('mu', float64[:]),         # array field for mean 
        ('sigma', float64[:,:]),    # array field for covariance matrix
        ('dim', int64)              # scalar for dimension  
    ]

@jitclass(spec)
class MvNormal_c(object):

    def __init__(self, mu, sigma):
        self.mu = mu #np.array(mu, dtype=np.float64)
        self.sigma = sigma #np.array(sigma, dtype=np.float64)
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
        \"""Return the parameters.\"""
        return (self.mu, self.sigma)

    @property
    def length(self):
        \"""Return the length of random vector.\"""
        return self.dim

    # ==========
    # Statistics
    # ==========

    @property
    def mean(self):
        \"""Return the mean vector.\"""
        return self.mu

    @property
    def cov(self):
        \"""Return the covariance matrix.\"""
        return self.sigma

    @property
    def var(self):
        \"""Return the vector of component-wise variances.\"""
        return np.diag(self.sigma)
    
    @property
    def corr(self):
        \"""Return the correlation matrix.\"""  
        return np.diag(self.var**(-.5)) @ self.cov @ np.diag(self.var**(-.5))


    @property
    def entropy(self):
        \"""Return the entropy.\"""
        return .5*(self.length*(1 + np.log(2*np.pi)) + np.log(np.linalg.det(self.sigma)))


    # ==========
    # Evaluation
    # ==========

    def insupport(self, x):
        \"""Return whether 'x' is within the support of the distribution. 'x' has to be
        the same dimension as 'mu' and has to have 'dtype=np.float64'.\"""
        # can't raise valueerror in numba
        #if x.shape[0] != self.dim:
        #   raise ValueError("Array 'x' must be of dimension (%d)" % self.dim)
        return (-np.inf < x).all() and (x < np.inf).all() and x.shape[0] <= self.dim


    def pdf(self, x):
        \"""Return the probabilty density evaluated at 'x'. 'x' has to be
        the same dimension as 'mu' and has to have 'dtype=np.float64'.\"""

        # pdf only exists if 'sigma' is positive definite
        # if not np.all(np.linalg.eigvals(self.sigma) > 0):
        #     raise ValueError("The pdf only exists if 'sigma' is positive definite.")
        return mvnormal_pdf(self.mu, self.sigma, self.dim, x)

    def logpdf(self, x):
        \"""Return the logarithm of the probabilty density evaluated
        at 'x'. 'x' has to be the same dimension as 'mu' 
        and has to have 'dtype=np.float64'.\"""
        return np.log(self.pdf(x))

    # ========
    # Sampling
    # ========

    def rand(self, n):
        \"""Sample n vectors from the distribution. This returns 
        a matrix of size (n, dim), where each row is a sample.\"""
        
        L = np.linalg.cholesky(self.sigma)
        d = self.mu.size
        out = np.empty((n,d), dtype=np.float64)
        for i in np.arange(n):
            out[i, :] = L @ mvnormal_rand(self.dim) + self.mean
        return out
        

# define wrapper to overcome failure of type inference
def MvNormal(mu, sigma):
    return MvNormal_c(np.array(mu, dtype=np.float64), np.array(sigma, dtype=np.float64))
    """
    with open(os.path.join("rvlib", "multivariate.py"), "w") as f:
        f.write(textwrap.dedent(pre_code))


def main():
    # Write out specials.py
    _initiate_specials()
    
    # Write out multivariate
    _write_multivariate()

    # Preamble for univariate.py
    _initiate_univariate()

    # Normal
    _import_rmath("norm", "norm", "mu", "sigma")
    _write_class_specific(mtdt["Normal"], "mu", "sigma")
    _write_class_rmath("norm",  "norm", "mu", "sigma")

    # Chisq
    _import_rmath("chisq", "chisq", "v")
    _write_class_specific(mtdt["Chisq"], "v")
    _write_class_rmath("chisq",  "chisq", "v")

    # Uniform
    _import_rmath("unif", "unif", "a", "b")
    _write_class_specific(mtdt["Uniform"], "a", "b")
    _write_class_rmath("unif",  "unif", "a", "b")

    # T
    _import_rmath("t", "tdist", "v")
    _write_class_specific(mtdt["T"], "v")
    _write_class_rmath("t", "tdist", "v")

    # LogNormal
    _import_rmath("lnorm", "lognormal", "mu", "sigma")
    _write_class_specific(mtdt["LogNormal"], "mu", "sigma")
    _write_class_rmath("lnorm", "lognormal", "mu", "sigma")

    # F
    _import_rmath("f", "fdist", "v1", "v2")
    _write_class_specific(mtdt["F"], "v1", "v2")
    _write_class_rmath("f", "fdist", "v1", "v2")

    # Gamma
    _import_rmath("gamma", "gamma", "alpha", "beta")
    _write_class_specific(mtdt["Gamma"], "alpha", "beta")
    _write_class_rmath("gamma", "gamma", "alpha", "beta")

    # Beta
    _import_rmath("beta", "beta", "alpha", "beta")
    _write_class_specific(mtdt["Beta"], "alpha", "beta")
    _write_class_rmath("beta", "beta", "alpha", "beta")

    # Exponential
    _import_rmath("exp", "exp", "theta")
    _write_class_specific(mtdt["Exponential"], "theta")
    _write_class_rmath("exp", "exp", "theta")

    # Cauchy
    _import_rmath("cauchy", "cauchy", "mu", "sigma")
    _write_class_specific(mtdt["Cauchy"], "mu", "sigma")
    _write_class_rmath("cauchy", "cauchy", "mu", "sigma")

    # Poisson
    _import_rmath("pois", "pois", "mu")
    _write_class_specific(mtdt["Poisson"], "mu")
    _write_class_rmath("pois", "pois", "mu")

    # Geometric
    _import_rmath("geom", "geom", "p")
    _write_class_specific(mtdt["Geometric"], "p")
    _write_class_rmath("geom", "geom", "p")

    # Binomial
    _import_rmath("binom", "binom", "n", "p")
    _write_class_specific(mtdt["Binomial"], "n", "p")
    _write_class_rmath("binom", "binom", "n", "p")

    # Logistic
    _import_rmath("logis", "logis", "mu", "theta")
    _write_class_specific(mtdt["Logistic"], "mu", "theta")
    _write_class_rmath("logis", "logis", "mu", "theta")

    # Weibull
    _import_rmath("weibull", "weibull", "alpha", "theta")
    _write_class_specific(mtdt["Weibull"], "alpha", "theta")
    _write_class_rmath("weibull", "weibull", "alpha", "theta")

    # Hypergeometric
    _import_rmath("hyper", "hyper", "s", "f", "n")
    _write_class_specific(mtdt["Hypergeometric"], "s", "f", "n")
    _write_class_rmath("hyper", "hyper", "s", "f", "n")

    # NegativeBinomial
    _import_rmath("nbinom", "nbinom", "r", "p")
    _write_class_specific(mtdt["NegativeBinomial"], "r", "p")
    _write_class_rmath("nbinom", "nbinom", "r", "p")

with open(os.path.join("rvlib", "metadata.yaml"), 'r') as ymlfile:
    mtdt = yaml.load(ymlfile)
    main()
