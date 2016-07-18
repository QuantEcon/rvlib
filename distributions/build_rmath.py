import glob
import os
import textwrap

import yaml

from cffi import FFI

include_dirs = [os.path.join('..', 'src'),
                os.path.join('..', 'include')]

rmath_src = glob.glob(os.path.join('..', 'src', '*.c'))

# Take out dSFMT dependant files; Just use the basic rng
rmath_src = [f for f in rmath_src if ('librandom.c' not in f) and ('randmtzig.c' not in f)]

extra_compile_args = ['-DMATHLIB_STANDALONE']
extra_compile_args.append('-std=c99')

ffi = FFI()
ffi.set_source('_rmath_ffi', '#include <Rmath.h>',
               include_dirs=include_dirs,
               sources=rmath_src,
               libraries=[],
               extra_compile_args=extra_compile_args)

# This is an incomplete list of the available functions in Rmath
# add new one with extensions of implemented distributions

ffi.cdef('''\

// Special functions
double  gammafn(double);
double  lgammafn(double);
double  digamma(double);
double  beta(double, double);
double  bessel_k(double, double, double);

// Distributions

// Normal Distribution
double dnorm(double, double, double, int);
double pnorm(double, double, double, int, int);
double rnorm(double, double);
double qnorm(double, double, double, int, int);

// Uniform Distribution
double	dunif(double, double, double, int);
double	punif(double, double, double, int, int);
double	qunif(double, double, double, int, int);
double	runif(double, double);

// Gamma Distribution
double	dgamma(double, double, double, int);
double	pgamma(double, double, double, int, int);
double	qgamma(double, double, double, int, int);
double	rgamma(double, double);

// Beta Distribution
double	dbeta(double, double, double, int);
double	pbeta(double, double, double, int, int);
double	qbeta(double, double, double, int, int);
double	rbeta(double, double);

// Lognormal Distribution
double	dlnorm(double, double, double, int);
double	plnorm(double, double, double, int, int);
double	qlnorm(double, double, double, int, int);
double	rlnorm(double, double);

// Chi-squared Distribution
double	dchisq(double, double, int);
double	pchisq(double, double, int, int);
double	qchisq(double, double, int, int);
double	rchisq(double);

// F Distibution
double	df(double, double, double, int);
double	pf(double, double, double, int, int);
double	qf(double, double, double, int, int);
double	rf(double, double);

// Student t Distibution
double	dt(double, double, int);
double	pt(double, double, int, int);
double	qt(double, double, int, int);
double	rt(double);

// Binomial Distribution
double	dbinom(double, double, double, int);
double	pbinom(double, double, double, int, int);
double	qbinom(double, double, double, int, int);
double	rbinom(double, double);

// Cauchy Distribution
double	dcauchy(double, double, double, int);
double	pcauchy(double, double, double, int, int);
double	qcauchy(double, double, double, int, int);
double	rcauchy(double, double);

// Exponential Distribution
double	dexp(double, double, int);
double	pexp(double, double, int, int);
double	qexp(double, double, int, int);
double	rexp(double);

// Geometric Distribution
double	dgeom(double, double, int);
double	pgeom(double, double, int, int);
double	qgeom(double, double, int, int);
double	rgeom(double);

// Hypergeometric Distibution
double	dhyper(double, double, double, double, int);
double	phyper(double, double, double, double, int, int);
double	qhyper(double, double, double, double, int, int);
double	rhyper(double, double, double);

// Negative Binomial Distribution
double	dnbinom(double, double, double, int);
double	pnbinom(double, double, double, int, int);
double	qnbinom(double, double, double, int, int);
double	rnbinom(double, double);

// Poisson Distribution
double	dpois(double, double, int);
double	ppois(double, double, int, int);
double	qpois(double, double, int, int);
double	rpois(double);

// Weibull Distribution
double	dweibull(double, double, double, int);
double	pweibull(double, double, double, int, int);
double	qweibull(double, double, double, int, int);
double	rweibull(double, double);

// Logistic Distribution
double	dlogis(double, double, double, int);
double	plogis(double, double, double, int, int);
double	qlogis(double, double, double, int, int);
double	rlogis(double, double);
''')


# =========================================
# write out preamble for special functions
# =========================================

def _initiate_sepcials():
    '''
    Initiate python file for special functions which are present in the Rmath.h file
    '''

    # Define code which appears irrespective of specific functions
    pre_code = """\
    import _rmath_ffi
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

    """
    with open("specials.py", "w") as f:
        f.write(textwrap.dedent(pre_code))


# =========================================
# write out preamble for classes
# =========================================

def _initiate_univariate():
    '''
    Initiate python file which collects all the classes of different distributions.
    '''

    # Define code which appears irrespective of specific classes of distributions
    pre_code = """\
    from numba import vectorize, jit, jitclass
    from numba import int32, float32

    import numpy as np
    from math import inf, ceil, floor
    from .specials import gamma, lgamma, digamma, beta, bessel_k

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
    with open("univariate.py", "w") as f:
        f.write(textwrap.dedent(pre_code))


# ======================================================================
# globals called from the textwrapper with distribution specific content
# ======================================================================


# function to import and @vectorize the 
# distribution specific rmath functions 

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
    with open("univariate.py", "a") as f:
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
    with open("univariate.py", "a") as f:
        f.write(textwrap.dedent(rand_code))




# function to write out the dist specific part

def _write_class_specific(metadata, *pargs):

    # write out the distribution specific part 
    # of the class which is not related to the 
    # imported rmath functions 

    # use _metadata_DIST and some derived locals

    p_args = ", ".join(pargs)

    p_args_self = ", ".join(["".join(("self.", par)) for par in pargs])


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
            \"""Returns parameters.\"""
            return ({p_args_self})

        @property
        def location(self):
            \"""Returns location parameter if exists.\"""
            return {loc}

        @property
        def scale(self):
            \"""Returns scale parameter if exists.\"""
            return {scale}

        @property
        def shape(self):
            \"""Returns shape parameter if exists.\"""
            return {shape}

        # ==========
        # Statistics
        # ==========

        @property
        def mean(self):
            \"""Returns mean.\"""
            return {mean}

        @property
        def median(self):
            \"""Returns median.\"""
            return {median}

        @property
        def mode(self):
            \"""Returns mode.\"""
            return {mode}

        @property
        def var(self):
            \"""Returns variance.\"""
            return {var}

        @property
        def std(self):
            \"""Returns standard deviation.\"""
            return {std}

        @property
        def skewness(self):
            \"""Returns skewness.\"""
            return {skewness}

        @property
        def kurtosis(self):
            \"""Returns kurtosis.\"""
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
            \"""Returns entropy.\"""
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
            it returns whether every element.\"""
            return {insupport}
        """.format(**locals(), **metadata)

    # append code for class to main file
    with open("univariate.py", "a") as f:
        f.write(textwrap.dedent(class_specific))


def _write_class_rmath(rname, pyname, *pargs):
    """
    # call top level @vectorized evaluation methods
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
    with open("univariate.py", "a") as f:
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
        \"""Generates a random draw from the distribution.\"""
        out = np.empty(n)
        for i, _ in np.ndenumerate(out):
            out[i] = {rand}({p_args})
        return out

    """.format(**locals())
    with open("univariate.py", "a") as f:
        f.write(rand_code)

with open("metadata.yml", 'r') as ymlfile:
    mtdt = yaml.load(ymlfile)



if __name__ == '__main__':
    # ffi.compile(verbose=True)
    
    # Write out specials.py
    _initiate_sepcials()

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