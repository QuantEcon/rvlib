import glob
import os
import textwrap
# import platform

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
# but these are sufficient for our example purposes and gives a sense of
# the types of functions we can get
ffi.cdef('''\
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

// Not sure what we are going to use these for
//double  log1pmx(double);
//double  lgamma1p(double);
//double  logspace_add(double, double);
//double  logspace_sub(double, double);

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

// Non-central Chi-squared Distribution -- leave it out for now
// double	dnchisq(double, double, double, int);
// double	pnchisq(double, double, double, int, int);
// double	qnchisq(double, double, double, int, int);
// double	rnchisq(double, double);

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

// leave these out for now
// double	dnbinom_mu(double, double, double, int);
// double	pnbinom_mu(double, double, double, int, int);
// double	qnbinom_mu(double, double, double, int, int);
// double	rnbinom_mu(double, double);

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


def _import_rmath(rname, pyname, *pargs):
    """
    # now we map from the _rmath_ffi.lib.Xnorm functions
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
    import _rmath_ffi
    from numba import vectorize, jit
    from numba import cffi_support

    cffi_support.register_module(_rmath_ffi)

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

    # write code to file
    with open("rmath_{}.py".format(pyname), "w") as f:
        f.write(textwrap.dedent(code))

    if has_rand:
        rand_code = """\
        {rfun} = _rmath_ffi.lib.{rfun}

        @jit(nopython=True)
        def {rand}({p_args}):
            return {rfun}({p_args})

        """.format(**locals())
        with open("rmath_{}.py".format(pyname), "a") as f:
            f.write(textwrap.dedent(rand_code))


if __name__ == '__main__':
    # ffi.compile(verbose=False)
    _import_rmath("norm", "norm", "mu", "sigma")
    _import_rmath("unif", "uniform", "a", "b")
    _import_rmath("gamma", "gamma", "alpha", "beta")
    _import_rmath("beta", "beta", "alpha", "beta")
    _import_rmath("lnorm", "lognormal", "mu", "sigma")
    _import_rmath("chisq", "chisq", "v")
    _import_rmath("f", "fdist", "v1", "v2")
    _import_rmath("t", "tdist", "v")
    _import_rmath("binom", "n", "p")
    _import_rmath("cauchy", "cauchy", "mu", "sigma")
    _import_rmath("exp", "exp", "theta")
    _import_rmath("geom", "geom", "p")
    _import_rmath("hyper", "hyper", "s", "f", "n")
    _import_rmath("nbinom", "nbinom", "r", "p")
    _import_rmath("pois", "pois", "lambda")
    _import_rmath("weibull", "weibull", "alpha", "theta")
    _import_rmath("logis", "logis", "mu", "theta")