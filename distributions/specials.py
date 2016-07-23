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
