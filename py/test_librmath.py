import numpy as np
import numba as nb

import scipy.stats

# Import our Rmath module
import _rmath_ffi

dnorm = _rmath_ffi.lib.dnorm
pnorm = _rmath_ffi.lib.pnorm

dunif = _rmath_ffi.lib.dunif
punif = _rmath_ffi.lib.punif

dgamma = _rmath_ffi.lib.dgamma
pgamma = _rmath_ffi.lib.pgamma

from numba import cffi_support

cffi_support.register_module(_rmath_ffi)

@nb.jit(nopython=True)
def pnorm_nb(x):
    y = np.empty_like(x)
    for k in range(x.shape[0]):
        y[k] = pnorm(x[k], 0.0, 1.0, 1, 0)

    return y

@nb.vectorize(nopython=True)
def pnorm_nb_vec(x):
    return pnorm(x, 0.0, 1.0, 1, 0)

x = np.random.normal(size=(100,))

y1 = scipy.stats.norm.cdf(x)
y2 = pnorm_nb(x)
y3 = pnorm_nb_vec(x)

# Check that they all give the same results
print(np.allclose(y1, y2))
print(np.allclose(y1, y3))
