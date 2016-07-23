"""
Distributions.py: mimics the interface of Distributions.jl
<https://github.com/JuliaStats/Distributions.jl> while at the same time
attaining similar performance exploiting numba<http://numba.pydata.org/>.

@date: 07.06.2016

"""
__version__ = 0.1

from .univariate import *
from .specials import *
from ._rmath_ffi import *
