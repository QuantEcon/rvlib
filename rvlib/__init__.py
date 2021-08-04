"""
rvlib: mimics the interface of Distributions.jl
<https://github.com/JuliaStats/Distributions.jl> while at the same time
attaining similar performance exploiting numba<http://numba.pydata.org/>.

@authors :	Daniel Csaba <daniel.csaba@nyu.edu>
			Spencer Lyon <spencer.lyon@stern.nyu.edu>
@date : 2016-07-26

"""
from .version import version
__version__ = version

from .univariate import *
from .specials import *
from ._rmath_ffi import *
