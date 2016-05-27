"""
Distributions with Numba-fied Rmath functions in python. 
"""

# see if there is a more efficient way to import all the
# rmath_* modules later
import rmath_norm
import rmath_chisq

from math import sqrt
import numpy as np


class DistFromRmath(object):

    def __init__(self):
        pass

    #==========================================================
    # try with eval and string
    def pdf(self, x):
        """The pdf value(s) evaluated at x."""
        return eval(self.dist_name+"pdf")(*self._rmath_params, x)
    #============================================================
