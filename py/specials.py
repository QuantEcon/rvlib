# =======================================================
# @vectorize-d functions for the Distributions.py package
# =======================================================

import numpy as np
from numba import vectorize

# -------
# digamma
# -------

@vectorize(nopython=True)
def digamma(x):
	"""Approximates the digamma function using
	the Euler-Maclaurin formula. For small values
	use the following recurrence formula to shift 
	x above 6:
	\psi(x-1) = \psi(x) - 1/x

	Details can be found
	https://en.wikipedia.org/wiki/Digamma_function

	Parameters:
	-----------
	x: float, positive real number

	Returns:
	--------
	float, value of the digamma function
	"""

	if x < 0.0:
		return np.nan

	value = 0.0

	while x < 6.5:
		value = value - 1/x
		x += 1 

	value = value + ( np.log(x) - 1/(2*x) - 1/(12*x**2) \
			+ 1/(120*x**4)  - 1/(252*x**6) + 1/(240*x**8) \
			- 5/(660*x**10) + 691/(32760*x**12) - 1/(12*x**14))

	return value