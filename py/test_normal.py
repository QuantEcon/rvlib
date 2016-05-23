"""
Test Normal distribution with the Numba-fied versions of the Rmath functions.

This is from the corresponding Julia package.
Normal(μ,σ)
The *Normal distribution* with mean `μ` and standard deviation `σ` has probability density function
$f(x; \mu, \sigma) = \frac{1}{\sqrt{2 \pi \sigma^2}}
\exp \left( - \frac{(x - \mu)^2}{2 \sigma^2} \right)$


Normal()          # standard Normal distribution with zero mean and unit variance
Normal(mu)        # Normal distribution with mean mu and unit variance
Normal(mu, sig)   # Normal distribution with mean mu and variance sig^2
params(d)         # Get the parameters, i.e. (mu, sig)
mean(d)           # Get the mean, i.e. mu
std(d)            # Get the standard deviation, i.e. sig

External links

[Normal distribution on Wikipedia](http://en.wikipedia.org/wiki/Normal_distribution)
"""

import rmath_norm
import numpy as np
from numba import vectorize, jit
from math import inf


#  ------
#  Normal
#  ------


class Normal():
	"""
	The Normal distribution with mean mu and standard deviation sigma.

	Parameters
	----------
	mu : scalar(float)
		Mean of the normal distribution
	sigma : scalar(float)
		Standard deviaton of the normal distribution
	"""

	def __init__(self, mu, sigma):
		self.mu = mu
		self.sigma = sigma

	# ===================
	# Parameter retrieval
	# ===================

	@property
	def params(self):
		"""Returns parameters."""
		return (self.mu, self.sigma)

	@property
	def location(self):
		"""Returns parameters."""
		return self.mu

	@property
	def scale(self):
		"""Returns parameters."""
		return self.sigma
	
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
		return self.mu

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
		return  0.5 * (np.log(2*np.pi) + 1.0) + np.log(self.sigma)

	@property
	@vectorize(nopython=True)
	def mgf(self, x):
		"""Evaluate moment generating function at x."""
		return  np.exp(x * self.mu + 0.5 * self.sigma**2 * x**2)

	@property
	@vectorize(nopython=True)
	def cf(self, x):
		"""Evaluate characteristic function at x."""
		return  np.exp(1j * x * self.mu - 0.5 * self.sigma**2 * x**2)

	# ==========
	# Evaluation
	# ==========

	@property
	@vectorize(nopython=True)
	def insupport(self, x):
		"""When x is a scalar, it returns whether x is within the support of
		the distribution. When x is an array, it returns whether every element"""
		return  -inf < x < inf

	@property
	def pdf(self, x):
		"""The pdf value(s) evaluated at x."""
		return  norm_pdf(self.mu, self.sigma, x)


	@property
	def logpdf(self, x):
		"""The logarithm of the pdf value(s) evaluated at x."""
		return  norm_logpdf(self.mu, self.sigma, x)

	@property
	def loglikelihood(self, x):
		"""The log-likelihood of the Normal distribution w.r.t. all
		samples contained in array x."""
		return sum(norm_logpdf(self.mu, self.sigma, x))

	@property
	def cdf(self, x):
		"""The cdf value(s) evaluated at x."""
		return  norm_cdf(self.mu, self.sigma, x)

	@property
	def logcdf(self, x):
		"""The logarithm of the cdf value(s) evaluated at x."""
		return  norm_logcdf(self.mu, self.sigma, x)

	@property
	def logcdf(self, x):
		"""The logarithm of the cdf value(s) evaluated at x."""
		return  norm_logcdf(self.mu, self.sigma, x)

	@property
	def ccdf(self, x):
		"""The complementary cdf evaluated at x, i.e. 1- cdf(x)."""
		return  norm_ccdf(self.mu, self.sigma, x)

	@property
	def logccdf(self, x):
		"""The logarithm of the complementary cdf evaluated at x."""
		return  norm_logccdf(self.mu, self.sigma, x)

	@property
	def quantile(self, q):
		"""The quantile value evaluated at q."""
		return  norm_invcdf(self.mu, self.sigma, q)

	@property
	def cquantile(self, q):
		"""The complementary quantile value evaluated at q."""
		return  norm_invccdf(self.mu, self.sigma, q)

	@property
	def invlogcdf(self, lq):
		"""The inverse function of logcdf."""
		return  norm_invlogcdf(self.mu, self.sigma, lq)

	@property
	def invlogccdf(self, lq):
		"""The inverse function of logccdf."""
		return  norm_invlogccdf(self.mu, self.sigma, lq)

	# ========
	# Sampling
	# ========

	@property
	def rand(self, n=1):
		"""Generates a random draw from the distribution."""
		return  np.ndarray([norm_rand(self.mu, self.sigma) for x in range(n)])

	
