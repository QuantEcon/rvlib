## `rvlib`

Anyone who has used [`Distributions.jl`](https://github.com/JuliaStats/Distributions.jl) will tell
you how nice the interface is relative to the "exotic" (the most polite word
we can think of) interface to distributions exposed by
[scipy.stats](http://docs.scipy.org/doc/scipy-0.17.1/reference/stats.html).
`Distributions.jl` also brings better performace, particularly when its
methods are used inside loops.

For these reason we've put together `rvlib`, which mimics the
interface of [`Distributions.jl`](https://github.com/JuliaStats/Distributions.jl), while at the same
time attaining similar performance by exploiting [`numba`](http://numba.pydata.org/).

This package was inspired by Joshua Adelman's ([@synapticarbors](https://github.com/synapticarbors)) 
[blog post](https://www.continuum.io/blog/developer-blog/calling-c-libraries-numba-using-cffi) describing how 
to call the Rmath C library from numba using [CFFI](http://cffi.readthedocs.io/), and utilizes his build script 
to set up the CFFI interface.

### Objectives

* Follow the API of the `Distributions.jl` package as closely as possible 

* Create a python package that has better performance than `scipy.stats`. 

### Methodology

All the classes are marked for optimization using the `@jitclass` decorator. As a result, instances of different distributions can be called within user specific routines or passed as arguments in `nopython` mode using `numba`.

The evaluation and sampling methods are built on the `Rmath` C library -- also used by the `Distributions.jl` package.

### Distributions currently implemented

Univariate continuous:

* Normal
* Chisq
* Uniform
* T
* Log-normal
* F
* Beta
* Gamma
* Exponential
* Cauchy
* Logistic
* Weibull

Univariate discrete:

* Poisson
* Geometric
* Hypergeometric
* Binomial
* Negative Binomial


Multivariate continuous:

* check for updates on mulitvariate normal in `multivariate` branch

### Functionality

The following properties are shared by all the univariate distributions:

* `params`: tuple of the distribution's parameters
* `location`: the location of the distribution (if exists)
* `scale`: the scale of the distribution (if exists)
* `shape`: the shape of the distribution (if exists)
* `mean`: the mean of the distribution
* `median`: the median of the distribution
* `mode`: the mode of the distribution
* `var`: the variance of the distribution
* `std`: the standard deviation of the distribution
* `skewness`: the skewness of the distribution
* `kurtosis`: the kurtosis of the distribution
* `isplaykurtic`: boolean indicating if kurtosis is greater than zero
* `isleptokurtic`: boolean indicating if kurtosis is less than zero
* `ismesokurtic`: boolean indicating if kurtosis is equal to zero
* `entropy`: the entropy of the distribution

The following methods can be called for all univariate distributions:

* `mgf`: evaluate the moment generating function (if exists)
* `cf`: evaluate the characteristic function (if exists)
* `pdf`: evaluate the probability density function
* `logpdf`: evaluate the logarithm of the prabability density function
* `loglikelihood`: evaluate the log-likelihood of the distribution with respect to all samples contained in array x
* `cdf`: evaluate the cumulative distribution function
* `ccdf`: evaluate the complementary cdf, i.e. (1 - cdf)
* `logcdf`: evaluate the logarithm of the cdf
* `logccdf`: evaluate the logarithm of the complementary cdf
* `quantile`: evaluate the quantile function at a critical value
* `cquantile`: evaluate the complementary quantile function
* `invlogcdf`: evaluate the inverse function of the logcdf
* `invlogccdf`: evaluate the inverse function of the logccdf
* `rand`: generate array of independent random draws

Seed setting

As the package is built around the `Rmath` library the seed for the random number generator has to be set using the `Rmath` `set_seed(x,y)` function. For example:

```python
import rvlib as rl

rl.set_seed(123, 456) # note that it requires two arguments
```


### Use and Performance

Preliminary comparison with the `scipy.stats` package.

```python
from rvlib import Normal
from scipy.stats import norm
import numpy as np
import timeit

N_dist = Normal(0,1) # rvlib version
N_scipy = norm(0,1) # scipy.stats version

x = np.linspace(0,100,100)
```


```python
In [1]: %timeit N_dist.pdf(x)
Out[1]: The slowest run took 8.85 times longer than the fastest. This could mean that an intermediate result is being cached.
    100000 loops, best of 3: 9.69 µs per loop
    
In [2]: %timeit N_scipy.pdf(x)
Out[2]: 10000 loops, best of 3: 150 µs per loop
```


```python
In [3]: %timeit N_dist.cdf(x)
Out[3]: The slowest run took 20325.82 times longer than the fastest. This could mean that an intermediate result is being cached.
    100000 loops, best of 3: 8.08 µs per loop

In [4]: %timeit N_scipy.cdf(x)
Out[4]:The slowest run took 190.64 times longer than the fastest. This could mean that an intermediate result is being cached.
    10000 loops, best of 3: 126 µs per loop
```


```python
In [5]: %timeit N_dist.rand(1000)
Out[5]: The slowest run took 2166.80 times longer than the fastest. This could mean that an intermediate result is being cached.
    10000 loops, best of 3: 85.8 µs per loop
    
In [6]: %timeit N_scipy.rvs(1000)
Out[6]: 10000 loops, best of 3: 119 µs per loop
```


# Contributors

* Daniel Csaba (daniel.csaba@nyu.edu)
* Spencer Lyon (spencer.lyon@stern.nyu.edu)

---

This is a fork of the [Rmath-julia](https://github.com/JuliaLang/Rmath-julia)
library, with Python support added.

The original readme of the Rmath-julia repository is included below.

---

## Rmath-julia

This is the Rmath library from R, which is used mainly by Julia's
[Distributions.jl](https://github.com/JuliaStats/Distributions.jl)
package.

The main difference here is that this library has been patched to use
the [DSFMT](http://www.math.sci.hiroshima-u.ac.jp/~m-mat/MT/SFMT/) RNG
in `src/runif.c`.

The Julia RNG is in sync with the one used by the Distributions.jl package:

````
julia> srand(1);

julia> [rand(), rand()]
2-element Array{Float64,1}:
 0.236033
 0.346517

julia> srand(1);

julia> using Distributions

julia> [rand(Uniform()), rand(Uniform())]
2-element Array{Float64,1}:
 0.236033
 0.346517
````

### Build instructions

Rmath-julia requires GNU Make (https://www.gnu.org/software/make). Just run
`make` to compile the library.
