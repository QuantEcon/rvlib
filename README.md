## `Distributions.py`

A Python library that mimics
[`Distributions.jl`](https://github.com/JuliaStats/Distributions.jl)

### Objectives

Following the API of the `Distributions.jl` package as closely as possible create a python package that has better performance than `scipy.stats`. 

For this end [`numba`](http://numba.pydata.org/) is used to speed up all the attributes of the distribution specific classes. 

All the classes are marked for optimization using the `@jitclass` decorator. As such, instances of different distributions can be called within user specific routines or passed as arguments in `nopython` mode using `numba`.

The evaluation and sampling methods are built on the `Rmath` C library -- also used by the `julia` package.

### Distributions currently implemented

Univariate continuous:

* Normal
* Chisq
* Uniform
* T

Multivariate continuous:

### Functionality

The following properties are shared by all the distributions:

* `params`: return a tuple of the distributions parameters
* `location`: the location of the distribution (if exists)
* `scale`: the scale of the distribution (if exists)
* `shape`: the shape of the distribution (if exists)
* `mean`: the mean of the distribution
* `median`: the median of the distribution
* `mode`: the mode of the distribution
* `var`: the var of the distribution
* `std`: the std of the distribution
* `skewness`: the skewness of the distribution
* `kurtosis`: the kurtosis of the distribution
* `isplaykurtic`: boolean indicating if kurtosis is greater than zero
* `isleptokurtic`: boolean indicating if kurtosis is less than zero
* `ismesokurtic`: boolean indicating if kurtosis is equal to zero
* `entropy`: the entropy of the distribution

The following methods can be called for all distributions:


* `mgf`: evaluate the moment generating function
* `cf`: evaluate the characteristic function
* `pdf`: evaluate the probability density function
* `logpdf`: evaluate the log of the pdf
* `loglikelihood`: the loglikelihood of the distribution with respect to all samples contained in array x
* `cdf`: evaluate the cumulative density function
* `ccdf`: evaluate the complementary cdf, i.e. (1 - cdf)
* `logcdf`: evaluate the log of the cdf
* `logccdf`: compute the log of the complementary cumulative density function
* `quantile`: compute the quantile at critical value
* `cquantile`: evaluate the complementary quantile function
* `invlogcdf`: evaluate inverse function of the logcdf
* `invlogccdf`: evaluate inverse function of the logccdf
* `rand`: generate array of independent random draws

--

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
