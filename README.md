Rmath-julia
===========

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
