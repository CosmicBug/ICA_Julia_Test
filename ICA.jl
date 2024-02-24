import Pkg 
Pkg.add("MultivariateStats")
Pkg.add("LinearAlgebra")
Pkg.add("Random")
Pkg.add("StatsBase")
Pkg.add("Plots")

using MultivariateStats
using LinearAlgebra
using Random
using StatsBase
using Plots

function generatetestdata(rng, n, k, m)
    t = range(0.0, step=0.2, length=n)
    s1 = sin.(t * 2)
    s2 = s2 = 1.0 .- 2.0 * Bool[isodd(floor(Int, x / 3)) for x in t]
    s3 = Float64[mod(x, 5.0) for x in t]

    # s1 += 0.1 * randn(rng, n)
    # s2 += 0.1 * randn(rng, n)
    # s3 += 0.1 * randn(rng, n)

    S = hcat(s1, s2, s3)'
    @assert size(S) == (k, n)

    A = randn(rng, m, k) #Mixing matrix

    X = A * S
    mv = vec(mean(X, dims=2))
    @assert size(X) == (m, n)
    C = cov(X, dims=2)
    return X, S
end

nn = 1000 #Number of samples
k = 3
m = 3

rng = Random.seed!(123)

X, S = generatetestdata(rng, nn, k, m)

# Compute ICA
ica_model = fit(ICA, X, 3; do_whiten=true)

# Reconstruct signals
Recon = transform(ica_model, X)

# Estimated mixing matrix
A_ = inv(ica_model.W)

p21 = plot(S[1,1:120],label=:false, grid=:false)
plot!(Recon[3,1:120],label=:false, grid=:false)
p22 = plot(S[2,1:120].-1,label=:false, grid=:false)
plot!(Recon[2,1:120]*-1,label=:false, grid=:false)
p23 = plot(S[3,1:120],label=:false, grid=:false)
plot!(Recon[1,1:120]*-1,label=:false, grid=:false)
plot(p21, p22, p23, layout=(3, 1), size=(850,550))