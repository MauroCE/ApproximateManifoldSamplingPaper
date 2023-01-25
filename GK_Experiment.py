""" 
Reproduces the G-and-K distribution inference problem.
"""
from numpy.random import default_rng
from numpy import concatenate, exp, array, zeros, eye
from scipy.stats import multivariate_normal as MVN
from scipy.special import ndtri
from Manifolds import GKManifold


def data_generator(θ0, m, seed):
    """Stochastic Simulator. Generates y given θ for the G-and-K problem."""
    rng = default_rng(seed)
    z = rng.normal(size=m)
    ξ = concatenate((θ0, z))
    return ξ[0] + ξ[1]*(1 + 0.8*(1 - exp(-ξ[2]*ξ[4:]))/(1 + exp(-ξ[2]*ξ[4:]))) * ((1 + ξ[4:]**2)**ξ[3])*ξ[4:]


def generate_setting(m, ϵs, Bs, δ, n_chains, n_samples):
    """Generates an object from which one can grab the settings. This allows one to run multiple scenarios."""
    θ0        = array([3.0, 1.0, 2.0, 0.5])      # True parameter value on U(0, 10) scale.
    d         = 4 + m                            # Dimensionality of ξ=(θ, z)
    ystar     = data_generator(θ0, m, seed=1234) # Observed data
    q         = MVN(zeros(d), eye(d))            # Proposal distribution for THUG
    manifold  = GKManifold(ystar)
    ξ0        = manifold.find_point_on_manifold_from_θ(ystar=ystar, θfixed=ndtri(θ0/10), ϵ=1e-5, maxiter=5000, tol=1e-15)
    return {
        'θ0': θ0,
        'm' : m,
        'd' : d,
        'ystar': ystar,
        'q': q,
        'ξ0': ξ0,
        'ϵs': ϵs,
        'Bs': Bs,
        'δ': δ,
        'n_chains': n_chains,
        'n_samples': n_samples,
        'manifold': manifold
    }


### Dimensionality of the data: 50
