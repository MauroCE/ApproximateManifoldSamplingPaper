"""
Reproduces the Bayesian Inverse Problem experiment. We compare THUG, C-RWM, RM-HMC, and HMC.
THUG, RM-HMC, HMC are on the true posterior p_sigma(theta mid y*), whereas C-RWM is on
the lifted posterior p_sigma(theta, upsilon mid y*).
"""
from numpy import logspace, concatenate, linspace, finfo, float64, cos, pi, stack, array, full, seterr
from numpy import zeros, eye, log, nan, save
from numpy.linalg import norm
from numpy.random import default_rng
from scipy.stats import multivariate_normal as MVN
from Manifolds import BIPManifold
from ConstrainedRWM import CRWM
from TangentialHug import THUG
from HMC import HMC
import collections.abc
collections.Iterable = collections.abc.Iterable
collections.Mapping = collections.abc.Mapping
collections.MutableSet = collections.abc.MutableSet
collections.MutableMapping = collections.abc.MutableMapping
import sympy
from symnum import (
    numpify, named_array, jacobian, grad, 
    vector_jacobian_product, matrix_hessian_product)
from mici.systems import DenseRiemannianMetricSystem as DRMS
from mici.integrators import ImplicitLeapfrogIntegrator as ILI
from mici.samplers import StaticMetropolisHMC as SMHMC
from functools import partial
import symnum.numpy as snp
import os


### The following four functions `forward_func`, `split_into_integer_parts`, `grid_on_interval`,
### and `solve_for_limiting_manifold` are taken from the Notebooks accompanying 
### the paper "Manifold lifting: scaling MCMC to the vanishing noise regime"
### by Au et al 2020. In particular these functions can be found here
### https://github.com/thiery-lab/manifold_lifting/blob/master/notebooks/Two-dimensional_example.ipynb

def forward_func(θ):
    return array([θ[1]**2 + 3 * θ[0]**2 * (θ[0]**2 - 1)])

def split_into_integer_parts(n, m):
    return [round(n / m)] * (m - 1) + [n - round(n / m) * (m - 1)]

def grid_on_interval(interval, n_points, cosine_spacing=False):
    if cosine_spacing:
        # Use non-linear spacing with higher density near endpoints
        ts =  ((1 + cos(linspace(0, 1, n_points) * pi)) / 2)
    else:
        ts = linspace(0, 1, n_points)
    # If open interval space over range [left + eps, right - eps]
    eps = 10 * finfo(float64).eps
    left = (float(interval.left) + eps if interval.left_open 
            else float(interval.left))
    right = (float(interval.right) - eps if interval.right_open 
             else float(interval.right))
    return left + ts * (right - left)


def solve_for_limiting_manifold(y, n_points=200, cosine_spacing=False):
    assert n_points % 2 == 0, 'n_points must be even'
    θ = named_array('θ', 2)
    # solve F(θ) = y for θ[1] in terms of θ[0]
    θ_1_gvn_θ_0 = sympy.solve(forward_func(θ)[0] - y, θ[1])
    # find interval(s) over which θ[0] gives real θ[1] solutions
    # what is theta here ?
    θ_0_range = sympy.solveset(
        θ_1_gvn_θ_0[0]**2 > 0, θ[0], domain=sympy.Reals)
    θ_0_intervals = (
        θ_0_range.args if isinstance(θ_0_range, sympy.Union) 
        else [θ_0_range])
    # create  grid of values over valid θ[0] interval(s)
    n_intervals = len(θ_0_intervals)
    θ_0_grids = [
        grid_on_interval(intvl, n_pt + 1, cosine_spacing)
        for intvl, n_pt in zip(
            θ_0_intervals, 
            split_into_integer_parts(n_points // 2, n_intervals))]
    # generate NumPy function to calculate θ[1] in terms of θ[0]
    solve_func = sympy.lambdify(θ[0], θ_1_gvn_θ_0)
    manifold_points = []
    for θ_0_grid in θ_0_grids:
        # numerically calculate +/- θ[1] solutions over θ[0] grid
        θ_1_grid_neg, θ_1_grid_pos = solve_func(θ_0_grid)
        # stack θ[0] and θ[1] values in to 2D array in anticlockwise order
        manifold_points.append(stack([
            concatenate([θ_0_grid, θ_0_grid[-2:0:-1]]),
            concatenate([θ_1_grid_neg, θ_1_grid_pos[-2:0:-1]])
        ], -1))
    return manifold_points


if __name__ == "__main__":
    # Posterior Parameters
    σ = 0.1
    y = 1
    dim_θ = 2
    dim_y = 1

    # Settings
    N         = 50  # Number of samples
    n_chains  = 10
    B         = 20  # Number of bounces (THUG) and Leapfrog Steps (CRWM/HMC/RM-HMC)

    # Grids of values for scale and tolerance 
    σ_grid = logspace(-5, 0, 11)    # Noise scale
    δ_grid = logspace(-5, 0, 11)    # Step size

    # Initial points on the manifold
    θ_inits = concatenate(solve_for_limiting_manifold(y, n_chains))

    # Initialize storage for the results
    crwm_av_accept_probs  = full((σ_grid.shape[0], δ_grid.shape[0]), 0.0)
    hmc_av_accept_probs   = full((σ_grid.shape[0], δ_grid.shape[0]), 0.0)
    rmhmc_av_accept_probs = full((σ_grid.shape[0], δ_grid.shape[0]), nan)
    thug_av_accept_probs  = full((σ_grid.shape[0], δ_grid.shape[0]), 0.0)



    # Seed for reproducibility
    rng = default_rng(seed=20200310)
    seeds = [1122, 2233, 3344, 4455, 5566, 6677, 7788, 8899, 9900, 1100]
    rngs  = [default_rng(seed=seed) for seed in seeds]
    _ = seterr(invalid='ignore', over='ignore')

    # Functions for THUG
    prior_log_dens = lambda x: MVN(zeros(2), eye(2)).logpdf(x)
    grad_log_prior = lambda x: -x
    F = lambda θ: array([θ[1]**2 + 3 * θ[0]**2 * (θ[0]**2 - 1)])   # Forward function
    grad_F = lambda θ: array([12*θ[0]**3 - 6*θ[0], 2*θ[1]])
    # Functions for HMC and RM-HMC (from MICI package)
    @numpify(dim_θ)
    def forward_func(θ):
        return snp.array([θ[1]**2 + 3 * θ[0]**2 * (θ[0]**2 - 1)])

    @numpify(dim_θ + dim_y)
    def neg_log_prior_dens(q):
        return snp.sum(q**2) / 2

    @numpify(dim_θ, None, None)
    def neg_log_posterior_dens(θ, σ, y):
        return (snp.sum(θ**2, 0) + snp.sum((y - forward_func(θ))**2, 0) / σ**2) / 2

    @numpify(dim_θ, None)
    def metric(θ, σ):
        jac = jacobian(forward_func)(θ)
        return jac.T @ jac / σ**2 + snp.identity(dim_θ)

    @numpify(dim_θ, None, None, None)
    def neg_log_lifted_posterior_dens(θ, η, σ, y):
        jac = jacobian(forward_func)(θ)
        return snp.sum(θ**2, 0) / 2 + η**2 / 2 + snp.log(jac @ jac.T + σ**2)[0, 0] / 2

    @numpify(dim_θ + dim_y, None, None)
    def constr(q, σ, y):
        θ, η = q[:dim_θ], q[dim_θ:]
        return forward_func(θ) + σ * η - y
    grad_neg_log_posterior_dens = grad(neg_log_posterior_dens)
    grad_and_val_neg_log_posterior_dens = grad(neg_log_posterior_dens, return_aux=True)
    vjp_metric = vector_jacobian_product(metric, return_aux=True)
    grad_neg_log_prior_dens = grad(neg_log_prior_dens)
    jacob_constr = jacobian(constr, return_aux=True)
    mhp_constr = matrix_hessian_product(constr, return_aux=True)

    # Run Experiment
    for σ_ix, σ in enumerate(σ_grid):     # for each noise scale
        # Find `n_chains` points on σ-manifold (one for each chain)
        θ_inits = concatenate(solve_for_limiting_manifold(y, n_chains))          # Initial points on theta-manifold
        q_inits = [concatenate([θ, (y - forward_func(θ)) / σ]) for θ in θ_inits] # Corresponding points on xi-manifold
        # Instantiate lifted manifold corresponding to σ. Used by C-RWM.
        manifold = BIPManifold(σ=σ, ystar=y)
        # Instantiate functions for THUG and HMC
        log_posterior = lambda θ, y=y, σ=σ: prior_log_dens(θ) - norm(y - F(θ))**2 / (2*σ**2) - 1*log(σ)
        grad_log_post = lambda θ, y=y, σ=σ: grad_log_prior(θ) + (y - F(θ))*grad_F(θ) / (σ**2)
        for δ_ix, δ in enumerate(δ_grid): # for each step size
            for chain_ix in range(n_chains): # for each chain
                # Run Tangential Hug
                sTHUG, aTHUG = THUG(θ_inits[chain_ix], B*δ, B, N, α=0.0, logpi=log_posterior, jac=grad_F, method='2d', rng=rngs[chain_ix])
                thug_av_accept_probs[σ_ix, δ_ix] += (aTHUG.mean() / n_chains)
                # Run C-RWM (rattle version)
                sCRWM, _, aCRWM = CRWM(q_inits[chain_ix], manifold, N, B*δ, B, tol=1e-14, rev_tol=1e-14, rng=rngs[chain_ix])
                crwm_av_accept_probs[σ_ix, δ_ix] += (aCRWM.mean() / n_chains)
                # Run HMC (on true posterior, not lifted)
                neg_log_post = lambda θ: - log_posterior(θ)
                grad_neg_log_post = lambda θ: - grad_log_prior(θ) - (y - F(θ))*grad_F(θ) / (σ**2)
                hmc_sampler =  HMC(θ_inits[chain_ix], N, eye(2), B*δ, δ)
                hmc_sampler.dVdq = grad_neg_log_post
                hmc_sampler.neg_log_target = neg_log_post
                sHMC, aHMC = hmc_sampler.sample()
                hmc_av_accept_probs[σ_ix, δ_ix] += (aHMC.mean() / n_chains)
                # Run RM-HMC (on true posterior, not lifted).
                ### Careful: this will take a good while, perhaps 1 or 2 hours
                system = DRMS(neg_log_dens=partial(neg_log_posterior_dens, σ=σ, y=y), grad_neg_log_dens=partial(grad_neg_log_posterior_dens, σ=σ, y=y), metric_func=partial(metric, σ=σ), vjp_metric_func=partial(vjp_metric, σ=σ))
                system = DRMS(neg_log_dens=neg_log_post, grad_neg_log_dens=grad_neg_log_post, metric_func=partial(metric, σ=σ), vjp_metric_func=partial(vjp_metric, σ=σ))
                integrator = ILI(system, step_size=δ)
                sampler = SMHMC(system, integrator, rng, n_step=(N // 2))
                _, _, stats = sampler.sample_chains(N // 2, θ_inits, n_process=n_chains, display_progress=False)
                rmhmc_av_accept_probs[σ_ix, δ_ix] = concatenate([a for a in stats['accept_stat']]).mean()

    # Construct folder where to save the data
    mainfolder = "BIP_Experiment"
    subfolder = "_".join([str(seed) for seed in seeds])
    folder = os.path.join(mainfolder, subfolder)
    if not os.path.exists(folder):
        os.makedirs(folder)

    # Save data
    save(os.path.join(folder, 'THUG_AP.npy'), thug_av_accept_probs)
    save(os.path.join(folder, 'CRWM_AP.npy'), crwm_av_accept_probs)
    save(os.path.join(folder, 'HMC_AP.npy'), hmc_av_accept_probs)
    save(os.path.join(folder, 'RMHMC_AP.npy'), rmhmc_av_accept_probs)

    # Save noise scales and step sizes
    save(os.path.join(folder, 'SIGMA_GRID.npy'), σ_grid)
    save(os.path.join(folder, 'DELTA_GRID.npy'), δ_grid)
