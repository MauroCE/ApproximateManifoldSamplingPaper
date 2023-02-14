"""  
Reproduces the second plot for the BIP experiment, specifically the computational cost one. Here
we only compare THUG, CRWM, and HMC (RM-HMC takes so long that it makes no sense to compare it against the others).
"""
from numpy import logspace, concatenate, linspace, finfo, float64, cos, pi, stack, array, full, seterr
from numpy import zeros, eye, log, nan, save
from numpy.linalg import norm
from numpy.random import default_rng
from scipy.stats import multivariate_normal as MVN
from Manifolds import BIPManifold
from ConstrainedRWM import CRWM
from TangentialHug import THUG
from HelperFunctions import compute_arviz_miness_runtime
from HMC import HMC
import os
import collections.abc
collections.Iterable = collections.abc.Iterable
collections.Mapping = collections.abc.Mapping
collections.MutableSet = collections.abc.MutableSet
collections.MutableMapping = collections.abc.MutableMapping
import sympy
from symnum import named_array
import time


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
    δ = 0.1   # Step size is fixed here.

    # Settings
    N         = 2500  # Number of samples
    n_chains  = 12
    B         = 20  # Number of bounces (THUG) and Leapfrog Steps (CRWM/HMC/RM-HMC)

    # Grids of values for scale and tolerance 
    σ_grid = logspace(start=0.0, stop=-6, num=7, endpoint=True, base=10.0) # Noise Scale

    # Initial points on the manifold
    θ_inits = concatenate(solve_for_limiting_manifold(y, n_chains))

    # Initialize storage for the results of computational cost
    THUG_CC = zeros(len(σ_grid))
    THUG99_CC = zeros(len(σ_grid))
    CRWM_CC = zeros(len(σ_grid))
    HMC_CC  = zeros(len(σ_grid))

    # Initialize storage for the results of acceptance probability
    THUG_AP   = zeros(len(σ_grid))
    THUG99_AP = zeros(len(σ_grid))
    CRWM_AP   = zeros(len(σ_grid))
    HMC_AP    = zeros(len(σ_grid))

    # Seed for reproducibility
    seeds = [1122, 2233, 3344, 4455, 5566, 6677, 7788, 8899, 9900, 1100, 1830, 1038]
    rngs  = [default_rng(seed=seed) for seed in seeds]
    _ = seterr(invalid='ignore', over='ignore')

    # Functions for THUG
    prior_log_dens = lambda x: MVN(zeros(2), eye(2)).logpdf(x)
    grad_log_prior = lambda x: -x
    F = lambda θ: array([θ[1]**2 + 3 * θ[0]**2 * (θ[0]**2 - 1)])   # Forward function
    grad_F = lambda θ: array([12*θ[0]**3 - 6*θ[0], 2*θ[1]])

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
        chains = {'THUG': [], 'CRWM': [], 'HMC': [], 'THUG99': []}
        times  = {'THUG': [], 'CRWM': [], 'HMC': [], 'THUG99': []}
        avg_ap = {'THUG': 0.0, 'CRWM': 0.0, 'HMC': 0.0, 'THUG99': 0.0}
        for chain_ix in range(n_chains): # for each chain
            # Run Tangential Hug (alpha=0.0)
            start_time_thug = time.time()
            sTHUG, aTHUG = THUG(θ_inits[chain_ix], B*δ, B, N, α=0.0, logpi=log_posterior, jac=grad_F, method='2d', rng=rngs[chain_ix])
            runtime_thug = time.time() - start_time_thug
            chains['THUG'].append(sTHUG)
            times['THUG'].append(runtime_thug)
            avg_ap['THUG'] += (aTHUG.mean()/n_chains)
            # Run Tangential Hug (alpha=0.99)
            start_time_thug99 = time.time()
            sTHUG99, aTHUG99 = THUG(θ_inits[chain_ix], B*δ, B, N, α=0.99, logpi=log_posterior, jac=grad_F, method='2d', rng=rngs[chain_ix])
            runtime_thug99 = time.time() - start_time_thug99
            chains['THUG99'].append(sTHUG99)
            times['THUG99'].append(runtime_thug99)
            avg_ap['THUG99'] += (aTHUG99.mean()/n_chains)
            # Run C-RWM (rattle version)
            start_time_crwm = time.time()
            sCRWM, _, aCRWM = CRWM(q_inits[chain_ix], manifold, N, B*δ, B, tol=1e-14, rev_tol=1e-14, rng=rngs[chain_ix])
            runtime_crwm = time.time() - start_time_crwm
            chains['CRWM'].append(sCRWM)
            times['CRWM'].append(runtime_crwm)
            avg_ap['CRWM'] += (aCRWM.mean()/n_chains)
            # Run HMC (on true posterior, not lifted)
            neg_log_post = lambda θ: - log_posterior(θ)
            grad_neg_log_post = lambda θ: - grad_log_prior(θ) - (y - F(θ))*grad_F(θ) / (σ**2)
            hmc_sampler =  HMC(θ_inits[chain_ix], N, eye(2), B*δ, δ)
            hmc_sampler.dVdq = grad_neg_log_post
            hmc_sampler.neg_log_target = neg_log_post
            start_time_hmc = time.time()
            sHMC, aHMC = hmc_sampler.sample()
            runtime_hmc = time.time() - start_time_hmc
            chains['HMC'].append(sHMC)
            times['HMC'].append(runtime_hmc)
            avg_ap['HMC'] += (aHMC.mean()/n_chains)
        # Compute ESS across chains
        THUG_CC[σ_ix]   = compute_arviz_miness_runtime(chains['THUG'], times['THUG'])
        THUG99_CC[σ_ix] = compute_arviz_miness_runtime(chains['THUG99'], times['THUG99'])
        CRWM_CC[σ_ix]   = compute_arviz_miness_runtime(chains['CRWM'], times['CRWM'])
        HMC_CC[σ_ix]    = compute_arviz_miness_runtime(chains['HMC'], times['HMC'])
        # Store average acceptance probability
        THUG_AP[σ_ix]   = avg_ap['THUG']
        THUG99_AP[σ_ix] = avg_ap['THUG99']
        CRWM_AP[σ_ix]   = avg_ap['CRWM']
        HMC_AP[σ_ix]    = avg_ap['HMC']


            

    # Construct folder where to save the data
    mainfolder = "BIP_Experiment"
    subfolder = "_".join([str(seed) for seed in seeds])
    folder = os.path.join(mainfolder, subfolder)
    if not os.path.exists(folder):
        os.makedirs(folder)

    # Save ESS data
    save(os.path.join(folder, 'THUG_CC.npy'), THUG_CC)
    save(os.path.join(folder, 'THUG99_CC.npy'), THUG99_CC)
    save(os.path.join(folder, 'CRWM_CC.npy'), CRWM_CC)
    save(os.path.join(folder, 'HMC_CC.npy'), HMC_CC)

    # Save AP data
    save(os.path.join(folder, 'THUG_AVG_AP.npy'), THUG_AP)
    save(os.path.join(folder, 'THUG99_AVG_AP.npy'), THUG99_AP)
    save(os.path.join(folder, 'CRWM_AVG_AP.npy'), CRWM_AP)
    save(os.path.join(folder, 'HMC_AVG_AP.npy'), HMC_AP)

    # Save noise scales and step sizes
    save(os.path.join(folder, 'SIGMA_GRID_CC.npy'), σ_grid)
