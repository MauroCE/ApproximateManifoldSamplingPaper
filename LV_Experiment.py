"""
Reproduces the Lotka-Volterra experiment.
"""
import time 
from numpy import zeros, nan, repeat, save, unique
from copy import deepcopy
from ConstrainedRWM import CRWM
from HelperFunctions import compute_arviz_miness_runtime, generate_powers_of_ten
from TangentialHug import THUG
from Manifolds import LVManifold
import os


def generate_settings(N, δ, Ns, Bs, ϵs, seeds, n_chains=4, u1_true=True, tol=1e-15, maxiter=5000):
    """Generates variables for the experiment."""
    manifold = LVManifold(Ns=Ns, n_chains=n_chains, seeds=seeds)
    u0s = manifold.find_init_points_for_each_chain(u1_true=u1_true, tol=tol, maxiter=maxiter)
    print("Settings Ns = ", Ns, ". All rows equal? ", (u0s == u0s[0]).all())
    return {
        'N': N,
        'δ': δ,
        'Ns': Ns,
        'Bs': Bs,
        'ϵs': ϵs,
        'u0s': u0s,
        'manifold': manifold,
        'n_chains': n_chains,
        'seeds': seeds,
        'rngs': manifold.rngs
    }


def cc_experiment_thug(settings, α=0.0, verbose=False, safe=False):
    """Computational Cost of THUG and C-RWM."""
    verboseprint = print if verbose else lambda *a, **k: None
    ϵs, Bs = settings['ϵs'], settings['Bs']
    rngs = settings['rngs']
    u0s = settings['u0s']
    δ = settings['δ']
    N = settings['N']
    J = settings['manifold'].J
    n_chains = settings['n_chains']
    ESS_TABLE = zeros((len(ϵs), len(Bs)))
    AP_TABLE  = zeros((len(ϵs), len(Bs)))
    for ϵ_ix, ϵ in enumerate(ϵs):
        logηϵ = settings['manifold'].generate_logpi(ϵ)
        for B_ix, B in enumerate(Bs):
            chains = []
            times  = []
            avg_ap = 0.0
            for chain_ix in range(n_chains):
                start_time = time.time()
                s, a = THUG(u0s[chain_ix, :], B*δ, B, N, α, logηϵ, J, method='linear', rng=rngs[chain_ix], safe=safe)
                runtime = time.time() - start_time
                verboseprint("epsilon={} B={} time={} a={} uot={}".format(ϵ, B, runtime, a.mean(), unique(s, axis=0).shape[0]))
                chains.append(s)
                times.append(runtime)
                avg_ap += (a.mean() / n_chains)
            verboseprint()
            ESS_TABLE[ϵ_ix, B_ix] = compute_arviz_miness_runtime(chains, times)
            AP_TABLE[ϵ_ix, B_ix]  = avg_ap
    return ESS_TABLE, AP_TABLE


def cc_experiment_crwm(settings, tol=1e-11, rev_tol=1e-8, verbose=False):
    """Same as above but for C-RWM."""
    verboseprint = print if verbose else lambda *a, **k: None
    Bs = settings['Bs']
    u0s = settings['u0s']
    manifold = settings['manifold']
    N = settings['N']
    δ = settings['δ']
    n_chains = settings['n_chains']
    rngs = settings['rngs']
    ESS_TABLE = zeros(len(Bs))
    AP_TABLE  = zeros(len(Bs))
    for B_ix, B in enumerate(Bs):
        chains = []
        times  = []
        avg_ap = 0.0
        for chain_ix in range(n_chains):
            start_time = time.time()
            s, e, a = CRWM(u0s[chain_ix, :], manifold, N, δ*B, B, tol=tol, rev_tol=rev_tol, rng=rngs[chain_ix])
            runtime = time.time() - start_time
            verboseprint("B={} time={} a={}".format(B, runtime, a.mean()))
            chains.append(s)
            times.append(runtime)
            avg_ap += (a.mean() / n_chains)
        ESS_TABLE[B_ix] = compute_arviz_miness_runtime(chains, times)
        AP_TABLE[B_ix]  = avg_ap
    return ESS_TABLE, AP_TABLE


def show_only_positive_ap(cc, ap, ix):
    """USED FOR PLOTTING ONLY ESS WHERE WE HAD POSITIVE ACCEPTANCE PROBABILITY."""
    cc_copy = cc.copy()
    ap_copy = ap.copy()
    flag = ap_copy[:, ix] < 1e-8
    values = cc_copy[:, ix]
    values[flag] = nan
    return values


def show_only_positive_ap_crwm(out_cc, out_ap, ϵs, ix):
    """Same as above but for C-RWM."""
    cc_copy = deepcopy(out_cc.copy())
    ap_copy = deepcopy(out_ap.copy())
    flag = ap_copy < 1e-8
    cc_copy[flag] = nan
    return repeat(cc_copy[ix], len(ϵs))


if __name__== "__main__":
    # Global settings
    N_CHAINS = 4
    SEED_DATA_GENERATION = 1111                      # Used to generate y*
    SEEDS_FOR_CHAINS     = [1122, 2233, 3344, 4455] #[6666, 7777, 8888, 9999] #[2222, 3333, 4444, 5555]  # Each seed, used to find starting point of initial chain.
    Z_TRUE   = (0.4, 0.005, 0.05, 0.001)
    R0 = 100
    F0 = 100
    σR = 1
    σF = 1
    EPSILONS = generate_powers_of_ten(0, -6)   # [1.0, 0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001]
    BS = [1, 10, 20]
    STEP_SIZE = 0.01
    DISCRETIZATION_STEP_SIZE = 1.0
    SAFE_JACOBIAN = False
    assert len(SEEDS_FOR_CHAINS) == N_CHAINS, "For reproducibility, you need to choose `N_CHAINS` random seeds."

    # Settings for Ns=100
    SETTINGS100 = generate_settings(
    N=200, 
    δ=STEP_SIZE, 
    Ns=100, 
    Bs=BS, 
    ϵs=EPSILONS, 
    u1_true=False,
    tol=1e-14,
    n_chains=N_CHAINS,
    seeds=SEEDS_FOR_CHAINS
    )

#     # Settings for Ns=120
#     SETTINGS120 = generate_settings(
#     N=200, 
#     δ=STEP_SIZE, 
#     Ns=120, 
#     Bs=BS, 
#     ϵs=EPSILONS, 
#     u1_true=False,
#     tol=1e-14,
#     n_chains=N_CHAINS,
#     seeds=SEEDS_FOR_CHAINS
# )

    # Ns = 100
    # THUG00_CC_100, THUG00_AP_100 = cc_experiment_thug(SETTINGS100, 0.0, verbose=False, safe=SAFE_JACOBIAN)
    # THUG09_CC_100, THUG09_AP_100 = cc_experiment_thug(SETTINGS100, 0.9, verbose=False, safe=SAFE_JACOBIAN)
    # THUG99_CC_100, THUG99_AP_100 = cc_experiment_thug(SETTINGS100, 0.99, verbose=False, safe=SAFE_JACOBIAN)
    # CRWM_CC_100, CRWM_AP_100     = cc_experiment_crwm(SETTINGS100, tol=1e-11, verbose=False)
    # Ns = 120
    # THUG00_CC_120, THUG00_AP_120 = cc_experiment_thug(SETTINGS120, 0.0, verbose=False, safe=SAFE_JACOBIAN)
    # THUG09_CC_120, THUG09_AP_120 = cc_experiment_thug(SETTINGS120, 0.9, verbose=False, safe=SAFE_JACOBIAN)
    # THUG99_CC_120, THUG99_AP_120 = cc_experiment_thug(SETTINGS120, 0.99, verbose=False, safe=SAFE_JACOBIAN)
    # CRWM_CC_120, CRWM_AP_120     = cc_experiment_crwm(SETTINGS120, tol=1e-11, verbose=False)



    # Construct folder to save data
    mainfolder = "LV_Experiment"
    subfolder = "_".join([str(seed) for seed in SEEDS_FOR_CHAINS])
    folder = os.path.join(mainfolder, subfolder)
    if not os.path.exists(folder):
        os.makedirs(folder)

    # Save data for Ns=100
    # save(os.path.join(folder, 'THUG00_CC_100.npy'), THUG00_CC_100)
    # save(os.path.join(folder, 'THUG00_AP_100.npy'), THUG00_AP_100)
    # save(os.path.join(folder, 'THUG09_CC_100.npy'), THUG09_CC_100)
    # save(os.path.join(folder, 'THUG09_AP_100.npy'), THUG09_AP_100)
    # save(os.path.join(folder, 'THUG99_CC_100.npy'), THUG99_CC_100)
    # save(os.path.join(folder, 'THUG99_AP_100.npy'), THUG99_AP_100)
    # save(os.path.join(folder, 'CRWM_CC_100.npy'), CRWM_CC_100)
    # save(os.path.join(folder, 'CRWM_AP_100.npy'), CRWM_AP_100)

    # Save data for Ns=120
    # save(os.path.join(folder, 'THUG00_CC_120.npy'), THUG00_CC_120)
    # save(os.path.join(folder, 'THUG00_AP_120.npy'), THUG00_AP_120)
    # save(os.path.join(folder, 'THUG09_CC_120.npy'), THUG09_CC_120)
    # save(os.path.join(folder, 'THUG09_AP_120.npy'), THUG09_AP_120)
    # save(os.path.join(folder, 'THUG99_CC_120.npy'), THUG99_CC_120)
    # save(os.path.join(folder, 'THUG99_AP_120.npy'), THUG99_AP_120)
    # save(os.path.join(folder, 'CRWM_CC_120.npy'), CRWM_CC_120)
    # save(os.path.join(folder, 'CRWM_AP_120.npy'), CRWM_AP_120)

    # Save Epsilons
    # save(os.path.join(folder, 'EPSILONS.npy'), EPSILONS)
    # save(os.path.join(folder, 'BS.npy'), BS)
