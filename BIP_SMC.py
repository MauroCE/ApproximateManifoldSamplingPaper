"""
Code to reproduce the Sequential Monte Carlo (SMC) version of the Bayesian Inverse Problem (BIP) experiment. 
"""
import numpy as np
from numpy import arange, zeros, concatenate, hstack, unique, mean, array, save
from numpy import quantile, eye, log, exp, errstate
from numpy.random import randint, default_rng
from numpy.linalg import norm
from scipy.stats import multivariate_normal as MVN
from time import time
from warnings import catch_warnings, filterwarnings
import os

from RWM import RWM
from TangentialHug import THUG


class SMCTHUG:
    def __init__(self, N, d, ystar, logprior, ϵmin=None, pmin=0.2, pter=0.01, tolscheme='unique', η=0.9, mcmc_iter=5, propPmoved=0.99, δ0=0.2, a_star=0.3, B=5, manual_initialization=False, maxiter=300, thug=True, fixed_alpha=False, αmax=0.999, αmin=0.01, verbose=True, seed=None, thug_method='linear'):
        """SMC sampler using either a THUG or RWM kernel.

        Arguments:

        :param N: Number of particles.
        :type N: int

        :param d: Dimensionality of each particle. This is the dimensionality of the ambient space.
        :type d: int

        :param ystar: Observed data in BIP.
        :type ystar: 

        :param logprior: Evaluates the log-prior density.
        :type logprior: callable

        :param epsilonmin: Minimum tolerance. When a smaller tolerance is reached, we stop the sampler.
        :type epsilonmin: float

        :param pmin: Minimum acceptance probability we aim for. This is used to tune the step size.
        :type pmin: float

        :param pter: Terminal acceptance probability. When we go below this, we stop the sampler.
        :type pter: float

        :param tolscheme: Determines how the next tolerance epsilon is chosen. Can be either `'unique'` or `'ess'`.
        :type tolscheme: str

        :param eta: Quantile to use when determining a tolerance.
        :type eta: float

        :param mcmc_iter: Initial number of MCMC iterations for each particle at each step.
        :type mcmc_iter: int

        :param propPmoved:
        :type propPmoved: float

        :param delta0: Initial step size used by the kernel (either THUG or RWM).
        :type delta0: float

        :param a_star: 
        :type a_star: float

        :param B: Number of bounces in THUG.
        :type B: int

        :param manual_initialization: Boolean flag. If `True`, then the user can set `self.initialize_particles` to 
                                      a costume function. If `False`, then the particles are initialized from the prior.
        :type manual_initialization: bool

        :param maxiter: Maximum number of SMC iterations before stopping the sampler. Used in `self.stopping_criterion`.
        :type maxiter: int

        :param thug: Boolean flag. If `True` we use the THUG kernel, otherwise RWM.
        :type thug: bool

        :param fixed_alpha: If `True` then we set `alpha=0.0` across all iterations. Otherwise, we choose
                            the value of the squeezing parameter adaptively.
        :type fixed_alpha: bool

        :param alphamax:
        :type alphamax: 

        :param alphamin:
        :type alphamin: 

        :param verbose:
        :type verbose:

        :param seed:
        :type seed: int

        :param thug_method: 
        :type thug_method: 
        """
        # Store variables
        self.d = d                      # Dimensionality of each particle
        self.ystar = ystar              # Observed data
        self.ϵmin = ϵmin                # Lowest tolerance (for stopping)
        self.pmin = pmin                # Minimum acceptance prob (for stopping)
        self.pter = pter
        self.t = 0                      # Initialize iteration to 0
        self.η = η                      # Quantile for ϵ scheme
        self.a_star = a_star            # Target acceptance probability
        self.pPmoved = propPmoved       # Proportion of particles moved
        self.α = 0.0 if (fixed_alpha or not thug) else 0.01  # Initial squeezing parameter
        self.B = B
        self.q = MVN(zeros(self.d), eye(self.d))
        self.mcmc_iter = mcmc_iter
        self.N = N
        self.manual_initialization = manual_initialization
        self.maxiter = maxiter
        self.total_time = 0.0
        self.thug = thug
        self.fixed_alpha = fixed_alpha
        self.αmax = αmax
        self.αmin = αmin
        self.δ0 = δ0
        self.thug_method = thug_method
        self.verboseprint = print if verbose else lambda *a, **k: None
        
        # Set seed for reproducibility
        self.seed = randint(low=1000, high=9999) if seed is None else seed
        self.rng  = default_rng(seed=self.seed)

        # Initialize arrays
        self.W       = zeros((N, 1))           # Normalized weights
        self.D       = zeros((N, 1))           # Distances
        self.A       = zeros((N, 1), dtype=int)           # Ancestors
        self.P       = zeros((N, self.d, 1))   # Particles
        self.EPSILON = [np.inf]                # ϵ for all iterations
        self.ESS     = [0.0]                   # ESS
        self.n_unique_particles = [0.0]
        self.n_unique_starting = []
        self.avg_acc_prob_within_MCMC = zeros((N, 1)) # (n, t) entry is the average acceptance probability of self.MCMC[self.t] iterations stpes of MCMC
        self.accprob = [1.0]                   # Current acceptance probability
        self.step_sizes = [δ0]                 # Tracks step sizes
        self.ALPHAS = [self.α]                 # Tracks the α for THUG

        # Store log prior
        self.logprior = logprior

        if not thug and fixed_alpha:
            raise ValueError("`fixed_alpha` can only be set to True if `thug` is also set to True.")

        # Set stopping criterion or raise an error
        if (ϵmin is None) or (pter is None):
            raise NotImplementedError("Arguments ϵmin and pter mustn't be None.")
        else:
            self.stopping_criterion = self.min_tolerance_and_acc_prob
            self.verboseprint("### Stopping Criterion: Minimum Tolerance {} and Terminal Acceptance Probability {}.".format(ϵmin, pter))

        # Set tolerance scheme
        if tolscheme == 'unique':
            self.tol_scheme = self.unique_tol_scheme
        elif tolscheme == 'ess':
            self.tol_scheme = self.ess_tol_scheme
        else:
            raise NotImplementedError("Tolerance schemes: unique or ess.")

        # Set THUG kernel
        wrapMCMCoutput = lambda samples, acceptances: (samples[-1, :], mean(acceptances))
        if thug:
            if not fixed_alpha:   ##### THUG with adaptive alpha
                print("### MCMC kernel: THUG with adaptive alpha.")
                self.MCMCkernel = lambda *args: wrapMCMCoutput(*(THUG(*args)))
                self.MCMC_args  = lambda x0, N: (x0, self.B*self.step_sizes[-1], self.B, N, self.α, self.logpi, self.grad_h, self.thug_method, self.rng)
            else:   #### THUG with fixed alpha
                print("### MCMC kernel: THUG with fixed alpha.")
                self.MCMCkernel = lambda *args: wrapMCMCoutput(*(THUG(*args)))
                self.MCMC_args  = lambda x0, N: (x0, self.B*self.step_sizes[-1], self.B, N, 0.0, self.logpi, self.grad_h, self.thug_method, self.rng)

        # Or Random Walk
        else:
            print("### MCMC kernel: isotropic RWM.")
            self.MCMCkernel = lambda *args: wrapMCMCoutput(*RWM(*args))
            self.MCMC_args  = lambda x0, N: (x0, self.B*self.step_sizes[-1], N, self.logpi, self.rng)

        ### Finally, if using HUG or RWM simply remove the α update
        if (thug and fixed_alpha) or not thug:
            self.update_α = lambda a_hat, i: None

    @staticmethod
    def sample_prior(rng=None):
        """Samples xi = (theta, z) from prior distribution. Notice that the function must take as input
        a random number generator."""
        raise NotImplementedError

    def min_tolerance_and_acc_prob(self): return (self.EPSILON[-1] > self.ϵmin) and (self.accprob[-1] > self.pter) and (self.t < self.maxiter)

    def unique_tol_scheme(self): return max(self.ϵmin, quantile(unique(self.D[self.A[:, -1], -1]), self.η))
    def ess_tol_scheme(self):    return max(self.ϵmin, quantile(self.D[self.A[:, -1], -1], self.η))

    @staticmethod
    def h(ξ, ystar):
        """Computes ||f(xi) - y*||"""
        raise NotImplementedError

    @staticmethod
    def h_broadcast(ξ_matrix, ystar):
        """Computes ||f_broadcast(xi) - y*||"""
        raise NotImplementedError

    def logkernel(self, ξ):
        """Kernel used for logpi. Epanechnikov in this case."""
        u = self.h(ξ, self.ystar)
        ϵ = self.EPSILON[self.t]
        with errstate(divide='ignore'):
            return log((3*(1 - (u**2 / (ϵ**2))) / (4*ϵ)) * float(u <= ϵ))

    def logpi(self, ξ):
        """Target distribution."""
        return self.logprior(ξ) + self.logkernel(ξ)

    @staticmethod
    def grad_h(ξ):
        """Computes the gradient of h(xi). Used by HUG/THUG."""
        raise NotImplementedError

    def compute_distances(self, flag=None):
        """Computes distance between all particles and ystar. If `flag` is
        provided, then it only computes the distance of the particles
        whose flag is True."""
        if flag is None:
            return self.h_broadcast(self.P[:, :, -1], self.ystar)
        else:
            return self.h_broadcast(self.P[flag, :, -1], self.ystar)

    def compute_distance(self, ix):
        """Computes distance between ix particle and ystar."""
        return self.h(self.P[ix, :, -1], self.ystar)

    @staticmethod
    def get_γ(i):
        """User needs to set this method. Returns the step size for the α update."""
        raise NotImplementedError

    def update_α(self, a_hat, i):
        """Updates α based on current acceptance probability"""
        τ = log(self.α / (1 - self.α))
        γ = self.get_γ(i)
        τ = τ - γ*(a_hat - self.a_star)
        self.α = np.clip(1 / (1 + exp(-τ)), self.αmin, self.αmax)

    def resample(self):
        """Resamples indeces of particles"""
        return self.rng.choice(a=arange(self.N), size=self.N, replace=True, p=self.W[:, -1])

    @staticmethod
    def initialize_particles(N, rng=None):
        """Can be used to initialize particles in a different way. Must also take a RNG."""
        raise NotImplementedError("If manual_initialization=True then you must provide initialize_particles.")

    def sample(self):
        initial_time = time()

        ### INITIALIZE PARTICLES (EITHER FROM PRIOR OR MANUALLY)
        if self.manual_initialization:   # Custom function provided by user 
            particles = self.initialize_particles(self.N, self.rng)
            for i in range(self.N):
                self.P[i, :, 0] = particles[i, :]
                self.W[i, 0]    = 1 / self.N
            self.verboseprint("### Particles have been initialized manually.")
        else:  # From prior
            for i in range(self.N):
                self.P[i, :, 0] = self.sample_prior(rng=self.rng)  # Sample particles from prior
                self.W[i, 0]    = 1 / self.N           # Assign uniform weights
            self.verboseprint("### Particles have been initialized from the prior.")

        ### COMPUTE DISTANCES. USE LARGEST DISTANCE AS CURRENT EPSILON
        self.D[:, 0]               = self.compute_distances()     # Compute distances
        self.EPSILON[0]            = np.max(self.D[:, 0])         # Reset ϵ0 to max distance
        self.ESS[0]                = 1 / (self.W[:, 0]**2).sum()  
        self.n_unique_particles[0] = len(unique(self.D[:, 0]))
        self.verboseprint("### Starting with {} unique particles.".format(self.n_unique_particles[0]))

        # RUN ALGORITHM UNTIL STOPPING CRITERION IS MET
        while self.stopping_criterion():

            # RESAMPLING
            self.A[:, self.t] = self.resample()
            self.t += 1

            # SELECT TOLERANCE
            self.EPSILON.append(self.tol_scheme())

            # ADD ZERO-COLUMN TO ALL MATRICES FOR STORAGE OF THIS ITERATION
            self.A = hstack((self.A, zeros((self.N, 1), dtype=int)))
            self.D = hstack((self.D, zeros((self.N, 1))))
            self.W = hstack((self.W, zeros((self.N, 1))))
            self.P = concatenate((self.P, zeros((self.N, self.d, 1))), axis=2)
            self.avg_acc_prob_within_MCMC = hstack((self.avg_acc_prob_within_MCMC, zeros((self.N, 1))))

            # COMPUTE WEIGHTS
            self.W[:, -1] = self.D[self.A[:, -2], -2] < self.EPSILON[-1]
            with catch_warnings():
                filterwarnings('error')
                try:
                    self.W[:, -1] = self.W[:, -1] / self.W[:, -1].sum()  # Normalize
                except RuntimeWarning:
                    print("There's some issue with the weights. Exiting.")
                    return {
                        'P': self.P,
                        'W': self.W,
                        'A': self.A,
                        'D': self.D,
                        'EPSILON': self.EPSILON,
                        'AP': self.accprob,
                        'STEP_SIZES': self.step_sizes[:-1],
                        'ESS': self.ESS,
                        'UNIQUE_PARTICLES': self.n_unique_particles,
                        'UNIQUE_STARTING': self.n_unique_starting,
                        'ALPHAS': self.ALPHAS,
                        'TIME': self.total_time,
                        'SEED': self.seed
                    }

            # COMPUTE ESS
            self.ESS.append(1 / (self.W[:, -1]**2).sum())

            self.verboseprint("\n### SMC step: ", self.t)
            self.n_unique_starting.append(len(unique(self.D[self.A[:, -2], -2])))  # Unique after resampling
            self.verboseprint("ϵ = {:.10f}\t N unique starting: {}".format(round(self.EPSILON[-1], 5), self.n_unique_starting[-1]))

            # METROPOLIS-HASTINGS - MOVE ALIVE PARTICLES
            self.verboseprint("Metropolis-Hastings steps: ", self.mcmc_iter)
            alive = self.W[:, -1] > 0.0     # Boolean flag for alive particles
            index = np.where(alive)[0]      # Indices for alive particles
            for ix in index:
                self.P[ix, :, -1], self.avg_acc_prob_within_MCMC[ix, -1] = self.MCMCkernel(*self.MCMC_args(self.P[self.A[ix, -2], :, -2], self.mcmc_iter))
                self.D[ix, -1] = self.compute_distance(ix)
            self.n_unique_particles.append(len(unique(self.D[alive, -1])))

            # ESTIMATE ACCEPTANCE PROBABILITY
            self.accprob.append(self.avg_acc_prob_within_MCMC[:, -1].mean())
            self.verboseprint("Average Acceptance Probability: {:.4f}".format(self.accprob[-1]))

            # DO NOT TUNE STEP SIZE
            self.step_sizes.append(self.δ0)
            self.verboseprint("Stepsize used in next SMC iteration: {:.4f}".format(self.step_sizes[-1]))

            # TUNE SQUEEZING PARAMETER FOR THUG
            self.update_α(self.accprob[-1], self.t)
            self.ALPHAS.append(self.α)
            self.verboseprint("Alpha used in next SMC iteration: {:.4f}".format(self.α))

            if self.EPSILON[-1] == self.ϵmin:
                print("Latest ϵ == ϵmin. Breaking")
                break

        self.total_time = time() - initial_time

        return {
            'P': self.P,
            'W': self.W,
            'A': self.A,
            'D': self.D,
            'EPSILON': self.EPSILON,
            'AP': self.accprob,
            'STEP_SIZES': self.step_sizes[:-1],
            'ESS': self.ESS,
            'UNIQUE_PARTICLES': self.n_unique_particles,
            'UNIQUE_STARTING': self.n_unique_starting,
            'ALPHAS': self.ALPHAS,
            'TIME': self.total_time,
            'SEED': self.seed
        }


if __name__ == "__main__":
    
    # Settings
    N = 5000      # Number of particles
    B = 5         # Number of bounces & number of mcmc steps
    σ = 1e-8      # Target noise scale
    d = 3         # Dimensionality ambient space
    y = 1         # Observed data
    ϵmin = 1e-10  # Minimum allowable tolerance
    pmin = 0.3    
    pter = 0.01
    maxiter = 200
    αmax = 0.9999
    αmin = 0.01
    η = 0.9
    astar = 0.3
    pPm = 0.99
    seed = 1234
    VERBOSE = False

    # Functions
    logprior = lambda ξ: MVN(zeros(3), eye(3)).logpdf(ξ)          # Log-prior for approximate lifted distribution
    F   = lambda θ: array([θ[1]**2 + 3 * θ[0]**2 * (θ[0]**2 - 1)]) # Forward function
    FL  = lambda ξ: F(ξ[:2])[0] + σ*ξ[-1]                          # Constraint function for lifted problem
    FLb = lambda ξ, : ξ[:, 1]**2 + (3*ξ[:, 0]**2)*(ξ[:, 0]**2 - 1) + σ*ξ[:, 2]  # Broadcasted version of FL
    grad_FL = lambda ξ: array([12*ξ[0]**3 - 6*ξ[0], 2*ξ[1], σ])

    def sample_prior(rng=None):
        if rng is None:
            rng = default_rng(seed=randint(low=1000, high=9999))
        return rng.normal(loc=0.0, scale=1.0, size=3)

    # THUG with adaptive alpha
    αTHUG_001 = SMCTHUG(N=N, d=d, ystar=y,logprior=logprior,
                        ϵmin=ϵmin, pmin=pmin, pter=pter, tolscheme='unique', η=η, mcmc_iter=B, 
                        propPmoved=pPm, δ0=0.01, a_star=astar, B=B, manual_initialization=False, 
                        maxiter=maxiter, thug=True, fixed_alpha=False, αmax=αmax, αmin=αmin, 
                        verbose=VERBOSE, seed=seed, thug_method='2d')

    # THUG with fixed alpha=0.0
    THUG_001  = SMCTHUG(N=N, d=d, ystar=y,logprior=logprior,
                        ϵmin=ϵmin, pmin=pmin, pter=pter, tolscheme='unique', η=η, mcmc_iter=B, 
                        propPmoved=pPm, δ0=0.01, a_star=astar, B=B, manual_initialization=False, 
                        maxiter=maxiter, thug=True, fixed_alpha=True, αmax=αmax, αmin=αmin, 
                        verbose=VERBOSE, seed=seed, thug_method='2d')

    # RWM
    RWM_001   = SMCTHUG(N=N, d=d, ystar=y,logprior=logprior,
                        ϵmin=ϵmin, pmin=pmin, pter=pter, tolscheme='unique', η=η, mcmc_iter=B, 
                        propPmoved=pPm, δ0=0.01, a_star=astar, B=B, manual_initialization=False, 
                        maxiter=maxiter, thug=False, fixed_alpha=False, αmax=αmax, αmin=αmin, 
                        verbose=VERBOSE, seed=seed, thug_method='2d')

    # For all three samplers set the same functions
    SMC_SAMPLERS_001 = [αTHUG_001, THUG_001, RWM_001]
    for SMC in SMC_SAMPLERS_001:
        SMC.h            = lambda ξ, ystar: norm(FL(ξ) - ystar)
        SMC.h_broadcast  = lambda ξ, ystar: abs(FLb(ξ) - ystar)
        SMC.grad_h       = lambda ξ: grad_FL(ξ) * (FL(ξ) - y)
        SMC.sample_prior = lambda rng: sample_prior(rng)
        SMC.get_γ        = lambda i: 1.0

    # THUG with adaptive alpha
    αTHUG_01 = SMCTHUG(N=N, d=d, ystar=y,logprior=logprior,
                    ϵmin=ϵmin, pmin=pmin, pter=pter, tolscheme='unique', η=η, mcmc_iter=B, 
                    propPmoved=pPm, δ0=0.1, a_star=astar, B=B, manual_initialization=False, 
                    maxiter=maxiter, thug=True, fixed_alpha=False, αmax=αmax, αmin=αmin, 
                    verbose=VERBOSE, seed=seed, thug_method='2d')
    # THUG with fixed alpha=0.0
    THUG_01 = SMCTHUG(N=N, d=d, ystar=y,logprior=logprior,
                        ϵmin=ϵmin, pmin=pmin, pter=pter, tolscheme='unique', η=η, mcmc_iter=B, 
                        propPmoved=pPm, δ0=0.1, a_star=astar, B=B, manual_initialization=False, 
                        maxiter=maxiter, thug=True, fixed_alpha=True, αmax=αmax, αmin=αmin, 
                        verbose=VERBOSE, seed=seed, thug_method='2d')
    # RWM
    RWM_01   = SMCTHUG(N=N, d=d, ystar=y,logprior=logprior,
                        ϵmin=ϵmin, pmin=pmin, pter=pter, tolscheme='unique', η=η, mcmc_iter=B, 
                        propPmoved=pPm, δ0=0.1, a_star=astar, B=B, manual_initialization=False, 
                        maxiter=maxiter, thug=False, fixed_alpha=False, αmax=αmax, αmin=αmin, 
                        verbose=VERBOSE, seed=seed, thug_method='2d')

    SMC_SAMPLERS_01 = [αTHUG_01, THUG_01, RWM_01]
    for SMC in SMC_SAMPLERS_01:
        SMC.h            = lambda ξ, ystar: norm(FL(ξ) - ystar)
        SMC.h_broadcast  = lambda ξ, ystar: abs(FLb(ξ) - ystar)
        SMC.grad_h       = lambda ξ: grad_FL(ξ) * (FL(ξ) - y)
        SMC.sample_prior = lambda rng: sample_prior(rng)
        SMC.get_γ        = lambda i: 1.0

    # THUG with adaptive alpha
    αTHUG_05 = SMCTHUG(N=N, d=d, ystar=y,logprior=logprior,
                    ϵmin=ϵmin, pmin=pmin, pter=pter, tolscheme='unique', η=η, mcmc_iter=B, 
                    propPmoved=pPm, δ0=0.5, a_star=astar, B=B, manual_initialization=False, 
                    maxiter=maxiter, thug=True, fixed_alpha=False, αmax=αmax, αmin=αmin, 
                    verbose=VERBOSE, seed=seed, thug_method='2d')
    # THUG with fixed alpha=0.0
    THUG_05 = SMCTHUG(N=N, d=d, ystar=y,logprior=logprior,
                        ϵmin=ϵmin, pmin=pmin, pter=pter, tolscheme='unique', η=η, mcmc_iter=B, 
                        propPmoved=pPm, δ0=0.5, a_star=astar, B=B, manual_initialization=False, 
                        maxiter=maxiter, thug=True, fixed_alpha=True, αmax=αmax, αmin=αmin, 
                        verbose=VERBOSE, seed=seed, thug_method='2d')
    # RWM
    RWM_05   = SMCTHUG(N=N, d=d, ystar=y,logprior=logprior,
                        ϵmin=ϵmin, pmin=pmin, pter=pter, tolscheme='unique', η=η, mcmc_iter=B, 
                        propPmoved=pPm, δ0=0.5, a_star=astar, B=B, manual_initialization=False, 
                        maxiter=maxiter, thug=False, fixed_alpha=False, αmax=αmax, αmin=αmin, 
                        verbose=VERBOSE, seed=seed, thug_method='2d')

    SMC_SAMPLERS_05 = [αTHUG_05, THUG_05, RWM_05]
    for SMC in SMC_SAMPLERS_05:
        SMC.h            = lambda ξ, ystar: norm(FL(ξ) - ystar)
        SMC.h_broadcast  = lambda ξ, ystar: abs(FLb(ξ) - ystar)
        SMC.grad_h       = lambda ξ: grad_FL(ξ) * (FL(ξ) - y)
        SMC.sample_prior = lambda rng: sample_prior(rng)
        SMC.get_γ        = lambda i: 1.0

    # THUG with adaptive alpha
    αTHUG_1 = SMCTHUG(N=N, d=d, ystar=y,logprior=logprior,
                    ϵmin=ϵmin, pmin=pmin, pter=pter, tolscheme='unique', η=η, mcmc_iter=B, 
                    propPmoved=pPm, δ0=1.0, a_star=astar, B=B, manual_initialization=False, 
                    maxiter=maxiter, thug=True, fixed_alpha=False, αmax=αmax, αmin=αmin, 
                    verbose=VERBOSE, seed=seed, thug_method='2d')
    # THUG with fixed alpha=0.0
    THUG_1 = SMCTHUG(N=N, d=d, ystar=y,logprior=logprior,
                        ϵmin=ϵmin, pmin=pmin, pter=pter, tolscheme='unique', η=η, mcmc_iter=B, 
                        propPmoved=pPm, δ0=1, a_star=astar, B=B, manual_initialization=False, 
                        maxiter=maxiter, thug=True, fixed_alpha=True, αmax=αmax, αmin=αmin, 
                        verbose=VERBOSE, seed=seed, thug_method='2d')
    # RWM
    RWM_1   = SMCTHUG(N=N, d=d, ystar=y,logprior=logprior,
                        ϵmin=ϵmin, pmin=pmin, pter=pter, tolscheme='unique', η=η, mcmc_iter=B, 
                        propPmoved=pPm, δ0=1, a_star=astar, B=B, manual_initialization=False, 
                        maxiter=maxiter, thug=False, fixed_alpha=False, αmax=αmax, αmin=αmin, 
                        verbose=VERBOSE, seed=seed, thug_method='2d')

    SMC_SAMPLERS_1 = [αTHUG_1, THUG_1, RWM_1]
    for SMC in SMC_SAMPLERS_1:
        SMC.h            = lambda ξ, ystar: norm(FL(ξ) - ystar)
        SMC.h_broadcast  = lambda ξ, ystar: abs(FLb(ξ) - ystar)
        SMC.grad_h       = lambda ξ: grad_FL(ξ) * (FL(ξ) - y)
        SMC.sample_prior = lambda rng: sample_prior(rng)
        SMC.get_γ        = lambda i: 1.0

    # Sample
    OUTPUTS_001 = [SMC.sample() for SMC in SMC_SAMPLERS_001]
    OUTPUTS_01 = [SMC.sample() for SMC in SMC_SAMPLERS_01]
    OUTPUTS_05 = [SMC.sample() for SMC in SMC_SAMPLERS_05]
    OUTPUTS_1 = [SMC.sample() for SMC in SMC_SAMPLERS_1]


    # Save data
    folder = 'BIP_Experiment/SMC'
    save(os.path.join(folder, 'SMC_DICT_ATHUG_001.npy'), OUTPUTS_001[0])
    save(os.path.join(folder, 'SMC_DICT_THUG_001.npy'), OUTPUTS_001[1])
    save(os.path.join(folder, 'SMC_DICT_RWM_001.npy'), OUTPUTS_001[2])
    save(os.path.join(folder, 'SMC_DICT_ATHUG_05.npy'), OUTPUTS_05[0])
    save(os.path.join(folder, 'SMC_DICT_THUG_05.npy'), OUTPUTS_05[1])
    save(os.path.join(folder, 'SMC_DICT_RWM_05.npy'), OUTPUTS_05[2])
    save(os.path.join(folder, 'SMC_DICT_ATHUG_1.npy'), OUTPUTS_1[0])
    save(os.path.join(folder, 'SMC_DICT_THUG_1.npy'), OUTPUTS_1[1])
    save(os.path.join(folder, 'SMC_DICT_RWM_1.npy'), OUTPUTS_1[2])
