"""
Same as smc_thug_fixed_stepsize.py but we start off with RWM as the mutation kernel,
and once \epsilon is small enough (i.e. RWM fails), we switch to adaptive THUG.
"""
import numpy as np
from numpy import arange, ones, array, zeros, concatenate, hstack, unique, mean, save
from numpy import quantile, cov, eye, log, ceil, exp, clip, errstate, vstack
from numpy import array_equal
from numpy.linalg import cholesky, norm
from numpy.random import choice, uniform, default_rng, randint
from scipy.stats import multivariate_normal as MVN
from time import time  
import os


from RWM import RWM
from TangentialHug import THUG
from warnings import catch_warnings, filterwarnings


class SMCTHUG:
    def __init__(self, N, d, ystar, logprior, ϵmin=None, pmin=0.2, pter=0.01, tolscheme='unique', η=0.9, mcmc_iter=5, propPmoved=0.99, δ0=0.2, minstep=0.1, maxstep=100.0, a_star=0.3, B=5, manual_initialization=False, maxiter=300, fixed_alpha=False, αmax=0.999, αmin=0.01, pter_multiplier=1.1, verbose=False, seed=None, thug_method='2d'):
        """SMC sampler starting with a RWM kernel and then switch to THUG (with or without adaptive step size)
        once RWM starts failing.

        Parameters:

        :param N: Number of particles.
        :type N: int

        :param d: Dimensionality of each particle.
        :type d: int

        :param ystar: Observed data in BIP problem.
        :type ystar: ndarray

        :param logprior: Evaluates log prior density at ξ.
        :type logprior: callable

        :param epsilonmin: Minimum tolerance. If we reach a smaller tolerance, we stop. 
        :type epsilonmin: float

        :param pmin: Acceptance probability we aim for. Used to tune the step size.
        :type pmin: float

        :param pter: Terminal acceptance probability. If we go below this, then we stop.
        :type pter: float

        :param tolscheme: Either `'unique'` or `'ess'`. Determines how the next ϵ is chosen.
        :type tolscheme: str

        :param eta: Quantile to use when determining a tolerance.
        :type eta: float

        :param mcmc_iter: Number of MCMC iterations for each particle, at each step. 
        :type mcmc_iter: int

        :param delta0: Initial step size.
        :type delta0: float

        :param minstep: Minimum step size for adaptive step size finding.
        :type minstep: float
        
        :param maxstep: Maximum step size for adaptive step size finding.
        :type maxstep: float

        :param B: Number of bounces in THUG. During the RWM phase, the stepsize is B*delta.
        :type B: int

        :param manual_initialization: If `True` then user can set `self.initialize_particles`
                                      to a custom function instead of initializing from the prior.
        :type manual_initialization: bool

        :param maxiter: Maximum number of SMC iterations. Used in `self.stopping_criterion`.
        :type maxiter: int

        :param fixed_alpha: If true, then we use thug with alpha=0.0. That is we don't use thug with adaptive alpha.
        :type fixed_alpha: bool

        Returns: 
        
        :param output_dict: Dictionary containing various data from the sampler.
        :type output_dict: dict
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
        self.αmin = αmin
        self.α = 0.0 if fixed_alpha else self.αmin  # Initial squeezing parameter
        self.B = B
        self.q = MVN(zeros(self.d), eye(self.d))
        self.mcmc_iter = mcmc_iter
        self.N = N
        self.minstep = minstep
        self.maxstep = maxstep
        self.manual_initialization = manual_initialization
        self.maxiter = maxiter
        self.total_time = 0.0
        self.fixed_alpha = fixed_alpha
        self.δ0 = δ0   # NEW LINE
        self.αmax = αmax
        self.initial_rwm_has_failed = False
        self.pter_multiplier = pter_multiplier  # Used to figure out when THUG should kick in
        self.verboseprint = print if verbose else lambda *a, **k: None
        assert pter_multiplier >= 1.0, "pter multiplier must be larger than 1.0."
        self.thug_method = thug_method

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
        #self.accepted = zeros((N, 1))          # proportion of accepted thug steps within self.THUG
        self.accprob = [1.0]                   # Current acceptance probability
        self.step_sizes = [self.δ0]            # Tracks step sizes
        self.ALPHAS = [self.α]                 # Tracks the α for THUG

        # Store log prior
        self.logprior = logprior
        self.Σ = eye(self.d)
        self.Σfunc = lambda x: self.Σ

        # We only have one stopping criterion: ϵmin, pter and RWM/THUG
        if (ϵmin is None) or (pter is None):
            raise NotImplementedError("Arguments ϵmin and pter mustn't be None.")
        else:
            self.stopping_criterion = self.stopping_criterion_rwm
            self.verboseprint("### Stopping Criterion: Minimum Tolerance {} and Terminal Acceptance Probability {}".format(ϵmin, pter))

        # Set tolerance scheme
        if tolscheme == 'unique':
            self.tol_scheme = self.unique_tol_scheme
        elif tolscheme == 'ess':
            self.tol_scheme = self.ess_tol_scheme
        else:
            raise NotImplementedError("Tolerance schemes: unique or ess.")

        wrapMCMCoutput = lambda samples, acceptances: (samples[-1, :], mean(acceptances))
        # Set kernel to RWM (either preconditioned or isotropic)
        self.verboseprint("### MCMC kernel: isotropic RWM.")
        self.MCMCkernel = lambda *args: wrapMCMCoutput(*RWM(*args))
        self.MCMC_args  = lambda x0, N: (x0, self.B*self.step_sizes[-1], N, self.logpi, self.rng)
        # Set up a kernel containing the correct THUG kernel (Adaptive, non adaptive, preconditioned or not).
        # the MCMC kernel will be assigned to this when RWM fails.
        if not fixed_alpha:   ##### THUG (adaptive alpha)
            self.verboseprint("### THUG kernel (for later): THUG.")
            self.THUGkernel = lambda *args: wrapMCMCoutput(*(THUG(*args)))
            self.THUG_args  = lambda x0, N: (x0, self.B*self.step_sizes[-1], self.B, N, self.α, self.logpi, self.grad_h, self.thug_method, self.rng)
        else:   #### THUG (alpha=0.0 fixed)
            self.verboseprint("### THUG kernel (for later): HUG.")
            self.THUGkernel = lambda *args: wrapMCMCoutput(*(THUG(*args)))
            self.THUG_args  = lambda x0, N: (x0, self.B*self.step_sizes[-1], self.B, N, 0.0, self.logpi, self.grad_h, self.thug_method, self.rng)

        ### Finally, if using HUG or RWM simply remove the α update
        if fixed_alpha:
            self.update_α = lambda a_hat, i: None

    @staticmethod
    def sample_prior():
        """Samples xi = (theta, z) from prior distribution."""
        raise NotImplementedError

    # def min_tol_or_min_acc_prob(self): return (self.EPSILON[-1] > self.ϵmin) and (self.t < self.maxiter) and (self.accprob[-1] > self.pter)
    def stopping_criterion_rwm(self): return (self.t < self.maxiter) and (self.EPSILON[-1] > self.ϵmin) # Only check maxiter and epsilon
    def stopping_criterion_thug(self): return (self.t < self.maxiter) and (self.EPSILON[-1] > self.ϵmin) and (self.accprob[-1] > self.pter) # Additionally, check for pter

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
        self.α = np.clip(1 / (1 + exp(-τ)), self.αmin , self.αmax)

    def resample(self):
        """Resamples indeces of particles"""
        return choice(arange(self.N), size=self.N, replace=True, p=self.W[:, -1])

    @staticmethod
    def initialize_particles(N):
        """Can be used to initialize particles in a different way"""
        raise NotImplementedError("If manual_initialization=True then you must provide initialize_particles.")

    def sample(self):
        initial_time = time()
        # Initialize particles either manually
        if self.manual_initialization:
            particles = self.initialize_particles(self.N)
            for i in range(self.N):
                self.P[i, :, 0] = particles[i, :]
                self.W[i, 0]    = 1 / self.N
            self.verboseprint("### Particles have been initialized manually.")
        else:
            # or automatically from the prior
            for i in range(self.N):
                self.P[i, :, 0] = self.sample_prior(self.rng)  # Sample particles from prior
                self.W[i, 0]    = 1 / self.N           # Assign uniform weights
            self.verboseprint("### Particles have been initialized from the prior.")

        # Compute distances. Use largest distance as current ϵ
        self.D[:, 0]    = self.compute_distances() # Compute distances
        self.EPSILON[0] = np.max(self.D[:, 0])     # Reset ϵ0 to max distance
        self.ESS[0]     = 1 / (self.W[:, 0]**2).sum()
        self.n_unique_particles[0] = len(unique(self.D[:, 0]))
        self.verboseprint("### Starting with {} unique particles.".format(self.n_unique_particles[0]))

        # Run Algorithm until stopping criteria is met
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
            # Normalize weights (careful, some might be zeros or NaNs.)
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
                        'MCMC_ITER': self.mcmc_iter,
                        'STEP_SIZES': self.step_sizes[:-1],
                        'ESS': self.ESS,
                        'UNIQUE_PARTICLES': self.n_unique_particles,
                        'UNIQUE_STARTING': self.n_unique_starting,
                        'ALPHAS': self.ALPHAS,
                        'TIME': self.total_time
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

            # TUNE SQUEEZING PARAMETER FOR THUG ONLY IF RWM HAS ALREADY FAILED
            if self.initial_rwm_has_failed:
                self.update_α(self.accprob[-1], self.t)
                self.ALPHAS.append(self.α)
                print("Alpha used in next SMC iteration: {:.4f}".format(self.α))

            if self.EPSILON[-1] == self.ϵmin:
                print("Latest ϵ == ϵmin. Breaking")
                break

            # IF RWM IS STILL GOING, CHECK THE STOPPING CRITERION. IF IT FAILS
            # (I.E. RWM IS NOW TRYING TO SAMPLE FROM A FILAMENTARY DISTRIBUTION)
            # THEN CHANGE THE STOPPING CRITERION AND SWITCH TO USING THUG.
            if not self.initial_rwm_has_failed:
                # Check if pter has been reached. (actually, start it just before,
                # that's why it's multiplied by 1.1)
                if self.accprob[-1] <= (self.pter*self.pter_multiplier):
                    print("##############################################")
                    print("########### Initial RWM has failed ###########")
                    print("##############################################")
                    # RWM has failed
                    self.initial_rwm_has_failed = True
                    # Change stopping criterion
                    self.stopping_criterion = self.stopping_criterion_thug
                    # Change transition kernel to Thug
                    self.MCMCkernel = self.THUGkernel
                    self.MCMC_args  = self.THUG_args
                    # Store the iteration number at which we switch
                    self.t_at_which_we_switched_to_thug = self.t


        self.total_time = time() - initial_time

        return {
            'P': self.P,
            'W': self.W,
            'A': self.A,
            'D': self.D,
            'EPSILON': self.EPSILON,
            'AP': self.accprob,
            'MCMC_ITER': self.mcmc_iter,
            'STEP_SIZES': self.step_sizes[:-1],
            'ESS': self.ESS,
            'UNIQUE_PARTICLES': self.n_unique_particles,
            'UNIQUE_STARTING': self.n_unique_starting,
            'ALPHAS': self.ALPHAS,
            'TIME': self.total_time,
            'SWITCH_TO_THUG': self.t_at_which_we_switched_to_thug
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
    δ0   = 0.5
    maxiter = 200
    αmax = 0.9999
    αmin = 0.01
    η = 0.9
    astar = 0.3
    pPm = 0.99
    seed = 1234
    VERBOSE = True

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
    αTHUG_DICT = SMCTHUG(N=N, d=d, ystar=y,logprior=logprior,
                    ϵmin=ϵmin, pmin=pmin, pter=pter, tolscheme='unique', η=η, mcmc_iter=B, 
                    propPmoved=pPm, δ0=δ0, a_star=astar, B=B, manual_initialization=False, 
                    maxiter=maxiter, fixed_alpha=False, αmax=αmax, αmin=αmin, 
                    verbose=VERBOSE, seed=seed, thug_method='2d')

    # THUG with fixed alpha=0.0
    THUG_DICT  = SMCTHUG(N=N, d=d, ystar=y,logprior=logprior,
                    ϵmin=ϵmin, pmin=pmin, pter=pter, tolscheme='unique', η=η, mcmc_iter=B, 
                    propPmoved=pPm, δ0=δ0, a_star=astar, B=B, manual_initialization=False, 
                    maxiter=maxiter, fixed_alpha=True, αmax=αmax, αmin=αmin, 
                    verbose=VERBOSE, seed=seed, thug_method='2d')

    # For both samplers set the same functions
    SMC_SAMPLERS = [αTHUG_DICT, THUG_DICT]
    for SMC in SMC_SAMPLERS:
        SMC.h            = lambda ξ, ystar: norm(FL(ξ) - ystar)
        SMC.h_broadcast  = lambda ξ, ystar: abs(FLb(ξ) - ystar)
        SMC.grad_h       = lambda ξ: grad_FL(ξ) * (FL(ξ) - y)
        SMC.sample_prior = lambda rng: sample_prior(rng)
        SMC.get_γ        = lambda i: 1.0

    print(type(SMC_SAMPLERS[0]), SMC_SAMPLERS[0])
    print(type(SMC_SAMPLERS[1]), SMC_SAMPLERS[1])
    # Sample 
    OUTPUTS = [SMC.sample() for SMC in SMC_SAMPLERS]

    # Save data
    folder = 'BIP_Experiment/SMC_RWM_THEN_THUG'
    save(os.path.join(folder, 'SMC_DICT_RWM_THEN_ATHUG.npy'), OUTPUTS[0])
    save(os.path.join(folder, 'SMC_DICT_RWM_THEN_THUG.npy'), OUTPUTS[1])

