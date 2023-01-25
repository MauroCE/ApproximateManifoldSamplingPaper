"""
This file contains various classes relating to Manifolds.

    - Manifold: the base class, implements an abstract manifold. Every other class inherits from it.
    - Generalized Ellipse: level set of a MVN
    - LVManifold: manifold for the Lotka-Volterra problem.
    - GKManifold: manifold for the G-and-K problem.
"""
import math
import numpy as np
from numpy import log, pi, zeros, eye, ones, exp, sqrt, diag, apply_along_axis, concatenate, vstack, isfinite
from numpy.random import default_rng, randn
from numpy.linalg import svd, inv, det, norm
from scipy.optimize import fsolve
from scipy.stats import multivariate_normal as MVN
from scipy.stats import norm as ndist
from scipy.stats import uniform as udist
from scipy.special import ndtr, ndtri
from scipy.linalg import block_diag
from warnings import catch_warnings, filterwarnings


class Manifold:
    def __init__(self, m, d):
        """
        Generic Manifold Class. This class implements an abstract "manifold". It is a simple and compact way to 
        gether together several properties and functions related to a manifold. Importantly, this class only represents
        implicitly-defined manifold, i.e. given a smooth function f:Rn -> Rm with n > m, the manifold is defined as the 0-level
        set of f. There is nothing special about the value 0, but it is convenient: given another function g, if the level set of interest
        is {x \in Rn : g(x) = y} then we simply define f(x) = g(x) - y.

        m : Int
            Number of constraints & Co-dimension of the manifold. (e.g. 1 for Torus/Sphere in 3D)
        d : Int
            Dimension of the manifold. (e.g. 2 for Torus/Sphere in 3D)
        """
        self.m = m                # Co-Dimension of the Manifold
        self.d = d                # Dimension of the Manifold
        self.n = self.m + self.d  # Dimension of the Ambient Space

    def tangent_basis(self, Q):
        """
        Computes a tangent basis from the Q matrix (the transpose of the Jacobian matrix).

        Q : Numpy Array
            2D Numpy array of dimension (m + d, m) containing gradients of the constraints as columns.
        returns : Matrix containing basis of tangent space as its columns.
        """
        assert Q.shape == (self.m + self.d, self.m), "Q must have shape ({}, {}) but found shape {}".format(self.m+self.d, self.m, Q.shape)
        return svd(Q)[0][:, self.m:]

    def get_dimension(self):
        """
        Returns dimension of the manifold d
        """
        return self.d
    
    def get_codimension(self):
        """
        Returns co-dimension of the manifold d
        """
        return self.m


class GeneralizedEllipse(Manifold):
    def __init__(self, mu, Sigma, z):
        """
        Class implementing the ellipsoid corresponding to the `z`-level set of a 
        multivariate normal (MVN) with mean `mu` and covariance matrix `Sigma`.

        mu : Numpy Array
             Center of the sphere, and mean of the MVN. Must be a 1D array of dimension (3, )
        Sigma : Numpy Array
                Covariance matrix of the MVN, specifying the angle and scaling of the ellipse.
        z : float 
            Level set value. This identifies which level set / contour of the MVN to consider.
        """
        self.n = len(mu)    # Dimension of the ambient space
        # Store MVN parameters
        self.z = z
        self.mu = mu
        self.S = Sigma
        self.Sinv = inv(Sigma)
        self.mu = mu
        self.logdetS = log(det(self.S))
        self.MVN = MVN(self.mu, self.S)
        # Compute gamma (RHS) 
        self.gamma = -self.n * log(2*pi) - self.logdetS -2 * log(z)
        super().__init__(m=1, d=(self.n-1))

    def q(self, xyz):
        """Constraint function for the contour of MVN"""
        return (xyz - self.mu) @ (self.Sinv @ (xyz - self.mu)) - self.gamma

    def Q(self, xyz):
        """Q"""
        return (2 * self.Sinv @ (xyz - self.mu)).reshape(-1, self.m)

    def sample(self):
        """Samples from the contour by first sampling a point from the original
        MVN and then it rescales it until it is on the correct contour. This should
        work since the MVN is spherically symmetric."""
        start = self.MVN.rvs()   # Get initial MVN sample
        objective = lambda coef: self.MVN.pdf(coef*start) - self.z  # Objective function checks closeness to z
        optimal_coef = fsolve(objective, 1.0) # Find coefficient so that optimal_coef*start is on contour
        return start * optimal_coef
    

class LVManifold(Manifold):
    def __init__(self, Ns=50, step_size=1.0, σr=1.0, σf=1.0, r0=100, f0=100, z_true=(0.4, 0.005, 0.05, 0.001), seed=1111, seeds=[2222, 3333, 4444, 5555], n_chains=4):
        """Class defining the data manifold for the Lotka-Volterra ABC problem. The simulator is defined by an
        Euler-Marayama discretization of the LV SDE.

        Args:
            Ns (int, optional): Number of discretization time steps in the forward simulator. Defaults to 50.
            step_size (float, optional): Step size used in the discretization within the forward simulator. Defaults to 1.0.
            r0 (int, optional): Number of preys at time t=0. Defaults to 100.
            f0 (int, optional): Number of predators at time t=0. Defaults to 100.
            z_true (tuple, optional): True parameter values used to generate the data. Defaults to (0.4, 0.005, 0.05, 0.001).
            seed (int, optional): Random seed used to generate the data. Defaults to 1111.
            seeds (list, optional): List of seeds. Each seed used to find initial point for each chain. Defaults to [2222, 3333, 4444, 5555].
            n_chains (int, optional): Number of chains used to compute ESS using ArViz. Defaults to 4.
        """
        assert len(seeds) == n_chains, "Number of seeds must equal number of chains."
        self.Ns = Ns              # Number of steps used in integrating LV SDE
        self.m = 2*self.Ns        # Number of constraints = dimensionality of data
        self.d = 4                # Dimensionality of parameter
        self.n = self.d + self.m  # Dimensionality of ambient space
        self.δ = step_size        # Step size for discretization (not for sampling!)
        self.σr = σr              # Scale for noise in prey step
        self.σf = σf              # Scale for noise in predator step
        self.r0 = r0              # Initial prey population
        self.f0 = f0              # Initial predator population
        self.z_true = np.array(z_true)  # True parameter
        self.q_dist = MVN(zeros(self.n), eye(self.n))   # proposal for THUG
        self.seeds = seeds
        self.n_chains = n_chains 
        
        # generate data
        self.data_seed = seed
        self.rng = default_rng(self.data_seed)
        self.u1_true = self.z_to_u1(self.z_true)
        self.u2_true = self.rng.normal(loc=0.0, scale=1.0, size=2*self.Ns)
        self.u_true = np.concatenate((self.u1_true, self.u2_true))
        self.ystar  = self.u_to_x(self.u_true)
        
    def z_to_u1(self, z):
        """Transforms a parameter z into u1 (standard normal variables)."""
        assert len(z) == 4, "z should have length 4, but found {}".format(len(z))
        m_param = -2*ones(4)
        s_param = ones(4)
        return (log(z) - m_param) / s_param
    
    def u1_to_z(self, u1):
        """Given u1, it maps it to z."""
        assert len(u1) == 4, "u1 should have length 4, but found {}".format(len(u1))
        m_param = -2*ones(4)
        s_param = ones(4)
        return exp(s_param*u1 + m_param)
    
    def g(self, u):
        """Takes [u1, u2] and returns [z, u2]"""
        assert len(u) == self.n, "u should have length {}, but found {}".format(self.n, len(u))
        return np.concatenate((self.u1_to_z(u[:4]), u[4:]))
    
    def u_to_x(self, u):
        """Maps u=[u1, u2] to z."""
        assert len(u) == self.n, "u should have length {}, but found {}.".format(self.n, len(u))
        u1, u2 = u[:4], u[4:]
        u2_r   = u2[::2]
        u2_f   = u2[1::2]
        z1, z2, z3, z4 = self.u1_to_z(u1)
        r = np.full(self.Ns + 1, fill_value=np.nan)
        f = np.full(self.Ns + 1, fill_value=np.nan)
        r[0] = self.r0
        f[0] = self.f0
        for s in range(1, self.Ns+1):
            r[s] = r[s-1] + self.δ*(z1*r[s-1] - z2*r[s-1]*f[s-1]) + sqrt(self.δ)*self.σr*u2_r[s-1]
            f[s] = f[s-1] + self.δ*(z4*r[s-1]*f[s-1] - z3*f[s-1]) + sqrt(self.δ)*self.σf*u2_f[s-1]
        return np.ravel([r[1:], f[1:]], 'F')
    
    def zu2_to_x(self, zu2):
        """Same as u_to_x but this takes as input [z, u2]."""
        assert len(zu2) == self.n, "zu2 should have length {}, but found {}".format(self.n, len(zu2))
        z1, z2, z3, z4 = zu2[:4]
        u2 = zu2[4:]
        u2_r = u2[::2]
        u2_f = u2[1::2]
        r = np.full(self.Ns + 1, fill_value=np.nan)
        f = np.full(self.Ns + 1, fill_value=np.nan)
        r[0] = self.r0
        f[0] = self.f0
        for s in range(1, self.Ns+1):
            r[s] = r[s-1] + self.δ*(z1*r[s-1] - z2*r[s-1]*f[s-1]) + sqrt(self.δ)*self.σr*u2_r[s-1]
            f[s] = f[s-1] + self.δ*(z4*r[s-1]*f[s-1] - z3*f[s-1]) + sqrt(self.δ)*self.σf*u2_f[s-1]
        return np.ravel([r[1:], f[1:]], 'F')
    
    def Jg(self, ξ):
        """Jacobian of the function g:[u_1, u_2] --> [z, u_2]."""
        assert len(ξ) == self.n, "ξ should have length {}, but found {}.".format(self.n, len(ξ))
        m_param = -2*ones(4)
        s_param = ones(4)
        return diag(np.concatenate((s_param*exp(s_param*ξ[:4] + m_param), ones(2*self.Ns))))
    
    def oneat(self, ix, length=None):
        """Generates a vector of zeros of length `length` with a one at index ix."""
        assert type(ix) == int, "index for oneat() should be integer but found {}".format(type(ix))
        if length is None:
            length = self.n
        output = zeros(length)
        output[ix] = 1
        return output
    
    def Jf(self, ξ):
        """Jacobian of the function f:[z, u_2] --> x.
        Assume r and f contains r0 and f0 at the start."""
        assert len(ξ) == self.n, "ξ should have length {}, but found {}.".format(self.n, len(ξ))
        J = zeros((self.m, self.n))
        δ  = self.δ
        r0 = self.r0
        f0 = self.f0 
        σr = self.σr
        σf = self.σf
        # Sete first two rows: dr1_dξ and df1_dξ
        J[0, :] = np.concatenate(([δ*r0, -δ*r0*f0, 0, 0], sqrt(δ)*σr*self.oneat(0, length=self.m)))
        J[1, :] = np.concatenate(([0, 0, -δ*f0, δ*r0*f0], sqrt(δ)*σf*self.oneat(1, length=self.m)))
        # Evaluate function at the ξ to find r and f at this ξ.
        x = self.zu2_to_x(ξ)
        r = np.concatenate(([r0], x[::2]))
        f = np.concatenate(([f0], x[1::2]))
        # Grab the parameters
        z1, z2, z3, z4 = ξ[:4]
        # Loops through the time steps and compute the Markovian rows
        for s in range(1, self.Ns):
            J[2*s, :]     = J[2*s-2, :] + δ*(self.oneat(0)*r[s] + z1*J[2*s-2, :] -(self.oneat(1)*r[s]*f[s] + z2*J[2*s-2, :]*f[s] + z2*r[s]*J[2*s-1, :])) + sqrt(δ)*σr*self.oneat(2*s+4)
            J[2*s + 1, :] = J[2*s-1, :] + δ*(self.oneat(3)*r[s]*f[s] + z4*J[2*s-2, :]*f[s] + z4*r[s]*J[2*s-1, :] - self.oneat(2)*f[s] - z3*J[2*s-1, :]) + sqrt(δ)*σf*self.oneat(2*s+5)
        return J

    def q(self, ξ):
        """Constraint function taking u=[u1, u2] and comparing against true data."""
        assert len(ξ) == self.n, "ξ should have length {}, but found {}.".format(self.n, len(ξ))
        return self.u_to_x(ξ) - self.ystar
    
    def J(self, ξ):
        """Jacobian. Here u=[u1, u2]."""
        assert len(ξ) == self.n, "ξ should have length {}, but found {}.".format(self.n, len(ξ))
        return self.Jf(self.g(ξ)).dot(self.Jg(ξ))
    
    def Q(self, ξ):
        """Transpose of Jacobian."""
        assert len(ξ) == self.n, "ξ should have length {}, but found {}.".format(self.n, len(ξ))
        return self.J(ξ).T
    
    def logη(self, ξ):
        """Density on Manifold wrt Hausdorff measure."""
        assert len(ξ) == self.n, "ξ should have length {}, but found {}.".format(self.n, len(ξ))
        try:
            J = self.J(ξ)
            logprior = -ξ@ξ/2
            correction_term  = - math.prod(np.linalg.slogdet(J@J.T))/2 
            return  logprior + correction_term
        except ValueError as e:
            return -np.inf
        
    def find_point_on_manifold(self, maxiter=2000, tol=1e-14, random_u2_guess=False):
        """Finds a point on the Manifold with input u=[u1, u2]."""
        u2_guess = randn(self.m) if random_u2_guess else zeros(self.m)
        i = 0
        with catch_warnings():
            filterwarnings('error')
            while i <= maxiter:
                i += 1
                try: 
                    u1_init  = randn(self.d)*0.1 - 4
                    function = lambda u2: self.q(np.concatenate((u1_init, u2)))
                    fprime   = lambda u2: self.J(np.concatenate((u1_init, u2)))[:, self.d:]
                    u2_found = fsolve(function, u2_guess, xtol=tol, fprime=fprime)
                    u_found = np.concatenate((u1_init, u2_found))
                    return u_found
                except RuntimeWarning:
                    continue
        raise ValueError("Couldn't find a point, try again.")
        
    def find_point_on_manifold_given_u1true(self, maxiter=2000, tol=1e-14, random_u2_guess=False):
        """Finds a point on the Manifold starting from u1_true."""
        i = 0
        with catch_warnings():
            filterwarnings('error')
            while i <= maxiter:
                i += 1
                try:
                    u2_guess = randn(self.m) if random_u2_guess else zeros(self.m)
                    function = lambda u2: self.q(np.concatenate((self.u1_true, u2)))
                    u2_found = fsolve(function, u2_guess, xtol=tol)
                    u_found = np.concatenate((self.u1_true, u2_found))
                    return u_found
                except RuntimeWarning:
                    continue
        raise ValueError("Couldn't find a point, try again.")

    def find_init_points_for_each_chain(self, u1_true=True, random_u2_guess=False, tol=1e-14, maxiter=5000):
        """Finds `n_chains` initial points on the manifold.

        Args:
            u1_true (boool, optional): Whether to use u1 that generated the data or sample it at random.
            random_u2_guess (bool, optional): Whether to generate the initial u2 guess at random or as a zero vector. Defaults to False.
            tol (float, optional): tolerance for fsolve. Defaults to 1e-14.
            maxiter (int, optional): Maximum number of iterations for optimization procedure. Defaults to 5000.

        Returns:
            ndarray: array having dimension (n_chains, n), containing each point on a row.
        """
        u0s = zeros((self.n_chains, self.n))
        for i in range(self.n_chains):
            if u1_true:
                u0s[i, :] = self.find_point_on_manifold_given_u1true(maxiter=maxiter, tol=tol, random_u2_guess=random_u2_guess)
            else:
                u0s[i, :] = self.find_point_on_manifold(maxiter=maxiter, tol=tol, random_u2_guess=random_u2_guess)
        self.u0s = u0s 
        return self.u0s
            
    def transform_usamples_to_zsamples(self, samples):
        """Given samples of size (N, 4 + 2*Ns) it takes the first 4 columns and transforms them."""
        n_samples, input_dim = samples.shape
        assert input_dim == self.n, "Wrong dim. Expected {} , found {}".format(self.n, input_dim)
        return apply_along_axis(self.u1_to_z, 1, samples[:, :4])
    
    def log_normal_kernel(self, ξ, ϵ):
        """Log normal kernel density."""
        assert len(ξ) == self.n, "ξ should have length {}, but found {}.".format(self.n, len(ξ))
        u = norm(self.q(ξ))   ##### THIS IS NOT THE USUAL u
        return -u**2/(2*(ϵ**2)) -0.5*log(2*pi*(ϵ**2))

    def generate_logpi(self, ϵ):
        """Generates ABC posterior using a certain epsilon value. Uses a Gaussian kernel. """
        logηϵ = lambda ξ: self.log_normal_kernel(ξ, ϵ) - ξ@ξ/2
        return logηϵ
    
    def is_on_manifold(self, ξ, tol=1e-14):
        """Checks if a point is on the manifold."""
        return max(abs(self.q(ξ))) <= tol


class GKManifold(Manifold):
    def __init__(self, ystar):
        self.m = len(ystar)            # Number constraints = dimensionality of the data
        self.d = 4                     # Manifold has dimension 4 (like the parameter θ)
        self.n = self.d + self.m       # Dimension of ambient space is m + 4
        self.ystar = ystar
        # N(0, 1) ---> U(0, 10).
        self.G    = lambda θ: 10*ndtr(θ)
        # U(0, 10) ---> N(0, 1)
        self.Ginv = lambda θ: ndtri(θ/10)

    def q(self, ξ):
        """Constraint for G and K."""
        ξ = concatenate((self.G(ξ[:4]), ξ[4:]))  # expecting theta part to be N(0, 1)
        with catch_warnings():
            filterwarnings('error')
            try:
                return (ξ[0] + ξ[1]*(1 + 0.8*(1 - exp(-ξ[2]*ξ[4:]))/(1 + exp(-ξ[2]*ξ[4:]))) * ((1 + ξ[4:]**2)**ξ[3])*ξ[4:]) - self.ystar
            except RuntimeWarning:
                raise ValueError("Constraint found Overflow warning.")
                
    def _q_raw_uniform(self, ξ):
        """Constraint function expecting ξ[:4] ~ U(0, 10). It doesn't do any warning check."""
        return (ξ[0] + ξ[1]*(1 + 0.8*(1 - exp(-ξ[2]*ξ[4:]))/(1 + exp(-ξ[2]*ξ[4:]))) * ((1 + ξ[4:]**2)**ξ[3])*ξ[4:]) - self.ystar
    def _q_raw_normal(self, ξ):
        """Same as `_q_raw_uniform` except expects ξ[:4]~N(0,1)."""
        ξ = concatenate((self.G(ξ[:4]), ξ[4:])) 
        return self._q_raw_uniform(ξ)

    def Q(self, ξ):
        """Transpose of Jacobian for G and K. """
        ξ = concatenate((self.G(ξ[:4]), ξ[4:]))
        return vstack((
        ones(len(ξ[4:])),
        (1 + 0.8 * (1 - exp(-ξ[2] * ξ[4:])) / (1 + exp(-ξ[2] * ξ[4:]))) * ((1 + ξ[4:]**2)**ξ[3]) * ξ[4:],
        8 * ξ[1] * (ξ[4:]**2) * ((1 + ξ[4:]**2)**ξ[3]) * exp(ξ[2]*ξ[4:]) / (5 * (1 + exp(ξ[2]*ξ[4:]))**2),
        ξ[1]*ξ[4:]*((1+ξ[4:]**2)**ξ[3])*(1 + 9*exp(ξ[2]*ξ[4:]))*log(1 + ξ[4:]**2) / (5*(1 + exp(ξ[2]*ξ[4:]))),
        diag(ξ[1]*((1+ξ[4:]**2)**(ξ[3]-1))*(((18*ξ[3] + 9)*(ξ[4:]**2) + 9)*exp(2*ξ[2]*ξ[4:]) + (8*ξ[2]*ξ[4:]**3 + (20*ξ[3] + 10)*ξ[4:]**2 + 8*ξ[2]*ξ[4:] + 10)*exp(ξ[2]*ξ[4:]) + (2*ξ[3] + 1)*ξ[4:]**2 + 1) / (5*(1 + exp(ξ[2]*ξ[4:]))**2))
    ))
    
    def J(self, ξ):
        """Safely computes Jacobian."""
        with catch_warnings():
            filterwarnings('error')
            try:
                return self.Q(ξ).T
            except RuntimeWarning:
                raise ValueError("J computation found Runtime warning.")
                
    def fullJacobian(self, ξ):
        """J_f(G(ξ)) * J_G(ξ)."""
        JGbar = block_diag(10*np.diag(ndist.pdf(ξ[:4])), eye(len(ξ[4:])))
        return self.J(ξ) @ JGbar
                
    def log_parameter_prior(self, θ):
        """IMPORTANT: Typically the prior distribution is a U(0, 10) for all four parameters.
        We keep the same prior but since we don't want to work on a constrained space, we 
        reparametrize the problem to an unconstrained space N(0, 1)."""
        with catch_warnings():
            filterwarnings('error')
            try:
                return udist.logpdf(self.G(θ), loc=0.0, scale=10.0).sum() + ndist.logpdf(θ).sum()
            except RuntimeWarning:
                return -np.inf
            
    def logprior(self, ξ):
        """Computes the prior distribution for G and K problem. Notice this is already reparametrized."""
        return self.log_parameter_prior(ξ[:4]) - ξ[4:]@ξ[4:]/2

    def logη(self, ξ):
        """log posterior for c-rwm. This is on the manifold."""
        try:
            J = self.J(ξ)
            logprior = self.logprior(ξ)
            correction_term  = - math.prod(np.linalg.slogdet(J@J.T))/2 
            return  logprior + correction_term
        except ValueError as e:
            return -np.inf
        
    def generate_logηϵ(self, ϵ, kernel='normal'):
        """Returns the log abc posterior for THUG."""
        if kernel not in ['normal']:
            raise NotImplementedError
        else:
            def log_abc_posterior(ξ):
                """Log-ABC-posterior."""
                u = self.q(ξ)
                m = len(u)
                return self.logprior(ξ) - u@u/(2*ϵ**2) - m*log(ϵ) - m*log(2*pi)/2
            return log_abc_posterior
            
    def logp(self, v):
        """Log density for normal on the tangent space."""
        return MVN(mean=zeros(self.d), cov=eye(self.d)).logpdf(v)
    
    def is_on_manifold(self, ξ, tol=1e-8):
        """Checks if ξ is on the ystar manifold."""
        return np.max(abs(self.q(ξ))) < tol
    
    def find_point_on_manifold_from_θ(self, θfixed, ϵ, maxiter=2000, tol=1.49012e-08):
        """Same as the above but we provide the θfixed. Can be used to find a point where
        the theta is already θ0."""
        i = 0
        log_abc_posterior = self.generate_logηϵ(ϵ)
        function = lambda z: self._q_raw_normal(concatenate((θfixed, z)))
        with catch_warnings():
            filterwarnings('error')
            while i <= maxiter:
                i += 1
                try:
                    z_guess  = randn(self.m)
                    z_found  = fsolve(function, z_guess, xtol=tol)
                    ξ_found  = concatenate((θfixed, z_found))
                    if not isfinite([log_abc_posterior(ξ_found)]):
                        raise ValueError("Couldn't find a point.")
                    else:
                        return ξ_found
                except RuntimeWarning:
                    continue
            raise ValueError("Couldn't find a point, try again.")