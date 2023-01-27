from numpy import zeros, log, vstack, zeros_like, eye
from numpy.random import default_rng
from numpy.linalg import solve
from scipy.linalg import qr, lstsq
from scipy.stats import multivariate_normal as MVN
from warnings import catch_warnings, filterwarnings


def THUG(x0, T, B, N, α, logpi, jac, method='qr', seed=1234):
    """Tangential Hug Sampler  (THUG). Two projection methods available:
        - 'qr': projects onto row space of Jacobian using QR decomposition.
        - 'linear': solves a linear system to project.
    Arguments:
    x0 : ndarray
         Initial state of the Markov Chain. For the algorithm to work, this should be in a region of non-zero density.
    T  : float
         Total integration time. In the paper T = B*δ.
    B  : int
         Number of bounces per trajectory/sample. Equivalent to the number of leapfrog steps in HMC. Clearly δ = T/B.
    N  : int
         Number of samples.
    α  : float
         Squeezing parameter for THUG. Must be in [0, 1), the larger α, the more we squeeze the auxiliary velocity variable
        towards the tangent space.
    logpi : callable
            Function computing the log density for the target (which should be a filamentary distribution).
    jac   : callable
            Function computing the Jacobian of f at a point.
    method : string
             Method for projecting onto the row space of the Jacobian. Two options are available QR or 'linear'.

    Returns:
    samples : ndarray
              (N, len(x0)) array containing the samples from logpi.
    acceptances : ndarray
                  Array of 0s and 1s indicating whether a certain sample was an acceptance or a rejection.
    """
    assert method == 'qr' or method == 'linear' or method == 'lstsq'
    rng = default_rng(seed)
    def qr_project(v, J):
        """Projects using QR decomposition."""
        Q, _ = qr(J.T, mode='economic')
        return Q.dot((Q.T.dot(v)))
    def linear_project(v, J):
        """Projects by solving linear system."""
        return J.T.dot(solve(J.dot(J.T), J.dot(v)))
    def lstsq_project(v, J):
        """Projects using scipy's Least Squares Routine."""
        return J.T.dot(lstsq(J.T, v)[0])
    if method == 'qr':
        project = qr_project
    elif method == 'linear':
        project = linear_project
    else:
        project = lstsq_project
    # Jacobian function raising an error for RuntimeWarning
    def safe_jac(x):
        """Raises an error when a RuntimeWarning appears."""
        while catch_warnings():
            filterwarnings('error')
            try:
                return jac(x)
            except RuntimeWarning:
                raise ValueError("Jacobian computation failed due to Runtime Warning.")
    samples, acceptances = x0, zeros(N)
    q = MVN(mean=zeros_like(x0), cov=eye(len(x0)))
    # Compute initial Jacobian. 
    for i in range(N):
        v0s = rng.normal(size=len(x0))
        # Squeeze
        v0 = v0s - α * project(v0s, safe_jac(x0)) #jac(x0))
        v, x = v0, x0
        logu = log(rng.uniform())
        δ = T / B
        for _ in range(B):
            x = x + δ*v/2
            v = v - 2 * project(v, safe_jac(x)) #jac(x))
            x = x + δ*v/2
        # Unsqueeze
        v = v + (α / (1 - α)) * project(v, safe_jac(x)) #jac(x))
        if logu <= logpi(x) + q.logpdf(v) - logpi(x0) - q.logpdf(v0s):
            samples = vstack((samples, x))
            acceptances[i] = 1         # Accepted!
            x0 = x
        else:
            samples = vstack((samples, x0))
            acceptances[i] = 0         # Rejected
    return samples[1:], acceptances
