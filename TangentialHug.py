from numpy import zeros, log, vstack
from numpy.random import rand
from numpy.linalg import solve
from scipy.linalg import qr, lstsq
from warnings import catch_warnings, filterwarnings


def HugTangentialMultivariate(x0, T, B, N, α, q, logpi, jac, method='qr', return_n_grad=False):
    """Multidimensional Tangential Hug sampler. Two possible methods:
    - 'qr': projects onto row space of Jacobian using QR decomposition.
    - 'linear': solves a linear system to project.
    """
    assert method == 'qr' or method == 'linear' or method == 'lstsq'
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
    # Compute initial Jacobian. 
    n_grad_computations = 0
    for i in range(N):
        v0s = q.rvs()
        # Squeeze
        v0 = v0s - α * project(v0s, safe_jac(x0)) #jac(x0))
        n_grad_computations += int(α > 0)
        v, x = v0, x0
        logu = log(rand())
        δ = T / B
        for _ in range(B):
            x = x + δ*v/2
            v = v - 2 * project(v, safe_jac(x)) #jac(x))
            n_grad_computations += 1
            x = x + δ*v/2
        # Unsqueeze
        v = v + (α / (1 - α)) * project(v, safe_jac(x)) #jac(x))
        n_grad_computations += int(α > 0)
        if logu <= logpi(x) + q.logpdf(v) - logpi(x0) - q.logpdf(v0s):
            samples = vstack((samples, x))
            acceptances[i] = 1         # Accepted!
            x0 = x
        else:
            samples = vstack((samples, x0))
            acceptances[i] = 0         # Rejected
    if return_n_grad:
        return samples[1:], acceptances, n_grad_computations
    else: 
        return samples[1:], acceptances
