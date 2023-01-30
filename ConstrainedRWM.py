from numpy import inf, zeros, log, finfo
from numpy.random import default_rng
from numpy.linalg import solve, norm, cond
import scipy.linalg as la


def CRWM(x0, manifold, n, T, B, tol, rev_tol, maxiter=50, norm_ord=2, seed=1234):
    """Constrained Random Walk with RATTLE integration.
    
    Arguments

    :param x0: Initial state of the Markov Chain. Must lie on the manifold. 
    :type x0: ndarray

    :param manifold: Instance from the Manifold class. Contains the constraint function among other functions.
    :type manifold: Manifold

    :param n: Number of samples.
    :type n: int

    :param T: Total integration time.
    :type T: float

    :param B: Number of Leapfrog steps.
    :type B: int

    :param tol: Tolerance for forward projection.
    :type tol: float

    :param rev_tol: Tolerance for reverse projection.
    :type rev_tol: float

    :param maxiter: Maximum number of iterations for projection and reprojection steps.
    :type maxiter: int

    :param norm_ord: Order of the norm used to check convergence. Should be either `2` or `np.inf`
    :type norm_ord: float

    :param seed: Seed of the random number generator. Used for reproducibility.
    :type seed: int

    Returns
    
    :param samples: Array containing samples as rows. Has dimension (n, d) where d is the dimension of ambient space.
    :type samples: ndarray

    :param n_evals: Total number of Jacobian evaluations.
    :type n_evals: int

    :param accepted: Binary array of length `n` showing `1` if that samples was accepted, `0` otherwise.
    :type accepted: ndarray
    """
    assert type(B) == int
    assert norm_ord in [2, inf]
    assert len(x0) == manifold.n, "Initial point has wrong dimension."
    # Check arguments
    n = int(n)  
    B = int(B)
    δ = T / B
    d, m = manifold.get_dimension(), manifold.get_codimension()
    # Set random number generator
    rng = default_rng(seed=seed)

    # Initial point on the manifold
    x = x0
    compute_J = lambda x: manifold.fullJacobian(x)

    # House-keeping
    samples = zeros((n, d + m))    # Store n samples on the manifold
    samples[0, :] = x
    i = 1
    N_EVALS = {'jacobian': 0, 'density': 0}
    ACCEPTED = zeros(n)
    # Define function to compute density
    def logη(x):
        """Computes log density on Manifold but makes sure everything is behaving nicely."""
        return manifold.logη(x)

    # Log-uniforms for MH accept-reject step
    logu = log(rng.uniform(low=0.0, high=1.0, size=n))

    # Compute jacobian & density value
    Jx    = compute_J(x)
    logηx = logη(x)
    N_EVALS['jacobian'] += 1
    N_EVALS['density'] += 1
    
    def linear_project(v, J):
        """Projects by solving linear system.
        
        Arguments: 
        
        :param v: Velocity to be projected.
        :type v: ndarray
        
        :param J: Jacobian identifying on where `v` should be projected to.
        :type J: ndarray
        
        Returns:
        
        :param v_projected: Velocity projected onto normal space identified by `J`.
        :type v_projected: ndarray
        """
        return J.T @ solve(J@J.T, J@v)

    # Constrained Step Function
    def constrained_rwm_step(x, v, tol, maxiter, Jx, norm_ord=norm_ord):
        """One step of the constrained Leapfrog integrator for C-RWM.
        
        Arguments: 
        
        :param x: Initial position.
        :type x: ndarray
        
        :param v: Initial velocity.
        :type v: ndarray
        
        :param tol: Tolerance used to check projection onto manifold.
        :type tol: float
        
        :param maxiter: Maximum number of iterations allowed to project onto manifold. 
        :type maxiter: int
        
        :param Jx: Jacobian evaluated at initial position `x`.
        :type Jx: ndarray
        
        :param norm_ord: Order of the norm used to check convergence of the projection. Can be either `2` or `np.inf`.
        :type norm_ord: float
        
        Returns:
        
        :param y: Final position on the manifold. 
        :type y: ndarray
        
        :param v_projected_endposition: Final velocity, projected onto tangent space at `y`.
        :type v_projected_endposition: ndarray
        
        :param Jy: Jacobian evaluated at position `y`.
        :type Jy: ndarray
        
        :param flag: Flag indicating whether projection was successful (`1`) or not (`0`).
        :type flag: int
        
        :param n_grad: Number of Jacobian evaluations used to project onto manifold.
        :type n_grad: int
        """
        # Project momentum
        v_projected = v - linear_project(v, Jx) 
        # Unconstrained position step
        x_unconstr = x + v_projected
        # Position Projection
        a, flag, n_grad = projectCRWM(manifold, x_unconstr, Jx.T, tol, maxiter, norm_ord=norm_ord)
        y = x_unconstr - Jx.T @ a 
        try:
            Jy = compute_J(y) 
        except ValueError as e:
            print("Jacobian computation at projected point failed. ", e)
            return x, v, Jx, 0, n_grad + 1
        # backward velocity
        v_would_have = y - x
        # Find backward momentum & project it to tangent space at new position
        v_projected_endposition = v_would_have - linear_project(v_would_have, Jy) #qr_project(v_would_have, Jy) #qr_project((y - x) / δ, Jy)
        # Return projected position, projected momentum and flag
        return y, v_projected_endposition, Jy, flag, n_grad + 1
    
    def constrained_leapfrog(x0, v0, J0, B, tol, rev_tol, maxiter, norm_ord=norm_ord):
        """Constrained Leapfrog/RATTLE for C-RWM.
        
        Arguments: 
        
        :param x0: Initial position.
        :type x0: ndarray
        
        :param v0: Initial velocity.
        :type v0: ndarray
        
        :param J0: Jacobian at initial position `J(x0)`.
        :type J0: ndarray
        
        :param B: Number of Leapfrog steps. 
        :type B: int
        
        :param tol: Tolerance for checking forward projection was successful.
        :type tol: float
        
        :param rev_tol: Tolerance for checking backward projection was successful. 
        :type rev_tol: float
        
        :param maxiter: Maximum number of iterations allowed in both forward and backward projections.
        :type maxiter: int
        
        :param norm_ord: Order of the norm used to check convergence of both forward and backward projections. 
                         Can be either `2` or `np.inf`.
        :type norm_ord: float

        Returns:

        :param x: Final position.
        :type x: ndarray

        :param v: Final velocity
        :type v: ndarray

        :param J: Jacobian evaluated at final position. 
        :type J: ndarray

        :param successful: Boolean flag indicating whether both projections were successful.
        :type successful: bool

        :param n_jacobian_evaluations: Number of Jacobian evaluations used in forward and backward projections
                                       combined. 
        :type n_jacobian_evaluations: int
        """
        successful = True
        n_jacobian_evaluations = 0
        x, v, J = x0, v0, J0
        for _ in range(B):
            xf, vf, Jf, converged_fw, n_fw = constrained_rwm_step(x, v, tol, maxiter, J, norm_ord=norm_ord)
            xr, vr , Jr, converged_bw, n_bw = constrained_rwm_step(xf, -vf, tol, maxiter, Jf, norm_ord=norm_ord)
            n_jacobian_evaluations += (n_fw + n_bw)  # +2 due to the line Jy = manifold.Q(y).T
            if (not converged_fw) or (not converged_bw) or (norm(xr - x, ord=norm_ord) >= rev_tol):
                successful = False
                return x0, v0, J0, successful, n_jacobian_evaluations
            else:
                x = xf
                v = vf
                J = Jf
        return x, v, J, successful, n_jacobian_evaluations

    for i in range(n):
        v = rng.normal(loc=0.0, scale=δ, size=(m+d)) # Sample in the ambient space.
        xp, vp, Jp, LEAPFROG_SUCCESSFUL, n_jac_evals = constrained_leapfrog(x, v, Jx, B, tol=tol, rev_tol=rev_tol, maxiter=maxiter)
        N_EVALS['jacobian'] += n_jac_evals
        if LEAPFROG_SUCCESSFUL:
            logηp = logη(xp)
            N_EVALS['density'] += 1
            if logu[i] <= logηp - logηx - (vp@vp)/2 + (v@v)/2: 
                # Accept
                ACCEPTED[i - 1] = 1
                x, logηx, Jx = xp, logηp, Jp
                samples[i, :] = xp
            else:
                # Reject
                samples[i, :] = x
                ACCEPTED[i - 1] = 0
        else:
            # Reject
            samples[i, :] = x
            ACCEPTED[i - 1] = 0
    return samples, N_EVALS, ACCEPTED


def projectCRWM(manifold, z, Q, tol = 1.48e-08 , maxiter = 50, norm_ord=2):
    '''
    This version is the version of Miranda & Zappa. It also appears in Graham's papers.
    It retuns i, the number of iterations i.e. the number of gradient evaluations used.

    Arguments: 

    :param manifold: Instance of the class `Manifold`.
    :type manifold: `Manifolds.Manifold`

    :param z: Point on the tangent space that needs to be projected back to the manifold. Corresponds to `x+v`.
    :type z: ndarray

    :param Q: Transpose of Jacobian matrix at the original point `x`. Used to project the point back to the manifold.
    :type Q: ndarray

    :param tol: Tolerance used to check convergence onto manifold.
    :type tol: float

    :param maxiter: Maximum number of iterations allowed to try and project `z` onto manifold. 
    :type maxiter: int

    :param norm_ord: Order of the norm used to check convergence. Can be either `2` or `np.inf`.
    :type norm_ord: float

    Returns: 

    :param a: Value `a` such that `z - Q@a` lies on the manifold within tolerance `tol`.
    :type a: ndarray

    :param flag: Flag (either `0` or `1`) determining whehter projection was successful (`1`) or not (`0`).
    :type flag: int

    :param i: Number of iterations used to project (or not) `z` onto the manifold.
    :type i: int
    '''
    a, flag, i = zeros(Q.shape[1]), 1, 0

    # Compute the constrained at z - Q@a. If it fails due to overflow error, return a rejection altogether.
    try:
        projected_value = manifold.q(z - Q@a)
    except ValueError as e:
        return a, 0, i
    # While loop
    while la.norm(projected_value, ord=norm_ord) >= tol:
        try:
            Jproj = manifold.fullJacobian(z - Q@a)
        except ValueError as e:
            print("Jproj failed. ", e)
            return zeros(Q.shape[1]), 0, i
        # Check that Jproj@Q is invertible. Do this by checking condition number 
        # see https://stackoverflow.com/questions/13249108/efficient-pythonic-check-for-singular-matrix
        GramMatrix = Jproj@Q
        if cond(GramMatrix) < 1/finfo(z.dtype).eps:
            Δa = la.solve(GramMatrix, projected_value)
            a += Δa
            i += 1
            if i > maxiter:
                return zeros(Q.shape[1]), 0, i
            # If we are not at maxiter iteration, compute new projected value
            try:
                projected_value = manifold.q(z - Q@a)
            except ValueError as e:
                return zeros(Q.shape[1]), 0, i
        else:
            # Fail
            return zeros(Q.shape[1]), 0, i
    return a, 1, i
