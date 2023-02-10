""" 
Reproduces Figure 3 in Example 1.4.1 in the paper Approximate Manifold Sampling.
"""
from numpy import log, array, eye, errstate, zeros, meshgrid, vstack, arange
from numpy.linalg import solve, norm
from numpy.random import default_rng, randint
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal as MVN
from scipy.optimize import fsolve
from ..HelperFunctions import prep_contour
from ..RWM import RWM
from ..TangentialHug import THUG


def prep_contour(xlims, ylims, step, func):
    """Given a scalar-valued function `f(x, y)` called `func`, this helper function outputs
    `x`, `y`, and `f([x, y])` on a grid, so that it can be used to plot contours of `f`. 
    
    Arguments:
    
    :param xlims: Limits for the x-coordinates of the grid. Must be a list of the form `[xmin, xmax]`.
    :type xlims: list

    :param ylims: Limits for the y-coordinates of the grid. Must be a list of the form `[ymin, ymax]`.
    :type ylims: list

    :param step: Step-sized used within the grid `[xmin, xmax] x [ymin, ymax]` to generate gridpoints.
    :type step: float

    :param func: Scalar-valued unction taking 1D arrays of length 2 as input. This means it must take
                 `np.array([x, y])` as input and not `f(x, y)`.
    :type func: callable

    Returns:

    :param x: Grid of x values to feed to `plt.contour()`.
    :type x: ndarray

    :param y: Grid of y values to feed to `plt.contour()`.
    :type y: ndarray

    :param z: Grid of z values corresponding to `f([x, y])`.
    :type z: ndarray

    Example:

    ```
    import numpy as np
    from scipy.stats import multivariate_normal
    import matplotlib.pyplot as plt

    f = multivariate_normal(np.zeros(2), np.eye(2)).pdf
    fig, ax = plt.subplots()
    ax.contour(*prep_contour([-2, 2], [-2, 2], 0.01, f))
    plt.show()
    ```
    """
    x, y = meshgrid(arange(*xlims, step), arange(*ylims, step))
    xys  = vstack((x.flatten(), y.flatten())).T
    return x, y, func(xys).reshape(x.shape)


def logprior(xi):
    """Log-prior density. The prior is a mixture of two Gaussians. Mixture coefficients are both 0.5,
    the mean vectors are [-1, 0] and [1, 0] respectively, and the covariance matrices are both 0.5I,
    where I is a 2x2 identity matrix.
    """
    return log(0.5 * MVN(array([-1, 0]), 0.5*eye(2)).pdf(xi) + 0.5 * MVN(array([1, 0]), 0.5*eye(2)).pdf(xi))

def logkernel(xi_matrix, epsilon, z):
    """Log uniform kernel. This is the broadcasted version of the log of an indicator kernel. It is 
    broadcasted in the sense that it accepts a matrix of `xi`s and compute the log-kernel value at each row
    simultaneously. """
    m = 1 if type(z) == float else len(z)
    with errstate(divide='ignore'):
        return log((abs(f(xi_matrix) - z) <= epsilon).astype('float64'))

# Constraint function. In this case, it is an ellipse, hence the log-density of a multivariate normal.
Σ = array([[1.0, 0.3], [0.3, 1.0]])
μ = zeros(2)
f = MVN(μ, Σ).logpdf


if __name__ == "__main__":
    ### Figure 3 ###
    # Common variables, and plot settings
    Z = -2.9513586307684885     # Level-set value for `f` identifying the manifold of interest.
    EPSILONS = [1.0, 0.5, 0.2]  # Decreasing epsilon values used in the smoothing kernel.
    XLIM = [-2, 2]              # `[xmin, xmax]` for `prep_contour`. Range of values used to plot contours.
    YLIM = [-2, 2]              # `[ymin, ymax]` for `prep_contour`. Range of values used to plot contours.
    STEP = 0.001                # Step used to generate grid of values for contours.
    
    # Grids for filamentary distribution, unconstrained mixture, and manifold respectively
    CONTOURS_LOGPRIOR = prep_contour(XLIM, YLIM, 0.01, logprior)
    CONTOURS_LOGPOST  = [prep_contour(XLIM, YLIM, STEP, lambda x: logprior(x) + logkernel(x, E, Z)) for E in EPSILONS]
    MANIFOLD          = prep_contour(XLIM, YLIM, 0.01, f)

    # Plot contours of filamentary distribution for different epsilons (Figure 3)
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(18,6))
    for i in range(3):
        ax[i].contour(*CONTOURS_LOGPOST[i], colors='deepskyblue', linewidths=0.8, linestyles='solid', levels=20)
        ax[i].contour(*CONTOURS_LOGPRIOR, colors='gray', linestyles='solid', alpha=0.5, linewidths=0.5)
        ax[i].contour(*MANIFOLD, levels=[Z], linewidths=3.0, colors=['black'], linestyles='solid')
        ax[i].contour(*MANIFOLD, levels=[Z-EPSILONS[i], Z+EPSILONS[i]], linewidths=1.0, linestyles='--', colors='black')
        ax[i].set_xticks([])
        ax[i].set_yticks([])
        ax[i].set_aspect('equal')
    fig.tight_layout()
    plt.savefig("example141_figure3_contours_mixture_gaussian.png", dpi=500)
    plt.show()

    ### Figure 4 ###
    # Settings
    RWM_SCALES = [0.2, 0.7, 4.0]
    INITIAL_POINT_AMBIENT_SPACE = array([1.2, -0.5])
    N_SAMPLES = 10000
    EPSILON = 0.1
    SEED = 3585
    RNG = default_rng(seed=SEED)

    # Functions
    find_point_closest_to = lambda point: fsolve(func=lambda x: array([f(x) - Z, 0]), x0=point)
    logpost = lambda x: logprior(x) + logkernel(x, EPSILON, Z)
    jacobian_function = lambda x: -solve(Σ, x)

    # Sample
    x0 = find_point_closest_to(INITIAL_POINT_AMBIENT_SPACE)
    RWM_RESULTS = [RWM(x0, S, N_SAMPLES, logpost, rng=RNG) for S in RWM_SCALES]
    SAMPLES_THUG, ACC_THUG = THUG(x0=x0, T=5.0, B=5, N=N_SAMPLES, α=0.0, logpi=logpost, jac=jacobian_function, method='2d', rng=RNG)

    # Plot settings
    CONTOURS_LOGPOST  = prep_contour(XLIM, YLIM, STEP, logpost)
    N_DISPLAY = 500

    # Plot contours of filamenary distribution with samples from RWM and THUG
    fig, ax = plt.subplots(ncols=4, figsize=(18, 4))
    for i in range(4):
        ax[i].contour(*CONTOURS_LOGPOST, colors='deepskyblue', linewidths=0.8, linestyles='solid', levels=20)
        ax[i].contour(*CONTOURS_LOGPRIOR, colors='gray', linestyles='solid', alpha=0.5, linewidths=0.5)
        ax[i].contour(*MANIFOLD, levels=[Z], linewidths=3.0, colors=['black'], linestyles='solid')
        ax[i].contour(*MANIFOLD, levels=[Z-EPSILON, Z+EPSILON], linestyles='--', colors='black', linewidths=1.0)
        if i <= 2:   # RWM
            ax[i].plot(*RWM_RESULTS[i][0][:N_DISPLAY].T, color='red', lw=3, alpha=0.5, zorder=5, label='RWM {:.0f}%'.format(RWM_RESULTS[i][1].mean()*100))
            ax[i].scatter(*RWM_RESULTS[i][0][:N_DISPLAY].T, color='red', zorder=10)
        elif i == 3: # THUG
            ax[i].plot(*SAMPLES_THUG[:N_DISPLAY].T, color='green', lw=3, alpha=0.5, zorder=5, label="THUG {:.0f}%".format(ACC_THUG.mean()*100))
            ax[i].scatter(*SAMPLES_THUG[:N_DISPLAY].T, color='green', zorder=10)
        ax[i].legend(fontsize=15, loc='upper left')
        ax[i].set_aspect('equal')
        ax[i].set_xticks([])
        ax[i].set_yticks([])
    plt.tight_layout()
    plt.savefig('example_141_figure4_contours_with_samples.png')
    plt.show()