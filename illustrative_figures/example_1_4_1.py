""" 
Reproduces Figure 3 in Example 1.4.1 in the paper Approximate Manifold Sampling.
"""
from numpy import log, array, eye, errstate, zeros
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal as MVN
from ...ApproximateManifoldSampling.HelperFunctions import prep_contour


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
    with errstate(divide='ignore'):
        return log((abs(f(xi_matrix) - z) <= epsilon).astype('float64'))

# Constraint function. In this case, it is an ellipse, hence the log-density of a multivariate normal.
f = MVN(zeros(2), array([[1.0, 0.3], [0.3, 1.0]])).logpdf

if __name__ == "__main__":
    # Common variables, and plot settings
    Z = -2.9513586307684885     # Level-set value for `f` identifying the manifold of interest.
    EPSILONS = [1.0, 0.5, 2.0]  # Decreasing epsilon values used in the smoothing kernel.
    XLIM = [-2, 2]              # `[xmin, xmax]` for `prep_contour`. Range of values used to plot contours.
    YLIM = [-2, 2]              # `[ymin, ymax]` for `prep_contour`. Range of values used to plot contours.
    STEP = 0.001                # Step used to generate grid of values for contours.
    
    # Grids for filamentary distribution, unconstrained mixture, and manifold respectively
    CONTOURS_LOGPRIOR = prep_contour(XLIM, YLIM, 0.01, logprior)
    CONTOURS_LOGPOST  = prep_contour(XLIM, YLIM, STEP, lambda x: logprior(x) + logkernel(x, EPSILONS[i], Z))
    MANIFOLD          = prep_contour(XLIM, YLIM, 0.01, f)

    # Plot
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(18,6))
    for i in range(3):
        ax[i].contour(*CONTOURS_LOGPOST, colors='deepskyblue', linewidths=0.8, linestyles='solid', levels=20)
        ax[i].contour(*CONTOURS_LOGPRIOR, colors='gray', linestyles='solid', alpha=0.5, linewidths=0.5)
        ax[i].contour(*MANIFOLD, levels=[Z], linewidths=3.0, colors=['black'], linestyles='solid')
        ax[i].contour(*MANIFOLD, levels=[Z-EPSILONS[i], Z+EPSILONS[i]], linewidths=1.0, linestyles='--', colors='black')
        ax[i].set_xticks([])
        ax[i].set_yticks([])
        ax[i].set_aspect('equal')
    fig.tight_layout()
    #plt.savefig("figures/contours_filamentary_epsilon_new_uniform.png", dpi=500)
    plt.show()