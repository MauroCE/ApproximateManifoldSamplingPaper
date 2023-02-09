import numpy as np
from numpy import logspace, vstack, array, mean, meshgrid, arange
from arviz import convert_to_dataset, ess


def generate_powers_of_ten(max_exponent, min_exponent):
    """Generates an array with powers of tens starting from `10^max_exponent` up to `10^min_exponent`.
    Can be used to generate array of epsilon values. For instance `generate_powers_of_ten(2, -1)` will 
    return `np.array([100, 10, 0, 0.1])`.

    Arguments:

    :param max_exponent: Maximum exponent of `10`. For instance `max_exponent=2` means sequence starts from `100`.
    :type max_exponent: int

    :param min_exponent: Minimum exponent of `10`. For instance `min_exponent=-1` means sequence ends at `0.1`.
    :type min_exponent: int

    Returns:
    
    :param array_of_powers_of_ten: Array containing powers of `10` from `10^max_exponent` to `10^min_exponent` in 
                                   descending order.
    :type array_of_powers_of_ten: ndarray
    """
    number_of_powers = max_exponent + abs(min_exponent) + 1
    return logspace(start=max_exponent, stop=min_exponent, num=number_of_powers, endpoint=True)


def compute_arviz_miness_runtime(chains, times):
    """Computes bulk-minESS across components divided by total run time. Bulk because it is computed 
    across a number of chains, equals to `n_chains = len(chains) = len(times)`.
    
    Arguments: 

    :param chains: List where each element is a numpy array of shape `(n, d)` where `n` is the number of 
                   samples and `d` is the dimension of the ambient space. This should be the output of `THUG` or
                   `CRWM` algorithms.
    :type chains: list
    
    :param times: List where each element is a float representing the total runtime taken by the algorithm to produce
                  the corresponding samples. In other words `times[i]` is the time taken to compute `samples[i]`.
    :type times: list

    Returns:

    :param bulkminESS_runtime: Bulk-min-ESS across multiple chains computed using arviz package.
    :type bulkminESS_runtime: float
    """
    assert np.all([chain.shape == chains[0].shape for chain in chains]), "Chains must have same dimensions."
    n_samples = len(chains[0])
    stacked = vstack([chain.reshape(1, n_samples, -1) for chain in chains])
    dataset = convert_to_dataset(stacked)
    return min(array(ess(dataset).to_array()).flatten()) / mean(times)


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