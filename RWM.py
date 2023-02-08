import numpy as np
from numpy import zeros, log, eye, vstack
from numpy.random import default_rng, randint
from scipy.stats import multivariate_normal as MVN

def RWM(x0, s, N, logpi, rng=None):
    """Simple RWM function with proposal N(x, (s**2)*I)"""
    if rng is None:
        rng = default_rng(seed=randint(low=1000, high=9999))
    samples = x = x0                                   # Accepted samples will be stored here
    acceptances, logpx, d = zeros(N), logpi(x), len(x) # Accepted (=1), log(pi(x)), dimensionality
    logu = log(rng.uniform(size=N))                                # Used for Accept/Reject step

    for i in range(N):
        y = x + s*rng.normal(size=d)   # Sample candidate
        logpy = logpi(y)               # Compute its log density
        if logu[i] <= logpy - logpx:
            x, logpx, acceptances[i] = y, logpy, 1 # Accept! 
        samples = vstack((samples, x)) # Add sample to storage
    return samples[1:], acceptances