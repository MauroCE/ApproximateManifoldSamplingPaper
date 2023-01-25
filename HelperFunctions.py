import numpy as np
from numpy import logspace, vstack, array, mean
from arviz import convert_to_dataset, ess


def generate_powers_of_ten(max_exponent, min_exponent):
    """E.g. generate_powers_of_ten(2, -1) will return 100, 10, 0, 0.1."""
    number_of_powers = max_exponent + abs(min_exponent) + 1
    return logspace(start=max_exponent, stop=min_exponent, num=number_of_powers, endpoint=True)


def compute_arviz_miness_runtime(chains, times):
    """Computes minESS/runtime. Expects chains=[samples, samples, ...] and times = [time, time, ...]."""
    assert np.all([chain.shape == chains[0].shape for chain in chains]), "Chains must have same dimensions."
    n_samples = len(chains[0])
    stacked = vstack([chain.reshape(1, n_samples, -1) for chain in chains])
    dataset = convert_to_dataset(stacked)
    return min(array(ess(dataset).to_array()).flatten()) / mean(times)