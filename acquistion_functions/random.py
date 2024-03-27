""" This file contains the Greedy acquisition function. """
import numpy as np

def random(mean: np.ndarray, uncertainty: np.ndarray, **kwargs) -> np.ndarray:
    """
    Compute the Random acquisition function.
    This function actually returns a vector of zeros since all values have the same probability.
    :param mean: Mean of the Gaussian process predictions.
    :param uncertainty: Uncertainty of the Gaussian process predictions.
    :return: Zeros for each data point.
    """
    return np.zeros_like(mean)
