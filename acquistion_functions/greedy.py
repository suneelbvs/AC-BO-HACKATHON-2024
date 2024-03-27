""" This file contains the Greedy acquisition function. """
import numpy as np

def greedy(mean: np.ndarray, uncertainty: np.ndarray, **kwargs) -> np.ndarray:
    """
    Compute the Greedy acquisition function.
    This function corresponds to the mean.
    :param mean: Mean of the Gaussian process predictions.
    :param uncertainty: Uncertainty of the Gaussian process predictions.
    :return: The value of the greedy acquisition function for each data point.
    """
    return mean
