""" This file contains the Expected Improvement acquisition function. """
import numpy as np
from scipy.stats import norm

def expected_improvement(mean: np.ndarray, uncertainty: np.ndarray, best_observed_value: float, xi_factor: float, **kwargs) -> np.ndarray:
    """
    Compute the Expected Improvement acquisition function.
    :param mean: Mean of the Gaussian process predictions.
    :param uncertainty: Uncertainty of the Gaussian process predictions.
    :param best_observed_value: Best observed target value so far.
    :param xi_factor: Exploitation-exploration trade-off parameter.
    :return: Expected Improvement values for each data point.
    """
    z = mean - best_observed_value * xi_factor
    return z * norm.cdf(z / uncertainty) + uncertainty * norm.pdf(z / uncertainty)
