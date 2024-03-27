""" This file contains the Probability of Improvement acquisition function. """
import numpy as np
from scipy.stats import norm

def probability_of_improvement(mean: np.ndarray, uncertainty: np.ndarray, best_observed_value: float, **kwargs) -> np.ndarray:
    """
    Compute the Probability of Improvement acquisition function.
    :param mean: Mean of the Gaussian process predictions.
    :param uncertainty: Uncertainty of the Gaussian process predictions.
    :param best_observed_value: Best observed target value so far.
    :return: Probability of Improvement values for each data point.
    """
    z = (mean - best_observed_value) / uncertainty
    return norm.cdf(z)
