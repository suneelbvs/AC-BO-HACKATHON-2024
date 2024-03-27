""" This file contains the Upper Confidence Bound acquisition function. """
import numpy as np

def upper_confidence_bound(mean: np.ndarray, uncertainty: np.ndarray, beta: float, **kwargs) -> np.ndarray:
    """
    Compute the Upper Confidence Bound acquisition function.
    :param mean: Mean of the Gaussian process predictions.
    :param uncertainty: Uncertainty of the Gaussian process predictions.
    :param beta: Exploitation-exploration trade-off parameter.
    :return: Upper Confidence Bound values for each data point.
    """
    return mean + beta * uncertainty
