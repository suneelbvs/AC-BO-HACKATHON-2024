""" This file contains an optimizer for the acquisition function. """
from typing import Callable
import numpy as np

def optimize(acquisition_function: Callable, mean: np.ndarray,
             uncertainty: np.ndarray, n_results: 100, mode: str = "max", **kwargs) -> np.ndarray:
    """
    Optimize the acquisition function.
    :param acquisition_function: The acquisition function to optimize.
    :param mean: Mean of the surrogate model predictions.
    :param uncertainty: Uncertainty of the surrogate model predictions.
    :param n_results: The number of results to return.
    :param mode: The optimization mode, either "max" or "min".
    :param kwargs: Additional keyword arguments for the acquisition function.
    :return: The optimized acquisition function.
    """
    acquisition_function_values = acquisition_function(mean, uncertainty, **kwargs)
    if mode == "max":
        k_th_largest = np.partition(acquisition_function_values, -n_results)[-n_results]
        indices = np.nonzero(acquisition_function_values >= k_th_largest)[0]
    elif mode == "min":
        k_th_largest = np.partition(acquisition_function_values, n_results)[n_results]
        indices = np.nonzero(acquisition_function_values <= k_th_largest)[0]
    else:
        raise ValueError(f"Invalid mode: {mode}. Must be either 'max' or 'min'.")
    return np.random.choice(indices, n_results, replace=False)
