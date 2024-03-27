""" This file contains an optimizer for the acquisition function. """
from typing import Callable
import numpy as np

def optimize(*,
             acquisition_function: Callable, mean: np.ndarray,
             uncertainty: np.ndarray, active_dataset: np.ndarray,
             max_num_results: 100, mode: str = "max", **kwargs) -> np.ndarray:
    """
    Optimize the acquisition function.
    :param acquisition_function: The acquisition function to optimize.
    :param mean: Mean of the surrogate model predictions.
    :param uncertainty: Uncertainty of the surrogate model predictions.
    :param active_dataset: the indices of the unseen data.
    :param max_num_results: The maximum number of results to return.
    :param mode: The optimization mode, either "max" or "min".
    :param kwargs: Additional keyword arguments for the acquisition function.
    :return: The optimized acquisition function.
    """
    acquisition_function_values = acquisition_function(mean, uncertainty, **kwargs)
    if mode == "max":
        k_th_largest = np.partition(acquisition_function_values, -max_num_results)[-max_num_results]
        indices = np.nonzero(acquisition_function_values >= k_th_largest)[0]
    elif mode == "min":
        k_th_largest = np.partition(acquisition_function_values, max_num_results)[max_num_results]
        indices = np.nonzero(acquisition_function_values <= k_th_largest)[0]
    else:
        raise ValueError(f"Invalid mode: {mode}. Must be either 'max' or 'min'.")
    
    indices_not_in_active_dataset = indices[~np.isin(indices, active_dataset)] # filter for only indices that are NOT inside the active dataset (so we suggest new candidates)
    num_candidates = min(max_num_results, len(indices_not_in_active_dataset)) # guards against the case where there are fewer than max_num_results candidates
    return np.random.choice(indices_not_in_active_dataset, num_candidates, replace=False)
