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
    :return: The indices of the top candidates (that are not in the active dataset).
    """

    acquisition_function_values = acquisition_function(mean, uncertainty, **kwargs)

    # convert the array of values into an array of (value, index)
    # so when we remove the values that are in the active dataset, we can still keep track of the indices in the original array
    indices = np.arange(acquisition_function_values.shape[0])
    values_with_indices = np.column_stack((acquisition_function_values, indices))

    # filter for only indices that are NOT inside the active dataset (so we suggest new candidates)
    indices_not_in_active_dataset = indices[~np.isin(indices, active_dataset)]
    values_with_indices = values_with_indices[indices_not_in_active_dataset]

    if mode == "max":
        k_th_largest = np.partition(values_with_indices, -max_num_results, axis=0)[-max_num_results][0]
        best_rows = values_with_indices[values_with_indices[:, 0] >= k_th_largest]
    elif mode == "min":
        k_th_largest = np.partition(values_with_indices, max_num_results, axis=0)[max_num_results][0]
        best_rows = values_with_indices[values_with_indices[:, 0] <= k_th_largest]
    else:
        raise ValueError(f"Invalid mode: {mode}. Must be either 'max' or 'min'.")
    
    indices = best_rows[:, 1].astype(np.int64) # since the row was a float dtype (cause we concatenated with the values_with_indices), we need to convert the indices back to int
    num_candidates = min(max_num_results, len(indices)) # guards against the case where there are fewer than max_num_results candidates
    return np.random.choice(indices, num_candidates, replace=False)
