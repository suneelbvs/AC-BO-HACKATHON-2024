from dataclasses import dataclass
import numpy as np
from data_loaders.dataset import DataLoader

@dataclass
class Result:
    batch_number: int
    y_axis: int

# THRESHOLD = 0.001 # we are using a dynamic threshold now (see below)

def get_classification_results(*, mean: np.ndarray, y: np.ndarray, batch_num:int, acquisition_function_name: str, **kwargs) -> Result:
    THRESHOLD = np.sort(mean)[-309]
    y_hat = mean >= THRESHOLD
    is_hit = y_hat & y
    num_hits = is_hit.sum()

    print(f"[{acquisition_function_name}] number of hits: {num_hits}")

    return Result(
        batch_number=batch_num,
        y_axis=num_hits,
    )

def get_regression_results_highest_y(*, loader: DataLoader, active_dataset: np.ndarray, top_candidates: np.ndarray, 
                                       batch_num: int, acquisition_function_name: str, **kwargs) -> Result:
    active_dataset_ys = loader.y(active_dataset)
    top_candidates_ys = loader.y(top_candidates)
    # The number of hits is the number of top_candidates suggested that have a y that is greater than in the current active dataset
    best_actual_y = max(active_dataset_ys.max(), top_candidates_ys.max())

    print(f"[{acquisition_function_name}] batch_num: {batch_num} best_actual_y: {best_actual_y}")
    return Result(
        batch_number=batch_num,
        y_axis=best_actual_y,
    )

def get_regression_results_num_better_candidates(*, loader: DataLoader, active_dataset: np.ndarray, top_candidates: np.ndarray,
                                                 batch_num:int, acquisition_function_name: str, **kwargs) -> Result:
    active_dataset_ys = loader.y(active_dataset)
    top_candidates_ys = loader.y(top_candidates)

    num_candidates_better_than_active_dataset = np.count_nonzero(top_candidates_ys > active_dataset_ys.max())
    print(f"[{acquisition_function_name}] batch_num: {batch_num}, num_better_candidates: {len(num_candidates_better_than_active_dataset)}")

    return Result(
        batch_number=batch_num,
        y_axis=num_candidates_better_than_active_dataset,
    )