from collections import defaultdict
from dataclasses import dataclass
from typing import DefaultDict, List
import numpy as np
from data_loaders.dataset import DataLoader

@dataclass
class Result:
    batch_number: int
    y_axis: int

class ResultTracker:
    y_label: str
    results: DefaultDict[str, List[Result]] = defaultdict(list) # maps acquisition function name to results

    def add_result(self, **kwargs):
        pass

# THRESHOLD = 0.001 # we are using a dynamic threshold now (see below)

class ClassificationResultTracker(ResultTracker):
    y_label = "ClassificationResultTracker"

    def add_result(self, *, mean: np.ndarray, y: np.ndarray, batch_num:int, acquisition_function_name: str, **kwargs):
        THRESHOLD = np.sort(mean)[-309]
        y_hat = mean >= THRESHOLD
        is_hit = y_hat & y
        num_hits = is_hit.sum()

        print(f"[{acquisition_function_name}] number of hits: {num_hits}")

        self.results[acquisition_function_name].append(Result(
            batch_number=batch_num,
            y_axis=num_hits,
        ))

class RegressionHighestYResultTracker(ResultTracker):
    y_label = "Best Actual Y"

    def add_result(self, *, loader: DataLoader, active_dataset: np.ndarray, top_candidates: np.ndarray, batch_num: int, acquisition_function_name: str, **kwargs):
        active_dataset_ys = loader.y(active_dataset)
        top_candidates_ys = loader.y(top_candidates)
        # The number of hits is the number of top_candidates suggested that have a y that is greater than in the current active dataset
        best_actual_y = max(active_dataset_ys.max(), top_candidates_ys.max())

        print(f"[{acquisition_function_name}] batch_num: {batch_num} best_actual_y: {best_actual_y}")
        self.results[acquisition_function_name].append(Result(
            batch_number=batch_num,
            y_axis=best_actual_y,
        ))

# given the candidates in the active dataset, how many NEW candidates (that are going to be added in the next batch) are better
class RegressionNumBetterCandidatesResultTracker(ResultTracker):
    y_label = "Num Better Candidates"
    def __init__(self):
        super().__init__()
        self.cumulative_num_better_candidates = defaultdict(int)

    def add_result(self, *, loader: DataLoader, active_dataset: np.ndarray, top_candidates: np.ndarray, batch_num:int, acquisition_function_name: str, **kwargs):
        active_dataset_ys = loader.y(active_dataset)
        top_candidates_ys = loader.y(top_candidates)

        num_candidates_better_than_active_dataset = np.count_nonzero(top_candidates_ys > active_dataset_ys.max())
        self.cumulative_num_better_candidates[acquisition_function_name] += num_candidates_better_than_active_dataset
        print(f"[{acquisition_function_name}] batch_num: {batch_num}, num_better_candidates: {self.cumulative_num_better_candidates[acquisition_function_name]}")

        self.results[acquisition_function_name].append(Result(
            batch_number=batch_num,
            y_axis=self.cumulative_num_better_candidates[acquisition_function_name],
        ))

class RegressionNumOver200ResultTracker(ResultTracker):
    y_label = "Num Over 200"

    def add_result(self, *, loader: DataLoader, active_dataset: np.ndarray, top_candidates: np.ndarray,
                     batch_num:int, acquisition_function_name: str, **kwargs):
        active_dataset_ys = loader.y(active_dataset)
        top_candidates_ys = loader.y(top_candidates)

        num_candidates_over_200 = np.count_nonzero(active_dataset_ys > 200) + np.count_nonzero(top_candidates_ys > 200)
        print(f"[{acquisition_function_name}] batch_num: {batch_num}, num_over_200: {num_candidates_over_200}")

        self.results[acquisition_function_name].append(Result(
            batch_number=batch_num,
            y_axis=num_candidates_over_200,
        ))


class RegressionNumOver90PercentileResultTracker(ResultTracker):
    y_label = "Num Over 90 percentile"

    def __init__(self):
        self.ninty_percentile = None

    def add_result(self, *, loader: DataLoader, active_dataset: np.ndarray, top_candidates: np.ndarray,
                     batch_num:int, acquisition_function_name: str, **kwargs):
        
        if self.ninty_percentile is None:
            sorted_y_values = np.sort(loader.y(np.arange(loader.size()))) # PERF: do we need to sort?
            self.ninty_percentile = np.percentile(sorted_y_values, 90)

        active_dataset_ys = loader.y(active_dataset)
        top_candidates_ys = loader.y(top_candidates)

        num_candidates_over_200 = np.count_nonzero(active_dataset_ys > self.ninty_percentile) + np.count_nonzero(top_candidates_ys > self.ninty_percentile)
        print(f"[{acquisition_function_name}] batch_num: {batch_num}, num_over_200: {num_candidates_over_200}")

        self.results[acquisition_function_name].append(Result(
            batch_number=batch_num,
            y_axis=num_candidates_over_200,
        ))