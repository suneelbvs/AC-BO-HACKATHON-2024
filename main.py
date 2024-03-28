from acquistion_functions import optimize_acquisition_function, probability_of_improvement_acquisition, greedy_acquisition, expected_improvement_acquisition, random_acquisition, upper_confidence_bound_acquisition
from data_loaders.ames import Ames
from data_loaders.dataset import DataLoader
from data_loaders.halflife import halflife
from data_loaders.ld50 import LD50
from data_loaders.tox21 import Tox21
from models import XGBoostModel
from models.gaussian_process import GaussianProcessModel
from models.model import Model
from models.random_forest import RandomForestModel
from results import RegressionHighestYResultTracker, RegressionNumBetterCandidatesResultTracker, RegressionNumOver90PercentileResultTracker, ResultTracker
import numpy as np
from typing import Callable
import math

from visualizers.visualize_model_progress import visualize_results

NUM_ACTIVE_LEARNING_LOOPS = 20
NUM_NEW_CANDIDATES_PER_BATCH = 20 # papers show that 4 new candidates is good (prob because collecting data is expensive)

# Parameters for acquisition functions
xi_factor = 0.5 # for Expected Improvement # TODO: tune
beta = 0.5 # for Upper Confidence Bound


def test_acquisition_function(
        *,
        model: Model,
        loader: DataLoader,
        initial_dataset: np.ndarray,
        acquisition_function: Callable,
        result_tracker: ResultTracker,
        acquisition_function_name: str,
):
    entire_dataset = np.arange(0, loader.size()) # PERF: if the dataset is much larger, this way of indexing could be slow
    active_dataset = initial_dataset

    y = loader.y(entire_dataset) == 1

    # 3: carry out the active learning loop for N times
    for batch_num in range(1, NUM_ACTIVE_LEARNING_LOOPS + 1):
        print(f"[{acquisition_function_name}] active_dataset size: ", len(active_dataset))
        # 3.1 update global parameters needed for acquisition functions
        best_observed_value = loader.y(active_dataset).max()

        # 3.2 train the surrogate model from model.py on the active learning dataset
        model.fit(loader.x(active_dataset), loader.y(active_dataset))

        # 3.3 run the surrogate model over the unseen dataset and get predicted values for endpoints
        mean, uncertainty = model.predict(loader.x(entire_dataset))
    
        # 3.4. get the top candidates from the acquisition function and add them to the active learning dataset
        optimize_acquisition_function_args = {
            "acquisition_function":acquisition_function,
            "mean": mean,
            "uncertainty": uncertainty,
            "active_dataset": active_dataset,
            "max_num_results": NUM_NEW_CANDIDATES_PER_BATCH,
            "best_observed_value": best_observed_value,
            "xi_factor": xi_factor,
            "beta": beta,
        }
        top_candidates = optimize_acquisition_function(**optimize_acquisition_function_args)

        # 3.5 compute the success metric
        result_tracker_args = {
            "mean": mean,
            "y": y,
            "loader": loader,
            "active_dataset": active_dataset,
            "top_candidates": top_candidates,
            "batch_num": batch_num,
            "acquisition_function_name": acquisition_function_name,
        }

        result_tracker.add_result(**result_tracker_args)

        # 3.6 update the active learning dataset
        active_dataset = np.concatenate([active_dataset, top_candidates])

if __name__ == "__main__":
    # 1: load the fingerprint + label data + set the threshold for the succes metric
    loader = LD50()
    # loader = Ames()
    initial_dataset_size = loader.size()
    print(f"loaded {loader.name} dataset. num entries: {initial_dataset_size}, num_y=1:{loader.y(np.arange(initial_dataset_size)).sum()}")

    # 2: initialize the model + initialise the first 100 data points from the dataset
    # papers show that the first 6% of the dataset is a good starting point
    active_dataset = np.arange(0, math.ceil(initial_dataset_size*0.06)) # TODO: randomize the indices?

    acquisition_functions = [
        (expected_improvement_acquisition, "Expected Improvement"),
        (greedy_acquisition, "Greedy"),
        (probability_of_improvement_acquisition, "Probability of Improvement"),
        (random_acquisition, "Random"),
        (upper_confidence_bound_acquisition, "Upper Confidence Bound"),
    ] 

    # result_creator = get_regression_results_highest_y
    result_tracker = RegressionNumOver90PercentileResultTracker()

    # model = XGBoostModel()
    model = RandomForestModel()
    for acquisition_function, name in acquisition_functions:
        test_acquisition_function(
            model=model,
            loader=loader,
            initial_dataset=np.copy(active_dataset),
            acquisition_function=acquisition_function,
            result_tracker=result_tracker,
            acquisition_function_name=name)

    # 4: save everything
    visualize_results(result_tracker, loader.name, model.name)