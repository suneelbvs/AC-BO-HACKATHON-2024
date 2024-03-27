from acquistion_functions import optimize_acquisition_function, probability_of_improvement_acquisition, greedy_acquisition, expected_improvement_acquisition, random_acquisition, upper_confidence_bound_acquisition
from data_loaders.dataset import DataLoader
from models import XGBoostModel
from data_loaders.tox21 import Tox21
from results import Result
import numpy as np
from typing import Callable, Dict

from visualizers.visualize_model_progress import visualize_hits

NUM_ACTIVE_LEARNING_LOOPS = 5
THRESHOLD = 0.02

# Parameters for acquisition functions
xi_factor = 0.5 # for Expected Improvement
beta = 0.5 # for Upper Confidence Bound

def test_acquisition_function(
        *,
        loader: DataLoader,
        initial_dataset: np.ndarray,
        acquisition_function: Callable,
        acquisition_function_name: str,
) -> [Result]:
    active_dataset = initial_dataset
    results: [Result] = []

    # 3: carry out the active learning loop for N times
    for batch_num in range(1, NUM_ACTIVE_LEARNING_LOOPS + 1):
        print(f"[{acquisition_function_name}] active_dataset size: ", len(active_dataset))
        # 3.1 update global parameters needed for acquisition functions
        best_observed_value = loader.y(active_dataset).max()

        # 3.2 train the surrogate model from model.py on the active learning dataset
        model.fit(loader.x(active_dataset), loader.y(active_dataset))

        # 3.3 run the surrogate model over the unseen dataset and get predicted values for endpoints
        unseen_data = np.setdiff1d(np.arange(0, initial_dataset_size), active_dataset)
        mean, uncertainty = model.predict(loader.x(unseen_data))

        # 3.4 compute the success metric (number of 'hits', positive examples that are above the threshold (or top 10% of entire dataset))K
        is_classified_positive = mean >= THRESHOLD
        unseen_data_y = loader.y(unseen_data)
        is_hit = is_classified_positive & (unseen_data_y == 1)

        num_hits = is_hit.sum()
        positive_class_count_in_unseen = loader.y(unseen_data).sum()
        print(f"[{acquisition_function_name}] number of hits: {num_hits}", f"number of positive examples in unseen: {positive_class_count_in_unseen}")

        results.append(Result(
            batch_number=batch_num,
            num_hits=num_hits,
            positive_class_count_in_unseen=positive_class_count_in_unseen,
        ))

    
        # 3.5. get the top 100 candidates from the acquisition function and add them to the active learning dataset
        additional_args = {
            "best_observed_value": best_observed_value,
            "xi_factor": xi_factor,
            "beta": beta,
        }
        top_candidates = optimize_acquisition_function(acquisition_function,
                                                        mean,
                                                        uncertainty,
                                                        n_results=100,
                                                        **additional_args)

        # 3.6 update the active learning dataset
        active_dataset = np.concatenate([active_dataset, top_candidates])
    return results

if __name__ == "__main__":
    # 1: load the fingerprint + label data + set the threshold for the succes metric
    loader = Tox21()
    initial_dataset_size = loader.size()
    print(f"loaded {loader.name} dataset. num entries: {initial_dataset_size}")

    # 2: initialize the model + initialise the first 100 data points from the dataset
    active_dataset = np.arange(0, initial_dataset_size//20) # TODO: randomize the indices?

    acquisition_functions = [
        (expected_improvement_acquisition, "Expected Improvement"),
        (greedy_acquisition, "Greedy"),
        (probability_of_improvement_acquisition, "Probability of Improvement"),
        (random_acquisition, "Random"),
        (upper_confidence_bound_acquisition, "Upper Confidence Bound"),
    ] 

    model = XGBoostModel()
    optimization_results: Dict[str, Result] = {}
    for acquisition_function, name in acquisition_functions:
        optimization_results[name] = test_acquisition_function(
            loader=loader,
            initial_dataset=np.copy(active_dataset),
            acquisition_function=acquisition_function,
            acquisition_function_name=name)
    # 4: save everything
    visualize_hits(optimization_results)

