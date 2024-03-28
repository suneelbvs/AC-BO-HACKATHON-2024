""" This file contains is the main entry point of the program."""
from typing import Callable
import numpy as np
from acquistion_functions import (
    optimize_acquisition_function,
    probability_of_improvement_acquisition,
    greedy_acquisition,
    expected_improvement_acquisition,
    random_acquisition,
    upper_confidence_bound_acquisition,
)
from models.model import Model
from models import (
    GaussianProcessModel,
    XGBoostModel,
)
from data_loaders.dataset import DataLoader
from data_loaders import (
    LD50,
    Ames,
    Halflife,
    Tox21,
)

NUM_ACTIVE_LEARNING_LOOPS = 60
NUM_NEW_CANDIDATES_PER_BATCH = 4

def run_optimization(
    model_class: Model,
    data_loader: DataLoader,
    acquisition_function: Callable,
) -> list[int]:
    """
    Function that runs the optimization for a single configuration
    of model, data loader, and acquisition function.
    :param model_class: The model class to use.
    :param data_loader: The data loader to use.
    :param acquisition_function: The acquisition function to use.
    """
    loader = data_loader()
    active_indices = np.random.choice(loader.size(), NUM_NEW_CANDIDATES_PER_BATCH, replace=False)

    result = []
    for _ in range(NUM_ACTIVE_LEARNING_LOOPS):
        active_x = loader.x(active_indices)
        active_y = loader.y(active_indices)
        model = model_class()
        model.fit(active_x, active_y)
        mean, uncertainty = model.predict(loader.x())
        filtered_mean = mean[~np.isin(np.arange(loader.size()), active_indices)]
        filtered_uncertainty = uncertainty[~np.isin(np.arange(loader.size()), active_indices)]
        new_indices = optimize_acquisition_function(
            acquisition_function=acquisition_function,
            mean=filtered_mean,
            uncertainty=filtered_uncertainty,
            active_dataset=loader.x(active_indices),
            max_num_results=NUM_NEW_CANDIDATES_PER_BATCH,
            best_observed_value=np.max(active_y),
            xi_factor=1.01,
            beta=1,
        )
        active_indices = np.concatenate([active_indices, new_indices])
        result.append(np.sum(loader.y(active_indices) > loader.hit_threshold()))
        del model
    return result

def main():
    """ Main function to run the optimization. """
    # Prepare the data loaders, models, and acquisition functions
    data_loaders = [
        LD50,
        Ames,
        Halflife,
        Tox21
    ]
    models = [
        GaussianProcessModel,
        XGBoostModel
    ]
    acquisition_functions = [
        probability_of_improvement_acquisition,
        greedy_acquisition,
        expected_improvement_acquisition,
        random_acquisition,
        upper_confidence_bound_acquisition,
    ]

    # Loop over all combinations
    with open('results.csv', 'a', encoding='utf-8') as output_file:
        for data_loader in data_loaders:
            for model_class in models:
                for acquisition_function in acquisition_functions:
                    print(f"Testing {model_class.__name__}",
                          f"with {acquisition_function.__name__}",
                          f"on {data_loader.__name__}")
                    for i in range(5): # Run 10 optimization runs per setup
                        print(f"Run {i}", end='\r')
                        result = run_optimization(
                            model_class=model_class,
                            data_loader=data_loader,
                            acquisition_function=acquisition_function,
                        )
                        result_string = ','.join(map(str, result))
                        output_file.write(f"{data_loader.__name__},{model_class.__name__},"
                                          f"{data_loader.fingerprint},"
                                          f"{acquisition_function.__name__},{result_string}\n")

if __name__ == "__main__":
    main()
