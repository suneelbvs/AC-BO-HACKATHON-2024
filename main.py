from acquistion_functions import optimize, probability_of_improvement
from models import XGBoostModel
from data_loaders.tox21 import Tox21
from results import Result
import numpy as np

if __name__ == "__main__":
    NUM_ACTIVE_LEARNING_LOOPS = 5
    THRESHOLD = 0.02

    # 1: load the fingerprint + label data + set the threshold for the succes metric
    loader = Tox21()
    initial_dataset_size = loader.size()
    print(f"loaded {loader.name} dataset. num entries: {initial_dataset_size}")

    model = XGBoostModel()

    # 2: initialize the model + initialise the first 100 data points from the dataset
    active_dataset = np.arange(0, initial_dataset_size//20) # TODO: randomize the indices?

    acquisition_function = probability_of_improvement.probability_of_improvement
    results: [Result] = []

    # 3: carry out the active learning loop for N times
    for i in range(NUM_ACTIVE_LEARNING_LOOPS):
        print(f"active_dataset size: ", len(active_dataset))
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
        print(f"number of hits: {num_hits}", f"number of positive examples in unseen: {positive_class_count_in_unseen}")

        results.append(Result(
            batch_number=i,
            num_hits=num_hits,
            positive_class_count_in_unseen=positive_class_count_in_unseen
        ))

    
        # 3.5. get the top 100 candidates from the acquisition function and add them to the active learning dataset
        top_candidates = optimize.optimize(acquisition_function,
                                               mean,
                                               uncertainty,
                                               n_results=100,
                                               best_observed_value=best_observed_value)

        # 3.6 update the active learning dataset
        active_dataset = np.concatenate([active_dataset, top_candidates])
    
    # 4: save everything
    


