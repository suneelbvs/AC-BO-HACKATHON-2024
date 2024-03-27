from acquistion_functions import optimize, probability_of_improvement
from models import XGBoostModel
from data_loaders.tox21 import Tox21


if __name__ == "__main__":
    NUM_ACTIVE_LEARNING_LOOPS = 5
    THRESHOLD = 0.5

    # 1: load the fingerprint + label data + set the threshold for the succes metric
    loader = Tox21()
    print(f"loaded {loader.name} dataset")

    model = XGBoostModel()

    full_dataset = loader.full_dataset()

    # 2: initialize the model + initialise the first 100 data points from the dataset
    initial_dataset_size = full_dataset.shape[0]
    active_dataset = full_dataset[:initial_dataset_size] # TODO: randomly shuffle the data?

    acquisition_function = probability_of_improvement

    # 3: carry out the active learning loop for N times
    for i in range(NUM_ACTIVE_LEARNING_LOOPS):
        # 3.0.1 update global parameters needed for acquisition functions
        best_observed_value = loader.y(active_dataset).max()

        # 3.1 train the surrogate model from model.py on the active learning dataset
        model.fit(loader.x(active_dataset), loader.y(active_dataset))

        # 3.2 run the surrogate model over the unseen dataset and get predicted values for endpoints
        unseen_data = full_dataset[~full_dataset.index.isin(active_dataset.index)]
        mean, uncertainty = model.predict(loader.x(unseen_data))

        # 3.3 compute the success metric (number of 'hits', positive examples that are above the threshold (or top 10% of entire dataset))K
        is_classified_positive = mean >= THRESHOLD
        unseen_data_y = loader.y(unseen_data)
        is_hit = is_classified_positive & (unseen_data_y == 1)

        num_hits = is_hit.sum()
        positive_class_count = loader.y(unseen_data).sum()
        print(f"number of hits: {num_hits}", f"total number of positive examples: {positive_class_count}")

    
        # 3.4. get the top 100 candidates from the acquisition function and add them to the active learning dataset
        top_candidate_indexes = optimize(acquisition_function,
                                               mean,
                                               uncertainty,
                                               best_observed_value=best_observed_value)

        # 3.6 update the active learning dataset
        active_dataset.append(unseen_data.iloc[top_candidate_indexes])
    
    # 4: save everything
    


