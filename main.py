from models import XGBoostModel
from data_loaders.tox21 import Tox21


if __name__ == "__main__":
    NUM_ACTIVE_LEARNING_LOOPS = 5
    # 1: load the fingerprint + label data + set the threshold for the succes metric
    data_loader = Tox21()
    model = XGBoostModel()

    # 2: initialize the model + initialise the first 100 data points from the dataset
    initial_dataset_size = data_loader.full_dataset().shape[0]
    active_dataset = data_loader.full_dataset()[:initial_dataset_size] # TODO: randomly shuffle the data?

    # 3: carry out the active learning loop for N times
    for i in range(NUM_ACTIVE_LEARNING_LOOPS):
        # 3.1 train the surrogate model from model.py on the active learning dataset
        model.fit(data_loader.x(active_dataset), data_loader.y(active_dataset))

        # 3.2 run the surrogate model over the entire dataset and get predicted values for endpoints
        mean, uncertainty = model.predict(data_loader.full_dataset())
        print(mean, uncertainty)
    
        # 3.3. feed the predicted values into the acquisition function
    
        # 3.4 get the top 100 candidates from the acquisition function and add them to the active learning dataset
    
        # 3.5 compute the success metric (number of 'hits', positive examples that are above the threshold (or top 10% of entire dataset))

        # 3.6 repeat the loop
    
    # 4: save everything
    


