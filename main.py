
# THIS IS A STUPID SKELETON PROJECT


from model import ModelTrainer
from dataset import Dataset
from params import xgboost_params

from tdc.utils import retrieve_label_name_list
from xgboost import XGBRegressor
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_squared_error
import numpy as np
from tdc.single_pred import Tox


# HACK: to get dataset
label_list = retrieve_label_name_list('Tox21')
data = Tox(name = 'Tox21', label_name = label_list[0])
split = data.get_split()
test = split['test']
train = split['train']


if __name__ == "__main__":
    # params
    N = 5 # number of active learning loops


    # 1: load the fingerprint + label data + set the threshold for the succes metric
    dataset = Dataset()

    # 2: initialize the model + initialise the first 100 data points from the dataset
    active_dataset = dataset.get_init_datapoints()

    # 3: carry out the active learning loop for N times
    for i in range(N):
        # 3.1 train the surrogate model from model.py on the active learning dataset
        trainer = ModelTrainer(
            X=active_dataset['X'],
            Y=active_dataset['Y'],
            params=xgboost_params,
        )
        trainer.train_model()
        # 3.2 run the surrogate model over the entire dataset and get predicted values for endpoints
    
        # 3.3. feed the predicted values into the acquisition function
    
        # 3.4 get the top 100 candidates from the acquisition function and add them to the active learning dataset
    
        # 3.5 compute the success metric (number of 'hits', positive examples that are above the threshold (or top 10% of entire dataset))

        # 3.6 repeat the loop
    
    # 4: save everything
    


