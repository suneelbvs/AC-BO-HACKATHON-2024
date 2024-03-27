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


# HACK: below are model hypers for XGBoost
SEED = 9582
N_FOLD = 5

model_params = {
    'reg_alpha': 0.0008774661176012108,
    'reg_lambda': 2.542812743920178,
    'colsample_bynode': 0.7839026197349153,
    'subsample': 0.8994226268096415, 
    # subsample=1,
    'eta': 0.04730766698056879, 
    'max_depth': 3, 
    'n_estimators': 500,
    'random_state': SEED,
    'eval_metric': 'rmse',
    'n_jobs': -1,
    'learning_rate':0.023,
}


class ModelTrainer():
    def __init__(self, data_X, data_Y, params):
        self.params = params
        self.X = data_X
        self.Y = data_Y
        self.model = None
    
    # Get the trained model        
    def get_model(self):
        return self.model
    
    # TODO: here also add a way to add GP as a model
    def _init_model(self):
        new_params = self.params.copy()
        self.model = XGBRegressor(**new_params)

    def train_model(self):
        # TODO: might need to edit this once GP is added
        # Fit the model with train x and train y
        self.model.fit(self.X, self.Y, verbose=0)

# HACK: helper functions
# might delete them later
def gen_feats(dataset):
    dataset["num_oxygen"] = dataset["Drug"].str.count("O")
    return dataset

def fit_model(train_dataset):
    train_x = train_dataset["num_oxygen"]
    train_y = train_dataset["Y"]
    trainer = ModelTrainer(train_x, train_y, model_params)
    res = trainer.train_model()
    print(res["avg_rmse"])