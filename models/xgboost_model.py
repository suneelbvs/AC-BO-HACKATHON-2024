from xgboost import XGBRegressor
import numpy as np
from .model import Model

SEED = 9582
class XGBoostModel(Model):
    def __init__(self):
        self.models = []
        self.num_models = 3 # The reason why we want multiple models is so we can determine the variance for each prediction
        self.params = {
            'reg_alpha': 0.0008774661176012108,
            'reg_lambda': 2.542812743920178,
            'colsample_bynode': 0.7839026197349153,
            'subsample': 0.8994226268096415, 
            # subsample=1,
            'eta': 0.04730766698056879, 
            'max_depth': 3, 
            'n_estimators': 500,
            'eval_metric': 'rmse',
            'n_jobs': -1,
            'learning_rate':0.023,
        }
        self.name = "XGBoost"


    def fit(self, data_x, data_y):
        self.models = []
        for i in range(self.num_models):
            # Init a random seed for each model so we can get multiple predictions and calculate the variance of our model
            self.params["random_state"] = SEED + i
            model = XGBRegressor(**self.params)
            model.fit(data_x, data_y)
            self.models.append(model)

    def predict(self, test_x):
        predictions = np.empty((self.num_models, test_x.shape[0]))
        for i in range(self.num_models):
            predictions[i] = self.models[i].predict(test_x)
        mean = np.mean(predictions, axis=0)
        variance = np.var(predictions, axis=0)
        return mean, variance