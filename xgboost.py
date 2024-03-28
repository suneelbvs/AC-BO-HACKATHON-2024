from xgboost import XGBRegressor
from xgboost import XGBClassifier as XGBC
from typing import Tuple
import numpy as np
from .model import Model

from xgboost import XGBRegressor
import numpy as np

class XGBoostModel:
    def __init__(self, num_models=3, seed=9582):
        self.models = []
        self.num_models = num_models  # Allows for dynamic adjustment of the ensemble size
        self.seed = seed  # Base seed for reproducibility
        self.params = {
            'objective': 'reg:squarederror',  # Assuming a regression task; adjust as needed
            'reg_alpha': 0.0008774661176012108,
            'reg_lambda': 2.542812743920178,
            'colsample_bynode': 0.7839026197349153,
            'subsample': 0.8994226268096415,
            'eta': 0.04730766698056879,
            'max_depth': 3,
            'n_estimators': 500,
            'eval_metric': 'rmse',
            'n_jobs': -1,
            'learning_rate': 0.023,
            'random_state': seed  # Initial seed; will be updated per model in the ensemble
        }

    def fit(self, data_x, data_y):
        self.models.clear()  # Clear existing models if re-fitting
        for i in range(self.num_models):
            # Update the seed for each model to ensure diversity
            model_params = self.params.copy()
            model_params['random_state'] = self.seed + i
            model = XGBRegressor(**model_params)
            model.fit(data_x, data_y)
            self.models.append(model)

    def predict(self, test_x, return_variance=False):
        # Generate predictions from each model in the ensemble
        predictions = np.array([model.predict(test_x) for model in self.models])
        mean = np.mean(predictions, axis=0)
        variance = np.var(predictions, axis=0)
        
        # Return mean and variance if requested, else return mean only
        if return_variance:
            return mean, variance
        else:
            return mean

