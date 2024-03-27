from xgboost import XGBRegressor
import numpy as np
SEED = 9582
class XGBoostModel:
    def __init__(self):
        self.models = []
        self.num_models = 5 # The reason why we want multiple models is so we can determine the variance for each prediction
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


    def fit(self, data_x, data_y):
        for i in range(self.num_models):
            self.params["random_state"] = SEED + i
            model = XGBRegressor(**self.params)
            model.fit(data_x, data_y)
            self.models.append(model)

    def predict(self, test_X):
        means = np.ndarray((self.num_models, test_X.shape[0]))
        variances = []
        for i in range(self.num_models):
            model = self.models[i]
            if i == 0:
                mean = model.predict(test_X)
            else:
                mean += model.predict(test_X)
        return variances