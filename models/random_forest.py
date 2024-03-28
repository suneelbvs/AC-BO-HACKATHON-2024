import numpy as np
from .model import Model
from sklearn.ensemble import RandomForestRegressor
import numpy as np
from joblib import Parallel, delayed

class RandomForestModel(Model):
    name = "random forest"

    def fit(self, data_x, data_y):
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.model.fit(data_x, data_y)

    def predict(self, test_x):
        mean = self.model.predict(test_x) # PERF: maybe we can remove this call and just use individual_predictions below?
        individual_predictions = np.array([tree.predict(test_x) for tree in self.model.estimators_]) # PERF: parallelize this
        variance = np.var(individual_predictions, axis=0)

        return mean, variance
