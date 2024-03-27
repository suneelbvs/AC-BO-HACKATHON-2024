import numpy as np
from typing import Any

class Model:
    def __init__():
        pass

    def fit(train_x: np.ndarray, train_y: np.ndarray):
        pass

    def predict(test_x: np.ndarray) -> (np.ndarray, np.ndarray): # mean, variance
        pass


class ModelTrainer():
    def __init__(self, data_X: np.ndarray, data_Y: np.ndarray, model: Model, acquisition_function: Any):
        self.X = data_X
        self.Y = data_Y
        self.model = model

        self.active_dataset = np.empty()
        self.acquisition_function = acquisition_function

    def train_model(self):
        # TODO: perform iterative dataset selection
        self.model.fit(self.X, self.Y)