from typing import Any
import numpy as np

class Model:

    def __init__(self):
        pass

    def fit(self, train_x: np.ndarray, train_y: np.ndarray):
        pass

    def predict(self, test_x: np.ndarray) -> tuple[np.ndarray, np.ndarray]: # mean, variance
        pass


class ModelTrainer():

    def __init__(self, data_x: np.ndarray, data_y: np.ndarray, model: Model, acquisition_function: Any):
        self.x = data_x
        self.y = data_y
        self.model = model

        self.active_dataset = []
        self.acquisition_function = acquisition_function

    def train_model(self):
        # TODO: perform iterative dataset selection
        # self.model.fit(self.X, self.Y)
        pass
