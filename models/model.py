from typing import Any
import numpy as np

class Model:
    name: int

    def __init__(self):
        pass

    def fit(self, train_x: np.ndarray, train_y: np.ndarray):
        pass

    def predict(self, test_x: np.ndarray) -> tuple[np.ndarray, np.ndarray]: # mean, variance
        pass