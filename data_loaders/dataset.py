import numpy as np
import pandas as pd

class DataLoader:
    def __init__(self):
        # You should perform feature extraction and data cleaning here
        pass

    def full_dataset(self) -> np.ndarray:
        pass

    # for a given slice of the dataset, we'll return the x values
    def x(self, dataset_slice: pd.DataFrame) -> np.ndarray:
        pass

    # for a given slice of the dataset, we'll return the y values
    def y(self, dataset_slice: pd.DataFrame) -> np.ndarray:
        pass
    