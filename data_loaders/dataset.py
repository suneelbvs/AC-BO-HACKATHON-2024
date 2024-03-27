import numpy as np
import pandas as pd

# This class will be used to load the dataset and perform feature extraction
# When we do bayesian optimization, we'll call the dataloader to get the x and y values for each dataset slice
class DataLoader:
    def __init__(self):
        self.name = ""
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
    