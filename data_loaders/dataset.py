import numpy as np

# This class will be used to load the dataset and perform feature extraction
# When we do bayesian optimization, we'll call the dataloader to get the x and y values for each dataset slice
class DataLoader:
    def __init__(self):
        self.name = ""
        # You should perform feature extraction and data cleaning here
        pass

    def size(self) -> int:
        pass

    # for a given slice of the dataset, we'll return the x values
    # these slices are indexes of the original dataset. e.g. if you pass in [1, 2, 9], The features at index 1, 2, and 9 will be returned
    def x(self, dataset_slice: np.ndarray) -> np.ndarray:
        pass

    # for a given slice of the dataset, we'll return the y values
    def y(self, dataset_slice: np.ndarray) -> np.ndarray:
        pass