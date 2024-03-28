import numpy as np

class DataLoader:

    """
    This is an abstract class for different dataloaders.
    This class will be used to load the dataset and perform feature extraction.
    When we do bayesian optimization, we'll call the dataloader
    to get the x and y values for each dataset slice.
    """

    def __init__(self):
        """ You should perform feature extraction and data cleaning here """
        pass

    def size(self) -> int:
        """ Returns the size of the dataset """
        pass
 
    def x(self, dataset_slice: slice | np.ndarray = None) -> np.ndarray:
        """
        For a given slice of the dataset, we'll return the x values.
        These slices are indexes of the original dataset. e.g. if you pass in [1, 2, 9],
        The features at index 1, 2, and 9 will be returned.
        """
        pass

    def y(self, dataset_slice: slice | np.ndarray = None) -> np.ndarray:
        """
        For a given slice of the dataset, we'll return the y values.
        """
        pass
