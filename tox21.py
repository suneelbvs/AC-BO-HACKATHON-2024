import pandas as pd
import numpy as np

class Tox21:
    def __init__(self, filepath):
        # Attempt to load the dataset and handle any errors that might occur.
        try:
            self.data = pd.read_hdf(filepath)
        except Exception as e:
            print(f"Failed to load data: {e}")
            self.data = None

        # Check for the existence of the 'fingerprint_ECFP' column and prepare it for use.
        if 'fingerprint_ECFP' in self.data.columns:
            # Assuming 'fingerprint_ECFP' contains the fingerprints in a suitable format.
            self.data["fps"] = self.data['fingerprint_ECFP'].apply(lambda x: np.array(x))
        else:
            print("'fingerprint_ECFP' not found in data. Please check the dataset.")
            self.data = None  # Consider setting to None or handling this case appropriately.

        self.name = "Tox21"

    def size(self):
        # Returns the size of the dataset.
        return len(self.data)
    
    def x(self, dataset_slice_indices: np.ndarray) -> np.ndarray:
        fps = self.data.iloc[dataset_slice_indices]["fps"].to_numpy()
        # Replace 'your_feature_space_dimension' with the actual dimension of your feature space
        if len(fps) == 0:
            return np.empty((0, 2048))
        return np.vstack(fps)

    def y(self, dataset_slice_indices: np.ndarray) -> np.ndarray:
        # Extracts labels/targets based on provided indices.
        if self.data is not None:
            return self.data.iloc[dataset_slice_indices]["Y"].to_numpy()
        else:
            return np.array([])  # Return an empty array if there's an issue with the data.
    
    #define y_pred as Y
    def y_true(self):
        return self.data["Y"]
    #y_true = self.data["Y"]