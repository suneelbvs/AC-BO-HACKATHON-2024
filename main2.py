from acquistion_functions import optimize_acquisition_function, expected_improvement_acquisition, random_acquisition, upper_confidence_bound_acquisition
from data_loaders.dataset import DataLoader
from models import XGBoostModel 
#from xgboost import Result
from data_loaders.tox21 import Tox21
#from util import results
from results import Result
from typing import List
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
#import mae, mse
from typing import Callable, Dict
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score

NUM_ACTIVE_LEARNING_LOOPS = 6

# Parameters for acquisition functions
xi_factor = 0.5  # for Expected Improvement
beta = 2.0  # for Upper Confidence Bound to prioritize exploration

def test_acquisition_function(
        *,
        loader: DataLoader,
        initial_dataset: np.ndarray,
        acquisition_function,
        acquisition_function_name: str,
) -> List[Result]:
    entire_dataset = np.arange(loader.size())
    random_dataset = initial_dataset
    results = []

    y = loader.y(entire_dataset)  # True values for the regression task

    # Initialization with a high starting value for minimization
    best_observed_value = np.inf

    for batch_num in range(1, NUM_ACTIVE_LEARNING_LOOPS + 1):
        print(f"[{acquisition_function_name}] random_dataset size: ", len(random_dataset))

        # Update model with active learning dataset
        model.fit(loader.x(random_dataset), loader.y(random_dataset))

        #create test by random_dataset - entire_dataset
        #test = np.setdiff1d(random_dataset, entire_dataset)

        # Predict on the entire dataset
        predictions = model.predict(loader.x(entire_dataset))
        
        # If you need the variance for some reason (e.g., for acquisition functions that use uncertainty)
        predictions, variance = model.predict(loader.x(entire_dataset), return_variance=True)

        # Convert predicted probabilities
        y_pred = (predictions)

        # Compute regression metrics
        mse = mean_squared_error(y, predictions)
        mae = mean_absolute_error(y, predictions)

        print(f"[{acquisition_function_name}] MSE: {mse}, MAE: {mae}")

        #Define a tolerance level for hits
        tolerance = 0.1  # Example tolerance - adjust based on your specific needs
         
        # Calculate the absolute difference and count the 'hits'
        num_hits = np.sum(np.abs(predictions - y) <= tolerance)

        # If you're also interested in computing the number of 'hits' as before:
        #num_hits = (y_pred & y).sum()
        print(f"[{acquisition_function_name}] Number of 'hits' after batch {batch_num}: {num_hits}")

        # Based on prediction magnitude (for significance)
        K = 100  # Number of top predictions to consider
        top_k_indices = np.argsort(-predictions)[:K]
        top_k_predictions = predictions[top_k_indices]
        top_k_true_values = y[top_k_indices]
        
        # Calculate MAE and MSE for the top K predictions
        top_k_mae = mean_absolute_error(top_k_true_values, top_k_predictions)
        top_k_mse = mean_squared_error(top_k_true_values, top_k_predictions)
        
        print(f"Top {K} MAE: {top_k_mae}")
        print(f"Top {K} MSE: {top_k_mse}")

        #save predictions to csv
        np.savetxt(f"predictions_{acquisition_function_name}_{batch_num}.csv", predictions, delimiter=",")

        # Update the best observed value if the new MSE is lower
        if mse < best_observed_value:
            best_observed_value = mse
            
        results.append(Result(
            batch_number=batch_num,
            num_hits=num_hits,
        ))

        # 3.5. get the top 100 candidates from the acquisition function and add them to the active learning dataset
        top_candidates = optimize_acquisition_function(acquisition_function=acquisition_function,
                                                        mean=predictions,
                                                        uncertainty=variance,
                                                        random_dataset=random_dataset,
                                                        best_observed_value=best_observed_value,
                                                        xi_factor=xi_factor,
                                                        beta=beta,
                                                        max_num_results=100)
        
        #print no of compounds in top_candidates
        #print(f"[Number of compounds in top candidates: ",len(top_candidates))
        
        # Update the dataset for the next iteration
        random_dataset = np.concatenate([random_dataset, top_candidates])
        return results

# Main routine
if __name__ == "__main__":
    loader = Tox21('./data/LD50_Zhu_ECFP.h5')
    print(f"Loaded {loader.name} dataset. num entries: {loader.size()}")

    # Initialize the model and the first dataset slice
    model = XGBoostModel()
    initial_dataset = np.random.choice(np.arange(loader.size()), size=200, replace=False)  # Ensuring diverse initial data

    # Define acquisition functions and their configurations
    acquisition_functions = [
        (expected_improvement_acquisition, "Expected Improvement"),
        (random_acquisition, "Random"),
        (upper_confidence_bound_acquisition, "Upper Confidence Bound"),
    ]

    optimization_results = {}
    for function, name in acquisition_functions:
        optimization_results[name] = test_acquisition_function(
            loader=loader,
            initial_dataset=initial_dataset,
            acquisition_function=function,
            acquisition_function_name=name)
    
    # Visualization and further analysis here
    # visualize_results(optimization_results)