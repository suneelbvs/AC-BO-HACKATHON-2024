""" A module containing a collection of different acquisition functions for Bayesian optimization. """
from .expected_improvement import expected_improvement as expected_improvement_acquisition
from .upper_confidence_bound import upper_confidence_bound as upper_confidence_bound_acquisition
from .probability_of_improvement import probability_of_improvement as probability_of_improvement_acquisition
from .greedy import greedy as greedy_acquisition
from .random import random as random_acquisition
from .optimize import optimize as optimize_acquisition_function

__all__ = [
    "expected_improvement_acquisition",
    "upper_confidence_bound_acquisition",
    "probability_of_improvement_acquisition",
    "greedy_acquisition",
    "random_acquisition",
    "optimize_acquisition_function",
]
