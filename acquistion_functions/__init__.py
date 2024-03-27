""" A module containing a collection of different acquisition functions for Bayesian optimization. """
from .expected_improvement import expected_improvement
from .upper_confidence_bound import upper_confidence_bound
from .probability_of_improvement import probability_of_improvement

__all__ = ["expected_improvement", "upper_confidence_bound"]