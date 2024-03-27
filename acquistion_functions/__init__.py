""" A module containing a collection of different acquisition functions for Bayesian optimization. """
from .expected_improvement import expected_improvement
from .upper_confidence_bound import upper_confidence_bound

__all__ = ["expected_improvement", "upper_confidence_bound"]