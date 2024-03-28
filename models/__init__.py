"""
This module contains the implementations of the surrogate
models used in the Bayesian optimization project.
"""
from .gaussian_process_model import GaussianProcessModel
from .xgboost_model import XGBoostModel

__all__ = ["GaussianProcessModel", "XGBoostModel"]