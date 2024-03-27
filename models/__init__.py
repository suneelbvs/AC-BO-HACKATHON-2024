"""
This module contains the implementations of the surrogate
models used in the Bayesian optimization project.
"""
from .gaussian_process import GaussianProcessModel
from .xgboost import XGBoostModel

__all__ = ["GaussianProcessModel", "XGBoostModel"]