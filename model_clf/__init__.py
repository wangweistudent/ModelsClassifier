# model_afs/__init__.py

__version__ = "0.0.1"
__all__ = ["ModelsClassifier", "get_model_param_grid"]

from .model_param_grid import ModelsClassifier
from .model_config_loader import get_model_param_grid
