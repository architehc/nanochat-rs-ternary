"""nanochat: Ternary quantized code generation models.

Provides model zoo for downloading pre-trained models and utilities.
"""

from .model_zoo import ModelZoo, load_model

__version__ = "0.1.0"
__all__ = ["ModelZoo", "load_model"]
