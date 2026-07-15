"""
Utility functions for Noisefloor.
"""
import numpy as np
from typing import Union, List

def coerce_to_array(data: Union[List[float], np.ndarray]) -> np.ndarray:
    """Ensure data is a 1D numpy array."""
    arr = np.asarray(data)
    if arr.ndim != 1:
        raise ValueError("Data must be a 1D array or list.")
    return arr
