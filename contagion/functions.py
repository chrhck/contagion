"""
This module provides some custom functions and classes
"""
import logging
import scipy.stats  # type: ignore
from scipy.special import gamma as Gamma_Func
import numpy as np  # type: ignore

from .config import config

_log = logging.getLogger(__name__)

class GammaMaxVal(scipy.stats.rv_continuous):
    """
    class: GammaMaxVal
    Helper class to construct a custom gamma distribution
    Parameters:
        -float custom_shape:
            The shape parameter
        -float norm:
            The normalization
    Returns:
        -none
    """
    def _pdf(self, x,
             custom_shape,
             custom_scale,
             norm):
        y = x / custom_scale
        val = (
            y**(custom_shape - 1) * np.exp(-y) /
            Gamma_Func(custom_shape))
        val = val / custom_scale
        norm_func = val / norm
        norm_func[norm_func < 0] = 0.
        return norm_func